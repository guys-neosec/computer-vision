import json
import os

import albumentations as A
import cv2
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from loguru import logger
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu"),
)


# Dataset
class FaceDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform):
        with open(ann_file) as f:
            ann_data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = {img["id"]: img for img in ann_data["images"]}
        annotations = {}
        for ann in ann_data["annotations"]:
            annotations.setdefault(ann["image_id"], []).append(ann)
        self.data = []
        for img_id, anns in annotations.items():
            if len(anns) == 1:
                img_info = self.imgs[img_id]
                bbox = anns[0]["bbox"]
                self.data.append((img_info, bbox))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_info, bbox = self.data[index]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image, bboxes=[bbox], labels=[1])
        image = transformed["image"]
        b = transformed["bboxes"][0]
        _, H, W = image.shape
        cx = (b[0] + b[2] / 2) / W
        cy = (b[1] + b[3] / 2) / H
        bw = b[2] / W
        bh = b[3] / H
        bbox_norm = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
        return image, bbox_norm


# Model components
class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_outputs=5):
        super(YOLOHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_outputs, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class YOLOFaceDetector(nn.Module):
    def __init__(self):
        super(YOLOFaceDetector, self).__init__()
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.backbone = mobilenet.features
        self.conv_reduce = nn.Conv2d(960, 256, kernel_size=1)
        self.head = YOLOHead(256, 5)

    def forward(self, x):
        features = self.backbone(x)
        features = self.conv_reduce(features)
        out = self.head(features)
        return out


# Helper functions for loss
def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    prob = torch.sigmoid(inputs)
    p_t = targets * prob + (1 - targets) * (1 - prob)
    loss = alpha * (1 - p_t) ** gamma * bce_loss
    return loss.mean()


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def compute_giou_single(pred_box, gt_box, eps=1e-6):
    p_x1, p_y1, p_x2, p_y2 = box_cxcywh_to_xyxy(pred_box)
    g_x1, g_y1, g_x2, g_y2 = box_cxcywh_to_xyxy(gt_box)
    inter_x1 = max(p_x1, g_x1)
    inter_y1 = max(p_y1, g_y1)
    inter_x2 = min(p_x2, g_x2)
    inter_y2 = min(p_y2, g_y2)
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
    area_g = (g_x2 - g_x1) * (g_y2 - g_y1)
    union = area_p + area_g - inter_area + eps
    iou = inter_area / union
    # Smallest enclosing box
    c_x1 = min(p_x1, g_x1)
    c_y1 = min(p_y1, g_y1)
    c_x2 = max(p_x2, g_x2)
    c_y2 = max(p_y2, g_y2)
    area_c = (c_x2 - c_x1) * (c_y2 - c_y1) + eps
    giou = iou - ((area_c - union) / area_c)
    return giou


def build_target(bboxes, grid_h, grid_w, batch_size, device):
    target = torch.zeros((batch_size, 5, grid_h, grid_w), device=device)
    for i in range(batch_size):
        gt = bboxes[i]
        cell_x = int(gt[0].item() * grid_w)
        cell_y = int(gt[1].item() * grid_h)
        # Make sure cell indices are within bounds
        cell_x = min(cell_x, grid_w - 1)
        cell_y = min(cell_y, grid_h - 1)
        target[i, 0, cell_y, cell_x] = 1
        target[i, 1, cell_y, cell_x] = gt[0].item() * grid_w - cell_x
        target[i, 2, cell_y, cell_x] = gt[1].item() * grid_h - cell_y
        target[i, 3, cell_y, cell_x] = gt[2]
        target[i, 4, cell_y, cell_x] = gt[3]
    return target


def compute_loss(
    output,
    bboxes,
    device,
    lambda_bbox=1.0,
    current_epoch=0,
    warmup_epochs=20,
):
    batch_size = output.shape[0]
    grid_h, grid_w = output.shape[2], output.shape[3]
    target = build_target(bboxes, grid_h, grid_w, batch_size, device)

    # Classification loss with focal loss
    pred_conf = output[:, 0, :, :]
    loss_conf = focal_loss(pred_conf, target[:, 0, :, :])

    # Bounding box losses
    giou_loss_total = 0.0
    l1_loss_total = 0.0
    count = 0
    for i in range(batch_size):
        inds = (target[i, 0, :, :] == 1).nonzero(as_tuple=False)
        if inds.shape[0] == 0:
            continue
        for idx in inds:
            y_ind, x_ind = idx
            pred = output[i, 1:, y_ind, x_ind]
            pred_x = torch.sigmoid(pred[0])
            pred_y = torch.sigmoid(pred[1])
            pred_w = torch.sigmoid(pred[2])
            pred_h = torch.sigmoid(pred[3])
            cell_x = x_ind.item()
            cell_y = y_ind.item()
            pred_cx = (cell_x + pred_x) / grid_w
            pred_cy = (cell_y + pred_y) / grid_h
            target_x = target[i, 1, y_ind, x_ind]
            target_y = target[i, 2, y_ind, x_ind]
            target_w = target[i, 3, y_ind, x_ind]
            target_h = target[i, 4, y_ind, x_ind]
            gt_cx = (cell_x + target_x) / grid_w
            gt_cy = (cell_y + target_y) / grid_h
            pred_box = torch.stack([pred_cx, pred_cy, pred_w, pred_h])
            gt_box = torch.stack([gt_cx, gt_cy, target_w, target_h])
            # GIoU loss component: 1 - giou
            giou = compute_giou_single(pred_box, gt_box)
            # Clamp giou to avoid extremely large loss when negative:
            giou = torch.clamp(giou, min=0)
            giou_loss_total += 1 - giou
            # Smooth L1 loss on box coordinates
            l1_loss_total += F.smooth_l1_loss(pred_box, gt_box, reduction="mean")
            count += 1

    if count > 0:
        giou_loss = giou_loss_total / count
        l1_loss = l1_loss_total / count
    else:
        giou_loss = 0.0
        l1_loss = 0.0

    # Blending factor: gradually shift from L1 to GIoU over warmup_epochs
    alpha = min(1.0, current_epoch / warmup_epochs)
    bbox_loss = (1 - alpha) * l1_loss + alpha * giou_loss

    total_loss = loss_conf + lambda_bbox * bbox_loss

    # Return total loss and individual components for logging
    return total_loss, loss_conf, giou_loss, l1_loss


# Functions to freeze/unfreeze backbone for fine-tuning
def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True


def check_gradients(model):
    # Log gradient norms for key layers to ensure gradients are flowing
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                logger.warning(f"{name} has no gradient!")
            else:
                grad_norm = param.grad.norm().item()
                logger.info(f"{name} grad norm: {grad_norm:.4f}")


# Training function with scheduler, logging, gradient checking and backbone freezing/unfreezing
def train(model, dataloader, optimizer, scheduler, epochs, device, freeze_epochs=5):
    model.train()
    for epoch in range(epochs):
        # Freeze backbone for initial epochs
        if epoch < freeze_epochs:
            freeze_backbone(model)
        elif epoch == freeze_epochs:
            unfreeze_backbone(model)
            logger.info("Backbone unfrozen for fine-tuning.")
        total_loss_epoch = 0.0
        total_loss_conf = 0.0
        total_giou_loss = 0.0
        total_l1_loss = 0.0
        for imgs, bboxes in dataloader:
            imgs = imgs.to(device)
            bboxes = bboxes.to(device)
            optimizer.zero_grad()
            loss, loss_conf, giou_loss, l1_loss = compute_loss(
                output=model(imgs),
                bboxes=bboxes,
                device=device,
                current_epoch=epoch,
            )
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            total_loss_conf += loss_conf.item()
            total_giou_loss += giou_loss.item()
            total_l1_loss += l1_loss.item()

        avg_loss = total_loss_epoch / len(dataloader)
        avg_conf = total_loss_conf / len(dataloader)
        avg_giou = total_giou_loss / len(dataloader)
        avg_l1 = total_l1_loss / len(dataloader)
        scheduler.step()
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Total Loss: {avg_loss:.4f}, "
            f"Conf Loss: {avg_conf:.4f}, GIoU Loss: {avg_giou:.4f}, L1 Loss: {avg_l1:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}",
        )

        # Check gradients of key layers after each epoch
        check_gradients(model)


if __name__ == "__main__":
    # Data augmentation with an additional random rotation
    transform = A.Compose(
        [
            A.Resize(256, 256),
            # A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
    )

    train_dataset = FaceDataset(
        "/Users/gstrauss/Downloads/Personal/Face Detection.v24-resize416x416-aug3x-traintestsplitonly.coco/train",
        "/Users/gstrauss/Downloads/Personal/Face Detection.v24-resize416x416-aug3x-traintestsplitonly.coco/train/_annotations.coco.json",
        transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = YOLOFaceDetector().to(device)
    # Reduced learning rate to 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Use freeze_epochs=5 to unfreeze the backbone earlier
    train(model, train_loader, optimizer, scheduler, epochs, device, freeze_epochs=5)

    # Save the trained model
    torch.save(model.state_dict(), "trained_yolo_face_detector.pth")
