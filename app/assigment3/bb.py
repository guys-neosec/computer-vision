import json
import os
import math
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import MobileNet_V3_Large_Weights

# Use two classes: 0 for Background, 1 for Face.
num_classes = 2

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
anchors = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.float32, device=device)
num_anchors = anchors.shape[0]


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
        for img in ann_data["images"]:
            img_id = img["id"]
            anns = annotations.get(img_id, [])
            bboxes = [ann["bbox"] for ann in anns]
            # For every annotated face, assign label 1 (face). Anchors without objects will default to background (0).
            labels = [1 for _ in anns]
            self.data.append((img, bboxes, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_info, bboxes, labels = self.data[index]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        image = transformed["image"]
        _, H, W = image.shape
        norm_bboxes = []
        for b in transformed["bboxes"]:
            cx = (b[0] + b[2] / 2) / W
            cy = (b[1] + b[3] / 2) / H
            bw = b[2] / W
            bh = b[3] / H
            norm_bboxes.append([cx, cy, bw, bh])
        norm_bboxes = torch.tensor(norm_bboxes, dtype=torch.float32)
        labels = torch.tensor(transformed["labels"], dtype=torch.long)
        return image, norm_bboxes, labels


def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(boxes), list(labels)


class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_outputs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_outputs, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class YOLOFaceDetector(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone = mobilenet.features
        self.conv_reduce = nn.Conv2d(960, 256, kernel_size=1)
        # Output per anchor: 5 (objectness and bbox parameters) + num_classes (classification)
        self.head = YOLOHead(256, num_anchors * (5 + num_classes))
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        features = self.conv_reduce(features)
        out = self.head(features)
        return out


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
    inter_x1 = torch.max(p_x1, g_x1)
    inter_y1 = torch.max(p_y1, g_y1)
    inter_x2 = torch.min(p_x2, g_x2)
    inter_y2 = torch.min(p_y2, g_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
    area_g = (g_x2 - g_x1) * (g_y2 - g_y1)
    union = area_p + area_g - inter_area + eps
    iou = inter_area / union
    c_x1 = torch.min(p_x1, g_x1)
    c_y1 = torch.min(p_y1, g_y1)
    c_x2 = torch.max(p_x2, g_x2)
    c_y2 = torch.max(p_y2, g_y2)
    area_c = (c_x2 - c_x1) * (c_y2 - c_y1) + eps
    giou = iou - ((area_c - union) / area_c)
    return giou


def decode_predictions(pred, grid_x, grid_y, anchor, grid_w, grid_h):
    # Decode the bbox predictions.
    tx = torch.sigmoid(pred[1])
    ty = torch.sigmoid(pred[2])
    tw = pred[3]
    th = pred[4]
    cx = (grid_x + tx) / grid_w
    cy = (grid_y + ty) / grid_h
    bw = anchor[0] * torch.exp(tw)
    bh = anchor[1] * torch.exp(th)
    return torch.stack([cx, cy, bw, bh])


def build_target(bboxes, labels, grid_h, grid_w, batch_size, device, anchors, num_anchors, num_classes):
    # target_box: (batch, grid_h, grid_w, num_anchors, 5) for objectness and bbox regression.
    target_box = torch.zeros((batch_size, grid_h, grid_w, num_anchors, 5), device=device)
    # target_cls: (batch, grid_h, grid_w, num_anchors) for classification.
    # Initialize with 0 (background). For anchors with a face, we will set the value to 1.
    target_cls = torch.zeros((batch_size, grid_h, grid_w, num_anchors), dtype=torch.long, device=device)
    for i in range(batch_size):
        for face, lab in zip(bboxes[i], labels[i]):
            cx, cy, bw, bh = face.tolist()
            cell_x = int(cx * grid_w)
            cell_y = int(cy * grid_h)
            cell_x = min(cell_x, grid_w - 1)
            cell_y = min(cell_y, grid_h - 1)
            best_iou = 0.0
            best_anchor = 0
            for a in range(num_anchors):
                anchor_w, anchor_h = anchors[a].tolist()
                inter = min(bw, anchor_w) * min(bh, anchor_h)
                union = bw * bh + anchor_w * anchor_h - inter + 1e-6
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = a
            t_x = cx * grid_w - cell_x
            t_y = cy * grid_h - cell_y
            anchor_w, anchor_h = anchors[best_anchor].tolist()
            t_w = math.log(bw / (anchor_w + 1e-6) + 1e-6)
            t_h = math.log(bh / (anchor_h + 1e-6) + 1e-6)
            # Set objectness to 1 and store bbox regression targets.
            target_box[i, cell_y, cell_x, best_anchor, 0] = 1
            target_box[i, cell_y, cell_x, best_anchor, 1] = t_x
            target_box[i, cell_y, cell_x, best_anchor, 2] = t_y
            target_box[i, cell_y, cell_x, best_anchor, 3] = t_w
            target_box[i, cell_y, cell_x, best_anchor, 4] = t_h
            # For a face, set the classification target to 1.
            target_cls[i, cell_y, cell_x, best_anchor] = 1
    return target_box, target_cls


def compute_loss(output, bboxes, labels, device, anchors, num_classes, lambda_bbox=1.0, lambda_cls=1.0):
    batch_size = output.shape[0]
    grid_h, grid_w = output.shape[2], output.shape[3]
    num_anchors = anchors.shape[0]
    # Reshape the output into [B, grid_h, grid_w, num_anchors, 5+num_classes]
    output = output.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
    output = output.permute(0, 3, 4, 1, 2).contiguous()

    target_box, target_cls = build_target(bboxes, labels, grid_h, grid_w, batch_size, device, anchors, num_anchors,
                                          num_classes)
    # Objectness loss using focal loss
    pred_obj = output[..., 0]
    loss_obj = focal_loss(pred_obj, target_box[..., 0])

    # Bounding box regression (GIoU loss) computed over positive anchors only.
    giou_loss_total = 0.0
    count = 0
    for b in range(batch_size):
        inds = (target_box[b, ..., 0] == 1).nonzero(as_tuple=False).cpu()
        for idx in inds:
            gy, gx, anchor_idx = idx.tolist()
            pred = output[b, gy, gx, anchor_idx]
            pred_box = decode_predictions(pred, gx, gy, anchors[anchor_idx], grid_w, grid_h)
            t = target_box[b, gy, gx, anchor_idx]
            gt_cx = (gx + t[1]) / grid_w
            gt_cy = (gy + t[2]) / grid_h
            anchor_wh = anchors[anchor_idx]
            gt_w = anchor_wh[0] * math.exp(t[3].item())
            gt_h = anchor_wh[1] * math.exp(t[4].item())
            gt_w_tensor = torch.as_tensor(gt_w, device=device, dtype=gt_cx.dtype)
            gt_h_tensor = torch.as_tensor(gt_h, device=device, dtype=gt_cx.dtype)
            gt_box = torch.stack([gt_cx.detach(), gt_cy.detach(), gt_w_tensor, gt_h_tensor])
            giou = compute_giou_single(pred_box, gt_box)
            giou_loss_total += (1 - giou)
            count += 1
    giou_loss = giou_loss_total / count if count > 0 else 0.0

    # Classification loss computed over all anchors.
    # The classification predictions have shape [B, grid_h, grid_w, num_anchors, num_classes]
    pred_cls = output[..., 5:]
    cls_loss = F.cross_entropy(pred_cls.view(-1, num_classes), target_cls.view(-1))

    total_loss = loss_obj + lambda_bbox * giou_loss + lambda_cls * cls_loss
    return total_loss, loss_obj, giou_loss, cls_loss


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True


def check_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                logger.warning(f"{name} has no gradient!")
            else:
                logger.info(f"{name} grad norm: {param.grad.norm().item():.4f}")


def train(model, dataloader, optimizer, scheduler, epochs, device, writer, freeze_epochs=5):
    model.train()
    global_step = 0
    for epoch in range(epochs):
        if epoch < freeze_epochs:
            freeze_backbone(model)
        elif epoch == freeze_epochs:
            unfreeze_backbone(model)
            logger.info("Backbone unfrozen for fine-tuning.")
        total_loss_epoch = 0.0
        total_obj_loss = 0.0
        total_giou_loss = 0.0
        total_cls_loss = 0.0
        for imgs, bboxes, labels in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss, obj_loss, giou_loss, cls_loss = compute_loss(out, bboxes, labels, device, anchors, num_classes)
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            total_obj_loss += obj_loss.item()
            total_giou_loss += giou_loss if isinstance(giou_loss, float) else giou_loss.item()
            total_cls_loss += cls_loss if isinstance(cls_loss, float) else cls_loss.item()
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            global_step += 1
        avg_loss = total_loss_epoch / len(dataloader)
        avg_obj = total_obj_loss / len(dataloader)
        avg_giou = total_giou_loss / len(dataloader)
        avg_cls = total_cls_loss / len(dataloader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Total Loss: {avg_loss:.4f}, Obj Loss: {avg_obj:.4f}, Giou Loss: {avg_giou:.4f}, Cls Loss: {avg_cls:.4f}, LR: {current_lr:.6f}")
        writer.add_scalar("Train/EpochLoss", avg_loss, epoch)
        writer.add_scalar("Train/EpochObjLoss", avg_obj, epoch)
        writer.add_scalar("Train/EpochGiouLoss", avg_giou, epoch)
        writer.add_scalar("Train/EpochClsLoss", avg_cls, epoch)
        writer.add_scalar("Train/LearningRate", current_lr, epoch)
        check_gradients(model)


if __name__ == "__main__":
    transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.ColorJitter(p=0.5),
        A.Blur(blur_limit=3, p=0.2),
        A.CoarseDropout(p=0.3),
        A.Affine(shear=(-10, 10), p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="coco", label_fields=["labels"]))

    train_dataset = FaceDataset(
        "/root/projects/computer-vision/train",
        "/root/projects/computer-vision/train/_annotations.coco.json",
        transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=collate_fn)

    model = YOLOFaceDetector(num_anchors=num_anchors, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter()

    train(model, train_loader, optimizer, scheduler, epochs, device, writer, freeze_epochs=80)
    torch.save(model.state_dict(), "trained_yolo_face_detector_anchor_nms.pth")
    writer.close()
