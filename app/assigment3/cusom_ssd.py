import os
from datetime import datetime

import albumentations as A
import cv2
import torch
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from matplotlib import patches
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CocoDetection
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import (
    generalized_box_iou_loss,  # not used in this SSD training version
)

# Global settings
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
WRITER = SummaryWriter(f"logs/ssd_{TIMESTAMP}")
SIZE = 640  # Target image size


# Dataset: converts COCO-format bbox ([x, y, w, h] in pixels) into YOLO normalized format ([cx, cy, w, h])
class SingleObjectCocoDataset(Dataset):
    def __init__(self, root, annFile, transform):
        self.root = root
        self.coco = CocoDetection(root=root, annFile=annFile)
        self.transform = transform
        self.ids = sorted(self.coco.coco.imgs.keys())
        # Only use images with exactly one object (chair) annotation
        self.pictures = [pic for pic in self.coco if len(pic[1]) == 1]

    def __len__(self):
        return len(self.pictures)

    def __getitem__(self, idx):
        coco = self.coco.coco
        img_id = self.pictures[idx][1][0]["image_id"]
        path = coco.loadImgs(img_id)[0]["file_name"]
        img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR_RGB)
        _, targets = self.pictures[idx]
        target = targets[0]
        bbox = target["bbox"]  # COCO format: [x, y, w, h] (pixels)
        # Convert to YOLO format: [cx, cy, w, h] normalized by SIZE
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        bbox_yolo = [cx / SIZE, cy / SIZE, w / SIZE, h / SIZE]
        result = self.transform(image=img, bboxes=[bbox], class_labels=["Chair"])
        return (
            result["image"],
            torch.as_tensor(bbox_yolo, dtype=torch.float32),
            torch.as_tensor(target["category_id"], dtype=torch.long),
        )


# Transformation pipeline
transform = A.Compose(
    [
        A.RandomScale(scale_limit=0.5, p=0.5),
        A.Resize(height=SIZE, width=SIZE, p=1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.3,
        ),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(p=1),
    ],
    p=1.0,
    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["class_labels"],
        clip=True,
    ),
)


# MobileNetV3-SSD model for single-class (chair) detection.
# We set num_classes=2 (background and chair) and choose a small number of anchors.
class MobileNetV3SSD(nn.Module):
    def __init__(self, num_classes=2, num_anchors=3):
        super(MobileNetV3SSD, self).__init__()
        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone = mobilenet.features  # output: (B, 960, H, W)
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Extra layers for multi-scale features
        self.extra_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(960, 512, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
            ],
        )
        self.num_anchors = num_anchors
        # Detection heads for each scale (we use three scales: backbone output and two extras)
        self.cls_heads = nn.ModuleList(
            [
                nn.Conv2d(960, num_anchors * num_classes, kernel_size=3, padding=1),
                nn.Conv2d(512, num_anchors * num_classes, kernel_size=3, padding=1),
                nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, padding=1),
            ],
        )
        self.reg_heads = nn.ModuleList(
            [
                nn.Conv2d(960, num_anchors * 4, kernel_size=3, padding=1),
                nn.Conv2d(512, num_anchors * 4, kernel_size=3, padding=1),
                nn.Conv2d(256, num_anchors * 4, kernel_size=3, padding=1),
            ],
        )

    def forward(self, x, inference=False):
        B = x.size(0)
        features = []
        x = self.backbone(x)
        features.append(x)
        for layer in self.extra_layers:
            x = layer(x)
            features.append(x)
        cls_preds = []
        reg_preds = []
        for f, cls_head, reg_head in zip(
            features,
            self.cls_heads,
            self.reg_heads,
            strict=False,
        ):
            cls_out = cls_head(f)  # shape: (B, num_anchors*num_classes, H, W)
            reg_out = reg_head(f)  # shape: (B, num_anchors*4, H, W)
            # Global average pool each prediction map to get one prediction per scale
            cls_out = F.adaptive_avg_pool2d(cls_out, (1, 1)).view(
                B,
                -1,
            )  # (B, num_anchors*num_classes)
            reg_out = F.adaptive_avg_pool2d(reg_out, (1, 1)).view(
                B,
                -1,
            )  # (B, num_anchors*4)
            cls_preds.append(cls_out)
            reg_preds.append(reg_out)
        # Average predictions over scales
        final_cls_logits = torch.stack(cls_preds, dim=0).mean(
            dim=0,
        )  # (B, num_anchors*num_classes)
        final_reg = torch.stack(reg_preds, dim=0).mean(dim=0)  # (B, num_anchors*4)
        num_classes = 2
        final_cls_logits = final_cls_logits.view(B, self.num_anchors, num_classes).mean(
            dim=1,
        )  # (B, num_classes)
        final_reg = final_reg.view(B, self.num_anchors, 4).mean(dim=1)  # (B, 4)
        # Constrain box predictions to [0,1]
        final_reg = torch.sigmoid(final_reg)
        if inference:
            final_labels = final_cls_logits.argmax(dim=1)
            return final_labels, final_reg
        return final_cls_logits, final_reg


def yolo_to_corners(boxes):
    # boxes: tensor of shape (B,4) in YOLO format: [cx, cy, w, h] (normalized)
    cx, cy, w, h = boxes.unbind(dim=1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def train_epocs(model, optimizer, train_dl, val_dl, epochs=10):
    for epoch in range(epochs):
        model.train()
        total, sum_loss = 0, 0
        for index, (x, y_bb, y_class) in enumerate(train_dl):
            batch = y_class.shape[0]
            x = x.to(device).float()
            y_class = y_class.to(device)
            y_bb = y_bb.to(device).float()  # y_bb in YOLO format (normalized)
            out_class, out_bb = model(x, inference=False)
            loss_class = F.cross_entropy(out_class, y_class, reduction="mean")
            # Convert both predictions and targets from YOLO to corner format
            pred_corners = yolo_to_corners(out_bb)
            gt_corners = yolo_to_corners(y_bb)
            loss_bb = generalized_box_iou_loss(
                pred_corners,
                gt_corners,
                reduction="mean",
            )
            # You can adjust the weight as needed (here, multiplied by 2)
            loss = loss_class + 2 * loss_bb
            WRITER.add_scalar("Loss/train", loss, epoch * len(train_dl) + index + 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss / total
        val_loss, val_acc = val_metrics(model, val_dl)
        print(
            f"{epoch} train_loss {train_loss:.3f} val_loss {val_loss:.3f} val_acc {val_acc:.3f}",
        )
    return sum_loss / total


def val_metrics(model, valid_dl):
    model.eval()
    total, sum_loss, correct = 0, 0, 0
    with torch.no_grad():
        for x, y_bb, y_class in valid_dl:
            batch = y_class.shape[0]
            x = x.to(device).float()
            y_class = y_class.to(device)
            y_bb = y_bb.to(device).float()
            out_class, out_bb = model(x, inference=False)
            loss_class = F.cross_entropy(out_class, y_class, reduction="mean")
            pred_corners = yolo_to_corners(out_bb)
            gt_corners = yolo_to_corners(y_bb)
            loss_bb = generalized_box_iou_loss(
                pred_corners,
                gt_corners,
                reduction="mean",
            )
            loss = loss_class + loss_bb
            _, pred = torch.max(out_class, 1)
            correct += pred.eq(y_class).sum().item()
            sum_loss += loss.item()
            total += batch
    return sum_loss / total, correct / total


def update_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def visualize_predictions(model, dataset, device, num_images=5):
    model.eval()
    for i in range(num_images):
        image, bbox, label = dataset[i]
        input_img = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_label, pred_bbox = model(input_img, inference=True)
        # pred_bbox and bbox are in YOLO format: [cx, cy, w, h] normalized to [0,1]
        pred_bbox = pred_bbox.squeeze(0).cpu().numpy()  # [cx, cy, w, h]
        ground_bbox = bbox.squeeze(0).cpu().numpy()  # [cx, cy, w, h]

        # Convert normalized values to pixel coordinates
        pred_cx, pred_cy, pred_w, pred_h = pred_bbox * SIZE
        ground_cx, ground_cy, ground_w, ground_h = ground_bbox * SIZE

        # Convert center-based boxes to top-left-based boxes
        pred_x = pred_cx - pred_w / 2
        pred_y = pred_cy - pred_h / 2
        ground_x = ground_cx - ground_w / 2
        ground_y = ground_cy - ground_h / 2

        img_np = image.cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        fig, ax = plt.subplots(1)
        rect_predict = patches.Rectangle(
            (pred_x, pred_y),
            pred_w,
            pred_h,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        rect_ground = patches.Rectangle(
            (ground_x, ground_y),
            ground_w,
            ground_h,
            linewidth=2,
            edgecolor="b",
            facecolor="none",
        )
        ax.add_patch(rect_ground)
        ax.add_patch(rect_predict)
        ax.text(
            pred_x,
            pred_y,
            f"Pred: {pred_label.item()}",
            bbox=dict(facecolor="yellow", alpha=0.5),
        )
        ax.imshow(img_np)
        plt.show()


# Setup dataset and data loaders
base_path = "/Users/gstrauss/Downloads/chair detection 2"
train_path = os.path.join(base_path, "train")
valid_path = os.path.join(base_path, "valid")
test_path = os.path.join(base_path, "test")
train_ann = os.path.join(train_path, "_annotations.coco.json")
valid_ann = os.path.join(valid_path, "_annotations.coco.json")
test_ann = os.path.join(test_path, "_annotations.coco.json")

train_dataset = SingleObjectCocoDataset(
    root=train_path,
    annFile=train_ann,
    transform=transform,
)
valid_dataset = SingleObjectCocoDataset(
    root=valid_path,
    annFile=valid_ann,
    transform=transform,
)
test_dataset = SingleObjectCocoDataset(
    root=test_path,
    annFile=test_ann,
    transform=transform,
)

batch_size = 16
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu"),
)
model = MobileNetV3SSD(num_classes=2, num_anchors=20).to(device)
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
)

# Train and evaluate
train_epocs(model, optimizer, train_dl, valid_dl, epochs=50)
# torch.save(model.state_dict(), "mobilenetv3_ssd_chair.pth")
visualize_predictions(model, test_dataset, device, num_images=5)
val_loss, val_acc = val_metrics(model, test_dl)
print(f"Unseen Dataset val_loss {val_loss:.3f} val_acc {val_acc:.3f}")
visualize_predictions(model, train_dataset, device, num_images=5)
