"""
https://universe.roboflow.com/maria-team-1/chair-detection-2-14i7o-kzmv1-ldkzl-r7tml
"""

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
    generalized_box_iou_loss,
)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
WRITER = SummaryWriter(f"logs/basic_{TIMESTAMP}")


class SingleObjectCocoDataset(Dataset):
    def __init__(self, root, annFile, transform):
        self.root = root
        self.coco = CocoDetection(root=root, annFile=annFile)
        self.transform = transform
        self.ids = sorted(self.coco.coco.imgs.keys())

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        coco = self.coco.coco
        img_id = self.ids[idx]
        path = coco.loadImgs(img_id)[0]["file_name"]
        img = cv2.imread(str(os.path.join(self.root, path)), cv2.IMREAD_COLOR_RGB)

        _, targets = self.coco[idx]
        target = targets[0]
        bbox = target["bbox"]
        # bbox = [element-1 if element >0 else element for element in bbox]
        result = self.transform(image=img, bboxes=[bbox], class_labels=["Chair"])
        return (
            result["image"],
            torch.as_tensor(result["bboxes"][0], dtype=torch.float32),
            torch.as_tensor(target["category_id"], dtype=torch.long),
        )


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        # Use the pretrained feature extractor (frozen)
        self.features = mobilenet.features
        for param in self.features.parameters():
            param.requires_grad = False

        self.cls_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = mobilenet.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 4),
        )

        self.bb_conv = nn.Sequential(
            nn.Conv2d(in_features, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.bb_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bb_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        features = self.features(x)

        cls_features = self.cls_avgpool(features)
        cls_features = torch.flatten(cls_features, 1)
        cls_out = self.classifier(cls_features)

        bb_features = self.bb_conv(features)
        bb_features = self.bb_pool(bb_features)
        bb_features = torch.flatten(bb_features, 1)
        bb_out = self.bb_fc(bb_features)

        return cls_out, bb_out


def train_epocs(model, optimizer, train_dl, val_dl, epochs=10):
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for index, (x, y_bb, y_class) in enumerate(train_dl):
            batch = y_class.shape[0]
            x = x.to(device).float()
            y_class = y_class.to(device)
            y_bb = y_bb.to(device).float()
            out_class, out_bb = model(x)
            loss_class = F.cross_entropy(out_class, y_class, reduction="mean")
            out_bb_transformed = torch.cat(
                [out_bb[:, :2], out_bb[:, :2] + out_bb[:, 2:]],
                dim=1,
            )
            y_bb_transformed = torch.cat(
                [y_bb[:, :2], y_bb[:, :2] + y_bb[:, 2:]],
                dim=1,
            )
            loss_bb = generalized_box_iou_loss(
                out_bb_transformed,
                y_bb_transformed,
                reduction="mean",
            )
            loss = loss_class + 2 * loss_bb
            WRITER.add_scalar("Loss/train", loss, i * len(train_dl) + index + 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss / total
        val_loss, val_acc = val_metrics(model, valid_dl)
        print(
            f"{i} train_loss {train_loss:.3f} val_loss {val_loss:.3f} val_acc {val_acc:.3f}",
        )
    return sum_loss / total


def val_metrics(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x, y_bb, y_class in valid_dl:
        batch = y_class.shape[0]
        x = x.to(device).float()
        y_class = y_class.to(device)
        y_bb = y_bb.to(device).float()
        out_class, out_bb = model(x)
        loss_class = F.cross_entropy(out_class, y_class, reduction="mean")
        out_bb_transformed = torch.cat(
            [out_bb[:, :2], out_bb[:, :2] + out_bb[:, 2:]],
            dim=1,
        )
        y_bb_transformed = torch.cat([y_bb[:, :2], y_bb[:, :2] + y_bb[:, 2:]], dim=1)
        loss_bb = generalized_box_iou_loss(
            out_bb_transformed,
            y_bb_transformed,
            reduction="mean",
        )
        loss = loss_class + loss_bb
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
    return sum_loss / total, correct / total


def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


transform = A.Compose(
    [
        A.Resize(height=640, width=640, p=1),
        A.ShiftScaleRotate(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
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


def visualize_predictions(model, dataset, device, num_images=5):
    model.eval()
    for i in range(num_images):
        image, bbox, label = dataset[i]
        input_img = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_cls, pred_bbox = model(input_img)
        pred_cls = torch.argmax(pred_cls, dim=1).item()
        pred_bbox = pred_bbox.squeeze(0).cpu().numpy()
        ground_bbox = bbox.squeeze(0).cpu().numpy()
        img_np = image.cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        fig, ax = plt.subplots(1)
        rect_predict = patches.Rectangle(
            (pred_bbox[0], pred_bbox[1]),
            pred_bbox[2],
            pred_bbox[3],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        rect_ground = patches.Rectangle(
            (ground_bbox[0], ground_bbox[1]),
            ground_bbox[2],
            ground_bbox[3],
            linewidth=2,
            edgecolor="b",
            facecolor="none",
        )
        ax.add_patch(rect_ground)
        ax.add_patch(rect_predict)
        ax.text(
            pred_bbox[0],
            pred_bbox[1],
            f"Pred: {pred_cls}",
            bbox=dict(facecolor="yellow", alpha=0.5),
        )
        ax.imshow(img_np)
        plt.show()


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
model = BasicModel().to(device)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters)
train_epocs(model, optimizer, train_dl, valid_dl, epochs=50)
visualize_predictions(model, test_dataset, device, num_images=5)
val_loss, val_acc = val_metrics(model, test_dl)
print(f"Unseen Dataset val_loss {val_loss:.3f} val_acc {val_acc:.3f}")
visualize_predictions(model, train_dataset, device, num_images=5)
