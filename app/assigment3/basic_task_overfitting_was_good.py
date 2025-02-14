from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from matplotlib import patches
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CocoDetection
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import generalized_box_iou_loss

SIZE: int = 224
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


def get_train_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(SIZE, SIZE, p=1),
            A.Normalize(normalization="min_max_per_channel"),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels"],
            clip=True,
        ),
    )


def get_test_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(SIZE, SIZE, p=1),
            A.Normalize(),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="coco",
            clip=True,
            label_fields=["class_labels"],
        ),
    )


class SingleBoxDataset(Dataset):
    def __init__(self, root: str, ann_file: str, transform: A.Compose) -> None:
        self.coco = CocoDetection(root=root, annFile=ann_file)
        self.transform = transform
        self.data = [c for c in self.coco if len(c[1]) > 0]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, targets = self.data[idx]
        bbox = targets[0]["bbox"]
        class_labels = ["Chair"]
        transformed: dict[str, Any] = self.transform(
            image=np.array(img),
            bboxes=[bbox],
            class_labels=class_labels,
        )
        boxes_tensor = (
            torch.tensor(
                transformed["bboxes"],
                dtype=torch.float32,
            )
            / SIZE
        )
        return transformed["image"], boxes_tensor


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    images, boxes = zip(*batch, strict=False)
    images = torch.stack(images, dim=0)
    # Since each sample returns a tensor of shape (1, 4), we concatenate them along dim=0.
    boxes = torch.cat(boxes, dim=0)
    return images, boxes


class SingleBoxMobileNetV3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features
        # Freeze backbone parameters
        for p in self.features.parameters():
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            # In original 4096
            nn.Linear(960, 496),
            # nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        out = self.fc(x)  # (B,6)
        return out


def unfreeze_backbone_blocks(
    model: SingleBoxMobileNetV3,
    start_block: int = 10,
    end_block: int = 16,
) -> None:
    for name, param in model.features.named_parameters():
        for i in range(start_block, end_block + 1):
            if f"block_{i}" in name:
                param.requires_grad = True


def coco_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts boxes from COCO format [x, y, w, h] to corner format [x1, y1, x2, y2].
    """
    if boxes.dim() == 1:
        x, y, w, h = boxes
        return torch.tensor(
            [x, y, x + w, y + h],
            device=boxes.device,
        )
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([x, y, x + w, y + h], dim=1)


def train_full_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epochs: int = 50,
    lr: float = 1e-3,
) -> None:
    model.to(device)
    model.train()
    optimizer = optim.Adam(
        filter(lambda layer: layer.requires_grad, model.parameters()),
        lr=lr,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        verbose=True,
    )

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, ground_bbox in dataloader:
            images = images.to(device)

            # Use yolo_to_corners for predictions since the model outputs YOLO format.
            box_preds = yolo_to_corners(model(images)).to(device) * SIZE
            # Use coco_to_corners for ground truth boxes.
            gt_boxes = coco_to_corners(ground_bbox).to(device) * SIZE

            loss_iou = generalized_box_iou_loss(box_preds, gt_boxes, reduction="mean")
            loss_l1 = F.smooth_l1_loss(box_preds, gt_boxes)
            loss = loss_iou + loss_l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_epoch_loss)
            writer.add_scalar("Batch/Loss", loss.item(), global_step)
            global_step += 1

        avg_epoch_loss = epoch_loss / len(dataloader)
        writer.add_scalar("Epoch/Train_Loss", avg_epoch_loss, epoch)
        print(f"Epoch {epoch + 1}/{epochs} Loss: {avg_epoch_loss:.4f}")


def yolo_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from YOLO format [center_x, center_y, width, height]
    to corner format [x1, y1, x2, y2].

    Args:
        boxes (torch.Tensor): A tensor of shape (4,) or (N, 4) in YOLO format.

    Returns:
        torch.Tensor: A tensor of the same shape as input, with boxes in corner format.
    """
    if boxes.ndim == 1:
        # Single bounding box case.
        cx, cy, w, h = boxes
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.tensor([x1, y1, x2, y2], device=boxes.device, dtype=boxes.dtype)
    if boxes.ndim == 2:
        # Batch of bounding boxes.
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    raise ValueError("Input tensor must have shape (4,) or (N, 4).")


def visualize_sample(
    model: nn.Module,
    sample: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> None:
    model.eval()
    image, boxes_tensor = sample
    input_img = image.unsqueeze(0).to(device)
    with torch.no_grad():
        # Convert model output using yolo_to_corners.
        pred_box = yolo_to_corners(model(input_img))[0].cpu() * SIZE

    # Convert ground truth using coco_to_corners.
    gt_corners = coco_to_corners(boxes_tensor[0]) * SIZE

    img_np = image.cpu().numpy().transpose(1, 2, 0)
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_np)

    # Draw predicted bounding box (red)
    px1, py1, px2, py2 = pred_box.tolist()
    rect_pred = patches.Rectangle(
        (px1, py1),
        px2 - px1,
        py2 - py1,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect_pred)

    # Draw ground-truth bounding box (blue)
    gx1, gy1, gx2, gy2 = gt_corners.tolist()
    rect_gt = patches.Rectangle(
        (gx1, gy1),
        gx2 - gx1,
        gy2 - gy1,
        linewidth=2,
        edgecolor="b",
        facecolor="none",
    )
    ax.add_patch(rect_gt)
    plt.show()


def visualize_test_set(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    indices: list[int],
) -> None:
    for idx in indices:
        sample = dataset[idx]
        visualize_sample(model, sample, device)


def main(overfit_one_image: bool = False) -> None:
    base_path = "/Users/gstrauss/Downloads/Personal/Face Detection.v24-resize416x416-aug3x-traintestsplitonly.coco"
    train_path = Path(base_path) / "train"
    train_ann = train_path / "_annotations.coco.json"
    test_path = Path(base_path) / "test"
    test_ann = test_path / "_annotations.coco.json"

    train_dataset = SingleBoxDataset(
        root=str(train_path),
        ann_file=str(train_ann),
        transform=get_train_transform(),
    )
    test_dataset = SingleBoxDataset(
        root=str(test_path),
        ann_file=str(test_ann),
        transform=get_test_transform(),
    )

    if overfit_one_image:
        train_dataset = Subset(train_dataset, [3])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32 if overfit_one_image else 16,
        shuffle=True,
        num_workers=0 if overfit_one_image else 4,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    model = SingleBoxMobileNetV3()
    # unfreeze_backbone_blocks(model, start_block=10, end_block=16)

    model = model.to(DEVICE)

    writer = SummaryWriter(log_dir="runs/face_experiment")
    train_full_dataset(
        model,
        train_loader,
        DEVICE,
        writer,
        epochs=500 if overfit_one_image else 200,
        lr=1e-3,
    )
    # writer.add_scalar("Epoch/Validation_Loss", val_loss)
    test_indices = [0]
    visualize_test_set(model, train_dataset, DEVICE, test_indices)

    writer.close()


if __name__ == "__main__":
    main(overfit_one_image=False)
