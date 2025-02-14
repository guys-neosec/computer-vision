import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)


# ------------------------------
# Define a custom dataset
# ------------------------------
class FasterRCNNDataset(torch.utils.data.Dataset):
    """
    Wraps the COCO dataset to return images and targets in the format expected
    by torchvision detection models. For face detection, we assign each object
    a label of 1.
    """

    def __init__(self, root: str, ann_file: str, transform=None):
        self.coco = CocoDetection(root, ann_file)
        self.transform = transform

    def __getitem__(self, idx: int):
        img, ann = self.coco[idx]
        # Apply transform (e.g., ToTensor and optionally resizing)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        boxes = []
        labels = []
        for obj in ann:
            # COCO format: [x, y, width, height]
            x, y, w, h = obj["bbox"]
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # Face label
        if len(boxes) == 0:
            # No face found; use empty tensors.
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return img, target

    def __len__(self):
        return len(self.coco)


# ------------------------------
# Data Transforms & DataLoader
# ------------------------------
# You can add resizing here if needed.
data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize((416, 416)),  # Uncomment if you want to resize images
    ],
)

# Replace these paths with your actual train folder and annotation file.
train_root = "/Users/gstrauss/Downloads/Personal/Face Detection.v24-resize416x416-aug3x-traintestsplitonly.coco/train"  # e.g., "/data/face/train"
train_ann_file = "/Users/gstrauss/Downloads/Personal/Face Detection.v24-resize416x416-aug3x-traintestsplitonly.coco/train/_annotations.coco.json"  # e.g., "/data/face/train/_annotations.coco.json"

train_dataset = FasterRCNNDataset(train_root, train_ann_file, transform=data_transform)

# For detection, the DataLoader should return lists of images and targets.
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x, strict=False)),
)

# ------------------------------
# Create the Faster R-CNN Model
# ------------------------------
# Load a pretrained Faster R-CNN with ResNet-50 FPN.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Customize the RPN as desired.
anchor_generator = AnchorGenerator(
    sizes=tuple([(16, 32, 64, 128, 256) for _ in range(5)]),
    # one tuple per feature map
    aspect_ratios=tuple([(0.75, 0.5, 1.25) for _ in range(5)]),
)
rpn_head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
model.rpn = RegionProposalNetwork(
    anchor_generator=anchor_generator,
    head=rpn_head,
    fg_iou_thresh=0.7,
    bg_iou_thresh=0.3,
    batch_size_per_image=48,  # fewer proposals per image
    positive_fraction=0.5,
    pre_nms_top_n=dict(training=200, testing=100),
    post_nms_top_n=dict(training=160, testing=80),
    nms_thresh=0.7,
)

# Replace the box predictor to output 2 classes (background and face).
num_classes = 2  # 0: background, 1: face
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# ------------------------------
# Training Setup
# ------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0005, weight_decay=0.0005)

# ------------------------------
# Simple Training Loop
# ------------------------------
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # When in training mode, the model returns a dict of losses.
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
    print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss / len(train_loader):.4f}")
