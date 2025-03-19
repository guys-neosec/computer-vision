#!/usr/bin/env python3
import argparse

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights

# Settings
num_classes = 3  # class 0 = background, class 1 and class 2 are objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anchors = torch.tensor(
    [
        [0.63066907, 0.39734325],
        [0.84080636, 0.83308687],
        [0.44301002, 0.59707842],
        [0.22850459, 0.20988701],
        [0.8448287, 0.51072606],
        [0.22718694, 0.459403],
        [0.80876488, 0.25842262],
        [0.65927155, 0.65566495],
        [0.54515325, 0.21731671],
        [0.24488351, 0.74377441],
        [0.50000774, 0.85840037],
        [0.40410871, 0.35211747],
    ],
    dtype=torch.float32,
    device=device,
)
num_anchors = anchors.shape[0]


# Model definition (detection head matching training)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_outputs):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEBlock(32),
        )
        self.out_conv = nn.Conv2d(32, num_outputs, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.out_conv(x)


class YOLOFaceDetector(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        mobilenet = models.mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT
        )
        self.backbone = mobilenet.features
        self.conv_reduce = nn.Conv2d(960, 256, kernel_size=1)
        self.head = YOLOHead(256, num_anchors * (5 + num_classes))
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        features = self.conv_reduce(features)
        out = self.head(features)
        return out


# Utility functions for decoding and box conversion
def decode_predictions(pred, grid_x, grid_y, anchor, grid_w, grid_h):
    tx = torch.sigmoid(pred[1])
    ty = torch.sigmoid(pred[2])
    tw = pred[3]
    th = pred[4]
    cx = (grid_x + tx) / grid_w
    cy = (grid_y + ty) / grid_h
    bw = anchor[0] * torch.exp(tw)
    bh = anchor[1] * torch.exp(th)
    return torch.stack([cx, cy, bw, bh])


def box_cxcywh_to_xyxy(box, img_width, img_height):
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return [int(x1), int(y1), int(x2), int(y2)]


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def non_max_suppression(detections, iou_threshold=0.5):
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    final_dets = []
    while detections:
        best = detections.pop(0)
        final_dets.append(best)
        detections = [
            d for d in detections if iou(best["bbox"], d["bbox"]) < iou_threshold
        ]
    return final_dets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for YOLOFaceDetector"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="test_overfit.pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--conf_thresh", type=float, default=0.15, help="Confidence threshold"
    )
    parser.add_argument("--nms_thresh", type=float, default=0.1, help="NMS threshold")
    args = parser.parse_args()

    model = YOLOFaceDetector(num_anchors=num_anchors, num_classes=num_classes).to(
        device
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded and set to evaluation mode.")

    image = cv2.imread(args.image_path)
    if image is None:
        print("Error: Could not load image at", args.image_path)
        exit(1)
    orig_height, orig_width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    transformed = transform(image=image_rgb)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
    batch_size, C, grid_h, grid_w = output.shape
    output = output.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
    output = output.permute(
        0, 3, 4, 1, 2
    ).contiguous()  # shape: [1, grid_h, grid_w, num_anchors, 5+num_classes]

    detections = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                pred = output[0, gy, gx, a, :]
                obj_score = torch.sigmoid(pred[0]).item()
                if obj_score < args.conf_thresh:
                    continue
                class_logits = pred[5:]
                class_prob = torch.softmax(class_logits, dim=-1)
                class_id = int(torch.argmax(class_prob).item())
                if class_id == 0:
                    continue  # skip background boxes
                class_conf = class_prob[class_id].item()
                confidence = obj_score * class_conf
                bbox_norm = decode_predictions(pred, gx, gy, anchors[a], grid_w, grid_h)
                bbox_norm = (
                    bbox_norm.cpu().detach().numpy().tolist()
                )  # [cx, cy, bw, bh]
                abs_bbox = box_cxcywh_to_xyxy(bbox_norm, orig_width, orig_height)
                detections.append(
                    {
                        "bbox": abs_bbox,
                        "class": class_id,
                        "confidence": confidence,
                    }
                )

    final_dets = non_max_suppression(detections, iou_threshold=args.nms_thresh)

    for det in final_dets:
        x1, y1, x2, y2 = det["bbox"]
        label = f"Cls {det['class']}: {det['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
