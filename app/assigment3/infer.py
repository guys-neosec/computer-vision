import os
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

num_classes = 3

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
anchors = torch.tensor([[0.63066907 ,0.39734325],
                        [0.84080636 ,0.83308687],
                        [0.44301002 ,0.59707842],
                        [0.22850459 ,0.20988701],
                        [0.8448287  ,0.51072606],
                        [0.22718694 ,0.459403  ],
                        [0.80876488 ,0.25842262],
                        [0.65927155 ,0.65566495],
                        [0.54515325 ,0.21731671],
                        [0.24488351 ,0.74377441],
                        [0.50000774 ,0.85840037],
                        [0.40410871 ,0.35211747]], dtype=torch.float32, device=device)
num_anchors = anchors.shape[0]
CONF_THRESH = 0.0


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_outputs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEBlock(32),  # Adding SE attention module
            nn.Conv2d(32, num_outputs, kernel_size=1)
        )
    def forward(self, x):
        return self.conv(x)

class YOLOFaceDetector(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
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

transform = A.Compose([
    A.Resize(400, 400),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

model = YOLOFaceDetector(num_anchors, num_classes).to(device)
model.load_state_dict(torch.load("face_banana_model_10.pth", map_location=device))
model.eval()
import torchvision.ops as ops

def inference_frame(frame, model, transform, conf_thresh=CONF_THRESH):
    orig_h, orig_w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed = transform(image=frame_rgb)
    inp = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)

    B, C, grid_h, grid_w = out.shape
    out = out.view(B, num_anchors, 5 + num_classes, grid_h, grid_w)
    out = out.permute(0, 3, 4, 1, 2).contiguous()
    out = out[0]  # shape [grid_h, grid_w, num_anchors, 5+num_classes]

    # Collect raw detections (x1, y1, x2, y2, score, cls_label)
    boxes_raw = []
    for i in range(grid_h):
        for j in range(grid_w):
            for a in range(num_anchors):
                pred = out[i, j, a]
                obj_score = torch.sigmoid(pred[0]).item()

                # Classification probability
                cls_logits = pred[5:]
                cls_prob = torch.softmax(cls_logits, dim=0)
                cls_label = torch.argmax(cls_prob).item()
                score = obj_score * cls_prob[cls_label].item()
                if score > conf_thresh and cls_label!=0:
                    pred_box = decode_predictions(pred, j, i, anchors[a], grid_w, grid_h)
                    pred_box = pred_box.cpu().numpy()
                    cx, cy, bw, bh = pred_box
                    x1 = int((cx - bw/2) * orig_w)
                    y1 = int((cy - bh/2) * orig_h)
                    x2 = int((cx + bw/2) * orig_w)
                    y2 = int((cy + bh/2) * orig_h)
                    boxes_raw.append((x1, y1, x2, y2, score, cls_label))

    # ------------------------------------------------
    # Now apply NMS using torchvision
    # ------------------------------------------------
    boxes = []
    if len(boxes_raw) > 0:
        boxes_xyxy = torch.tensor([b[:4] for b in boxes_raw], dtype=torch.float32)
        scores = torch.tensor([b[4] for b in boxes_raw], dtype=torch.float32)
        keep_indices = ops.nms(boxes_xyxy, scores, iou_threshold=0.4)  # adjust IoU as needed

        for idx in keep_indices:
            # Rebuild the detection tuple with the original info
            x1, y1, x2, y2, score, cls_label = boxes_raw[idx]
            boxes.append((x1, y1, x2, y2, score, cls_label))

    return boxes


def run_video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    class_names = {0: "background", 1: "face", 2: "car", 3: "dog", 4: "cat", 5: "tree", 6: "building", 7: "unknown"}  # Update this
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = inference_frame(frame, model, transform, conf_thresh=CONF_THRESH)
        for (x1, y1, x2, y2, score,cls_label) in boxes:
            class_name = class_names.get(cls_label, "unknown")  # Get class name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{cls_label}: {score:.2f}", (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow("Video Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video("parlament.webm")
