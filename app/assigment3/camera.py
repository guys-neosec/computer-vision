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

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
num_classes = 2
anchors = torch.tensor([[0.05, 0.05],
                        [0.1, 0.1],
                        [0.2, 0.2],
                        [0.3, 0.3]], dtype=torch.float32, device=device)
num_anchors = anchors.shape[0]
CONF_THRESH = 0.0

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
model.load_state_dict(torch.load("trained_yolo_face_detector_anchor_nms.pth", map_location=device))
model.eval()

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
    out = out[0]
    boxes = []
    for i in range(grid_h):
        for j in range(grid_w):
            for a in range(num_anchors):
                pred = out[i, j, a]
                obj_score = torch.sigmoid(pred[0]).item()
                cls_logits = pred[5:]
                cls_prob = torch.softmax(cls_logits, dim=0)
                cls_label = torch.argmax(cls_prob).item()
                score = obj_score * cls_prob[cls_label].item()
                if score > conf_thresh and cls_label == 1:
                    pred_box = decode_predictions(pred, j, i, anchors[a], grid_w, grid_h)
                    pred_box = pred_box.cpu().numpy()
                    cx, cy, bw, bh = pred_box
                    x1 = int((cx - bw/2) * orig_w)
                    y1 = int((cy - bh/2) * orig_h)
                    x2 = int((cx + bw/2) * orig_w)
                    y2 = int((cy + bh/2) * orig_h)
                    boxes.append((x1, y1, x2, y2, score))
    return boxes

def run_video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = inference_frame(frame, model, transform, conf_thresh=CONF_THRESH)
        for (x1, y1, x2, y2, score) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"person: {score:.2f}", (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow("Video Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video(source=0)
