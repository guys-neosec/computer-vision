import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.ops as ops
import cv2
import numpy as np
import argparse
import math
from loguru import logger

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
anchors = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.float32, device=device)
num_anchors = anchors.shape[0]
num_classes = 1

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
        mobilenet = models.mobilenet_v3_large(pretrained=True)
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

def preprocess_frame(frame, input_size=256):
    orig_h, orig_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size))
    resized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm = (resized - mean) / std
    norm = norm.transpose(2, 0, 1)
    image_tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
    return image_tensor.to(device), (orig_w, orig_h)

def run_inference(model, image_tensor, orig_size, conf_threshold=0.5, nms_threshold=0.4):
    orig_w, orig_h = orig_size
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        bs, _, grid_h, grid_w = output.shape
        output = output.view(bs, num_anchors, (5 + num_classes), grid_h, grid_w)
        output = output.permute(0, 3, 4, 1, 2).contiguous()
        detections = []
        output = output[0]
        for i in range(grid_h):
            for j in range(grid_w):
                for a in range(num_anchors):
                    pred = output[i, j, a]
                    obj_score = torch.sigmoid(pred[0]).item()
                    if obj_score > conf_threshold:
                        box = decode_predictions(pred, j, i, anchors[a], grid_w, grid_h)
                        box = box.cpu().numpy()
                        cx, cy, bw, bh = box
                        cx *= orig_w
                        cy *= orig_h
                        bw *= orig_w
                        bh *= orig_h
                        x1 = cx - bw / 2
                        y1 = cy - bh / 2
                        x2 = cx + bw / 2
                        y2 = cy + bh / 2
                        class_logits = pred[5:]
                        class_prob = F.softmax(class_logits, dim=0)
                        class_conf, class_pred = torch.max(class_prob, dim=0)
                        score = obj_score * class_conf.item()
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'score': score,
                            'class': int(class_pred.item())
                        })
        if len(detections) == 0:
            return []
        boxes = torch.tensor([d['box'] for d in detections], dtype=torch.float32)
        scores = torch.tensor([d['score'] for d in detections], dtype=torch.float32)
        keep = ops.nms(boxes, scores, nms_threshold)
        final_detections = [detections[idx] for idx in keep.tolist()]
        return final_detections

def draw_detections(frame, detections):
    img = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = list(map(int, det['box']))
        label = f"Cls {det['class']}:{det['score']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Face Detector Inference (Webcam/Video)")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam or a file path")
    parser.add_argument("--weights", type=str, default="trained_yolo_face_detector_anchor_nms.pth", help="Path to model weights")
    parser.add_argument("--conf_threshold", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.4, help="NMS threshold")
    parser.add_argument("--input_size", type=int, default=256, help="Input size for the model")
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLOFaceDetector(num_anchors=num_anchors, num_classes=num_classes).to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    logger.info("Model loaded.")
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {args.source}")
        exit(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("No frame received, ending stream.")
            break
        image_tensor, orig_size = preprocess_frame(frame, input_size=args.input_size)
        detections = run_inference(model, image_tensor, orig_size, conf_threshold=args.conf_threshold, nms_threshold=args.nms_threshold)
        if len(detections) > 0:
            for det in detections:
                logger.info(f"Detection: Box={det['box']}, Score={det['score']:.2f}, Class={det['class']}")
        frame_out = draw_detections(frame, detections)
        cv2.imshow("YOLO Face Detector", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
