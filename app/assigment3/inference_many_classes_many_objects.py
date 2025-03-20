import argparse
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.ops as ops

# Device configuration
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# --- Model Definitions ---
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
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            SEBlock(in_channels // 4)
        )
        self.out_conv = nn.Conv2d(in_channels // 4, num_outputs, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.out_conv(x)


class YOLOFaceDetector(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        mobilenet = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone = mobilenet.features
        self.conv_reduce = nn.Conv2d(960, 384, kernel_size=1)
        # Head outputs 5 bbox attributes + num_classes scores per anchor.
        self.head = YOLOHead(384, num_anchors * (5 + num_classes))
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        features = self.conv_reduce(features)
        out = self.head(features)
        return out


# --- Vectorized Inference Function ---
def inference_vectorized(model, image, anchors, num_classes, conf_threshold=0.5,
                         iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
    # Expected output shape: (1, num_anchors*(5+num_classes), grid_h, grid_w)
    batch_size, _, grid_h, grid_w = output.shape
    num_anchors = anchors.shape[0]
    # Reshape and permute to [grid_h, grid_w, num_anchors, 5+num_classes]
    output = output.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
    output = output.permute(0, 3, 4, 1, 2).contiguous()[
        0]  # shape: (grid_h, grid_w, num_anchors, 5+num_classes)

    # Compute objectness scores and create a mask
    obj_scores = torch.sigmoid(output[..., 0])  # shape: (grid_h, grid_w, num_anchors)
    mask = obj_scores >= conf_threshold
    if mask.sum() == 0:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,),
                                                                   dtype=torch.long)

    # Create grid indices for each cell
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h, device=device),
                                    torch.arange(grid_w, device=device), indexing='ij')
    grid_x = grid_x.unsqueeze(-1).expand(grid_h, grid_w,
                                         num_anchors)  # shape: (grid_h, grid_w, num_anchors)
    grid_y = grid_y.unsqueeze(-1).expand(grid_h, grid_w, num_anchors)

    # Flatten all predictions and associated grid indices
    output_flat = output.view(-1, 5 + num_classes)
    obj_scores_flat = obj_scores.view(-1)
    grid_x_flat = grid_x.reshape(-1).float()
    grid_y_flat = grid_y.reshape(-1).float()
    mask_flat = mask.view(-1)

    valid_preds = output_flat[mask_flat]  # (N_valid, 5+num_classes)
    valid_scores = obj_scores_flat[mask_flat]  # (N_valid,)
    # Recover anchor index from mask indices using nonzero indices
    nonzero_indices = mask.nonzero(
        as_tuple=False)  # (N_valid, 3) -> (i, j, anchor_index)
    a_idx = nonzero_indices[:, 2]
    selected_anchors = anchors[a_idx]  # (N_valid, 2)
    grid_x_valid = grid_x_flat[mask_flat]  # (N_valid,)
    grid_y_valid = grid_y_flat[mask_flat]  # (N_valid,)

    # Compute bounding box parameters in a vectorized way.
    tx = torch.sigmoid(valid_preds[:, 1])
    ty = torch.sigmoid(valid_preds[:, 2])
    tw = valid_preds[:, 3]
    th = valid_preds[:, 4]
    # Normalized center coordinates relative to inference image size.
    cx = (grid_x_valid + tx) / grid_w
    cy = (grid_y_valid + ty) / grid_h
    bw = selected_anchors[:, 0] * torch.exp(tw)
    bh = selected_anchors[:, 1] * torch.exp(th)

    boxes = torch.stack([cx, cy, bw, bh], dim=1)  # normalized [cx, cy, bw, bh]

    # Convert boxes to xyxy format (still normalized) for NMS.
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # Get class predictions
    cls_probs = valid_preds[:, 5:]  # (N_valid, num_classes)
    pred_classes = torch.argmax(cls_probs, dim=1)

    # Skip detections with class 6 (Background)
    valid_class_mask = pred_classes != 6
    boxes_xyxy = boxes_xyxy[valid_class_mask]
    boxes = boxes[valid_class_mask]
    valid_scores = valid_scores[valid_class_mask]
    pred_classes = pred_classes[valid_class_mask]

    if boxes_xyxy.shape[0] == 0:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,),
                                                                   dtype=torch.long)

    # Apply Non-Max Suppression on normalized coordinates
    keep = ops.nms(boxes_xyxy, valid_scores, iou_threshold)
    boxes_final = boxes[keep]  # Still normalized [cx, cy, bw, bh]
    scores_final = valid_scores[keep]
    classes_final = pred_classes[keep]
    return boxes_final, scores_final, classes_final


def cxcywh_to_xyxy(box, img_width, img_height):
    """Convert normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return int(x1), int(y1), int(x2), int(y2)


# --- Label and Color Mappings ---
LABEL_NAMES = {
    0: "Ball",
    1: "ball",
    2: "made",
    3: "person",
    4: "rim",
    5: "shoot",
    6: "unknown"  # Background
}

LABEL_COLORS = {
    0: (0, 0, 255),  # Red
    1: (0, 255, 0),  # Green
    2: (255, 0, 0),  # Blue
    3: (0, 255, 255),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (255, 255, 0),  # Cyan
    6: (128, 128, 128)  # Gray
}


# --- Main Video Inference Script ---
def main(args):
    # Define inference transform: resize to fixed size for inference.
    inference_size = (400, 400)
    inference_transform = A.Compose([
        A.Resize(inference_size[0], inference_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Open input video
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Get original video properties (we will output at original resolution)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (orig_w, orig_h))

    # Load model and weights
    anchors = torch.tensor([[0.0345, 0.0513],
                            [0.4671, 0.8831],
                            [0.1763, 0.4546],
                            [0.8256, 0.8913],
                            [0.4828, 0.5786],
                            [0.1012, 0.1713],
                            [0.2491, 0.6914],
                            [0.2815, 0.2879],
                            [0.0883, 0.2786]], device=device)
    num_classes = 7
    num_anchors = anchors.shape[0]
    model = YOLOFaceDetector(num_anchors=num_anchors, num_classes=num_classes).to(
        device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig_frame = frame.copy()  # Original frame for drawing boxes
        # Prepare the frame for inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = inference_transform(image=frame_rgb)
        input_tensor = transformed['image']

        # Run vectorized inference
        boxes, scores, pred_classes = inference_vectorized(model, input_tensor, anchors,
                                                           num_classes,
                                                           conf_threshold=args.conf_threshold,
                                                           iou_threshold=args.iou_threshold)
        # Draw detections on the original frame
        for box, score, label in zip(boxes, scores, pred_classes):
            # Convert normalized box (from inference size) to original frame coordinates.
            x1, y1, x2, y2 = cxcywh_to_xyxy(box.cpu().numpy(), orig_w, orig_h)
            color = LABEL_COLORS.get(int(label.item()), (255, 255, 255))
            label_text = f"{LABEL_NAMES.get(int(label.item()), 'unknown')}:{score:.2f}"
            cv2.rectangle(orig_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(orig_frame, label_text, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(orig_frame)
        frame_count += 1
        print(f"Processed frame: {frame_count}", end='\r')

    cap.release()
    out.release()
    print("\nProcessing complete. Output saved to", args.output_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video inference script with vectorized operations. "
                    "Resizes only for inference but saves the original resolution. "
                    "Detections with class 6 (Background) are skipped.")
    parser.add_argument("--input_video", type=str, required=True,
                        help="Path to the input video (mp4).")
    parser.add_argument("--output_video", type=str, required=True,
                        help="Path to save the output video (mp4).")
    parser.add_argument("--weights", type=str, default="basketball_dynamic_anchors.pth",
                        help="Path to the model weights file.")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Confidence threshold for detections.")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for non-max suppression.")
    args = parser.parse_args()
    main(args)
