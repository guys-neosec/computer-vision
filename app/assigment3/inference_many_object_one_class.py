import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torchvision import models
from torchvision.ops import nms

# ---------------------------
# Global Settings
# ---------------------------
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
num_classes = 2
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


# ---------------------------
# Model Components
# ---------------------------
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
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
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


# ---------------------------
# Inference Helper Function
# ---------------------------
def decode_predictions(pred, grid_x, grid_y, anchor, grid_w, grid_h):
    """
    Decode a single prediction vector from the output.
    Returns bounding box in normalized coordinates (cx, cy, w, h).
    """
    tx = torch.sigmoid(pred[1])
    ty = torch.sigmoid(pred[2])
    tw = pred[3]
    th = pred[4]
    cx = (grid_x + tx) / grid_w
    cy = (grid_y + ty) / grid_h
    bw = anchor[0] * torch.exp(tw)
    bh = anchor[1] * torch.exp(th)
    return torch.stack([cx, cy, bw, bh])


# ---------------------------
# Preprocessing Transform
# ---------------------------
transform = A.Compose(
    [
        A.Resize(400, 400),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

# ---------------------------
# Load the Trained Model
# ---------------------------
model = YOLOFaceDetector(num_anchors=num_anchors, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("one_object_one_class.pth", map_location=device))
model.eval()

# ---------------------------
# Video Inference and Annotation (with Audio Removal)
# ---------------------------
# Change "input_video.mp4" to your input video file path.
cap = cv2.VideoCapture(
    "/Users/gstrauss/Downloads/Personal/Can't Tonight - SNL - Saturday Night Live (1080p, h264).mp4"
)
if not cap.isOpened():
    print("Failed to open video file.")
    exit()

# Get video properties and set up the VideoWriter
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 20.0
print("Input video FPS:", fps)
ret, frame = cap.read()
if not ret:
    print("No frames to read from the video.")
    cap.release()
    exit()
orig_h, orig_w = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(
    "annotated_output_video.mp4", fourcc, fps, (orig_w, orig_h)
)

# NMS IoU threshold
nms_threshold = 0.5

while ret:
    # Save the original frame for annotation
    orig_frame = frame.copy()

    # Brighten the frame for model inference while preserving original color.
    # Adjust alpha and beta as necessary (alpha=scale, beta=brightness offset).
    brightened_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=40)

    # Resize the brightened frame for inference (while keeping original frame intact for annotations)
    frame_resized = cv2.resize(brightened_frame, (400, 400))
    image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image_rgb)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(input_tensor)

    # Determine grid dimensions from the model output
    _, _, grid_h, grid_w = out.shape
    # Reshape and permute to [1, grid_h, grid_w, num_anchors, 5+num_classes]
    out = out.view(1, num_anchors, 5 + num_classes, grid_h, grid_w)
    out = out.permute(0, 3, 4, 1, 2).contiguous()

    # Get objectness scores after sigmoid
    obj_scores = torch.sigmoid(out[0, ..., 0])
    score_threshold = 0.6
    detections = []  # List: (x1, y1, x2, y2, class, score)

    # Find indices where objectness > threshold
    indices = (obj_scores > score_threshold).nonzero()
    for idx in indices:
        grid_y, grid_x, anchor_idx = idx.tolist()
        pred = out[0, grid_y, grid_x, anchor_idx]
        # Decode bounding box (normalized coordinates)
        box = decode_predictions(
            pred, grid_x, grid_y, anchors[anchor_idx], grid_w, grid_h
        )
        cx, cy, bw, bh = box.cpu().numpy()
        # Convert normalized bbox to pixel coordinates in 400x400 space
        x1 = int((cx - bw / 2) * 400)
        y1 = int((cy - bh / 2) * 400)
        x2 = int((cx + bw / 2) * 400)
        y2 = int((cy + bh / 2) * 400)
        # Get class prediction from logits
        class_scores = out[0, grid_y, grid_x, anchor_idx, 5:]
        predicted_class = torch.argmax(class_scores).item()
        score = obj_scores[grid_y, grid_x, anchor_idx].item()
        # Skip detections with class 0
        if predicted_class == 0:
            continue
        detections.append((x1, y1, x2, y2, predicted_class, score))

    # Apply per-class NMS
    final_detections = []
    if detections:
        detections = np.array(detections)  # shape: (N, 6)
        unique_classes = np.unique(detections[:, 4])
        for cls in unique_classes:
            cls_mask = detections[:, 4] == cls
            cls_dets = detections[cls_mask]
            boxes = torch.tensor(cls_dets[:, :4], dtype=torch.float32)
            scores = torch.tensor(cls_dets[:, 5], dtype=torch.float32)
            keep = nms(boxes, scores, nms_threshold)
            for i in keep:
                final_detections.append(cls_dets[i])
        final_detections = np.array(final_detections)
    else:
        final_detections = np.empty((0, 6))

    # Scale detections from 400x400 back to original frame dimensions and annotate
    scale_x = orig_w / 400.0
    scale_y = orig_h / 400.0
    for x1, y1, x2, y2, cls, score in final_detections:
        x1_orig = int(x1 * scale_x)
        y1_orig = int(y1 * scale_y)
        x2_orig = int(x2 * scale_x)
        y2_orig = int(y2 * scale_y)
        cv2.rectangle(
            orig_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2
        )
        cv2.putText(
            orig_frame,
            f"obj:{score:.2f}",
            (x1_orig, y1_orig - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Write annotated frame to the output video
    out_writer.write(orig_frame)
    # cv2.imshow("Annotated Video", orig_frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

    ret, frame = cap.read()

cap.release()
out_writer.release()
cv2.destroyAllWindows()
