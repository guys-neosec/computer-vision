import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torchvision import models

# Device and anchors (must match training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anchors = torch.tensor(
    [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
    dtype=torch.float32,
    device=device,
)
num_anchors = anchors.shape[0]


# Model definition (must match training)
class YOLOHead(torch.nn.Module):
    def __init__(self, in_channels, num_outputs):
        super(YOLOHead, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, num_outputs, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class YOLOFaceDetector(torch.nn.Module):
    def __init__(self, num_anchors):
        super(YOLOFaceDetector, self).__init__()
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.backbone = mobilenet.features
        self.conv_reduce = torch.nn.Conv2d(960, 256, kernel_size=1)
        self.head = YOLOHead(256, num_anchors * 5)
        self.num_anchors = num_anchors

    def forward(self, x):
        features = self.backbone(x)
        features = self.conv_reduce(features)
        out = self.head(features)
        return out


# Load the trained model
model = YOLOFaceDetector(num_anchors=num_anchors).to(device)
model.load_state_dict(
    torch.load("trained_yolo_face_detector_anchor_nms.pth", map_location=device),
)
model.eval()

# Inference transform (must match training input size and normalization)
inference_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
)


# Helper functions
def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def compute_giou_single(pred_box, gt_box, eps=1e-6):
    p_x1, p_y1, p_x2, p_y2 = box_cxcywh_to_xyxy(pred_box)
    g_x1, g_y1, g_x2, g_y2 = box_cxcywh_to_xyxy(gt_box)
    inter_x1 = torch.max(p_x1, g_x1)
    inter_y1 = torch.max(p_y1, g_y1)
    inter_x2 = torch.min(p_x2, g_x2)
    inter_y2 = torch.min(p_y2, g_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1,
        min=0,
    )
    area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
    area_g = (g_x2 - g_x1) * (g_y2 - g_y1)
    union = area_p + area_g - inter_area + eps
    iou = inter_area / union
    c_x1 = torch.min(p_x1, g_x1)
    c_y1 = torch.min(p_y1, g_y1)
    c_x2 = torch.max(p_x2, g_x2)
    c_y2 = torch.max(p_y2, g_y2)
    area_c = (c_x2 - c_x1) * (c_y2 - c_y1) + eps
    giou = iou - ((area_c - union) / area_c)
    return giou


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


def non_max_suppression(predictions, iou_thresh=0.5):
    if len(predictions) == 0:
        return []
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    final_preds = []
    while predictions:
        best = predictions.pop(0)
        final_preds.append(best)
        predictions = [
            pred
            for pred in predictions
            if compute_giou_single(
                torch.as_tensor(best[:4]),
                torch.as_tensor(pred[:4]),
            ).item()
            < iou_thresh
        ]
    return final_preds


# Inference loop from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_frame = frame.copy()
    orig_h, orig_w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed = inference_transform(image=rgb_frame)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    bs, ch, grid_h, grid_w = output.shape
    output = output.view(bs, num_anchors, 5, grid_h, grid_w)
    output = output.permute(0, 3, 4, 1, 2).contiguous()
    output = output[0]

    predictions = []
    conf_thresh = 0.3  # Lower threshold to capture more detections

    for i in range(grid_h):
        for j in range(grid_w):
            for a in range(num_anchors):
                pred = output[i, j, a]
                conf = torch.sigmoid(pred[0]).item()
                if conf > conf_thresh:
                    box = decode_predictions(pred, j, i, anchors[a], grid_w, grid_h)
                    predictions.append(
                        [
                            box[0].item(),
                            box[1].item(),
                            box[2].item(),
                            box[3].item(),
                            conf,
                        ],
                    )

    final_preds = non_max_suppression(predictions, iou_thresh=0.5)

    # Draw detections on the original frame.
    for det in final_preds:
        cx, cy, bw, bh, conf = det
        x = int((cx - bw / 2) * 256)
        y = int((cy - bh / 2) * 256)
        w = int(bw * 256)
        h = int(bh * 256)
        scale_x = orig_w / 256
        scale_y = orig_h / 256
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)
        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            orig_frame,
            f"{conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Webcam Face Detection", orig_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
