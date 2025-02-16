import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torchvision import models


# Define the YOLOHead and YOLOFaceDetector as in your training script.
class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_outputs=5):
        super(YOLOHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_outputs, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class YOLOFaceDetector(nn.Module):
    def __init__(self):
        super(YOLOFaceDetector, self).__init__()
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.backbone = mobilenet.features
        self.conv_reduce = nn.Conv2d(960, 256, kernel_size=1)
        self.head = YOLOHead(256, 5)

    def forward(self, x):
        features = self.backbone(x)
        features = self.conv_reduce(features)
        out = self.head(features)
        return out


# Simple decode_predictions function to extract boxes from model output.
def decode_predictions(output, conf_threshold, device):
    batch_boxes = []
    batch_scores = []
    batch_size = output.shape[0]
    grid_h, grid_w = output.shape[2], output.shape[3]
    for i in range(batch_size):
        pred = output[i]
        conf = torch.sigmoid(pred[0])
        boxes = []
        scores = []
        for y in range(grid_h):
            for x in range(grid_w):
                score = conf[y, x].item()
                if score > conf_threshold:
                    cell_pred = pred[1:, y, x]
                    tx = torch.sigmoid(cell_pred[0]).item()
                    ty = torch.sigmoid(cell_pred[1]).item()
                    tw = torch.sigmoid(cell_pred[2]).item()
                    th = torch.sigmoid(cell_pred[3]).item()
                    cx = (x + tx) / grid_w
                    cy = (y + ty) / grid_h
                    boxes.append([cx, cy, tw, th])
                    scores.append(score)
        if boxes:
            boxes = torch.tensor(boxes, device=device)
            scores = torch.tensor(scores, device=device)
            batch_boxes.append(boxes)
            batch_scores.append(scores)
        else:
            batch_boxes.append(torch.empty((0, 4), device=device))
            batch_scores.append(torch.empty((0,), device=device))
    return batch_boxes, batch_scores


# Set up device, load the trained model, and prepare the transformation.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOFaceDetector().to(device)
model.load_state_dict(torch.load("trained_yolo_face_detector.pth", map_location=device))
model.eval()

transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
)

# Open the webcam
cap = cv2.VideoCapture(0)
conf_threshold = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break
    orig_frame = frame.copy()
    # Convert BGR to RGB for processing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Apply the inference transformation
    transformed = transform(image=image_rgb)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
    boxes, scores = decode_predictions(output, conf_threshold, device)

    # Get predicted boxes from the first (and only) image in the batch
    boxes = boxes[0].cpu().numpy()
    h, w, _ = frame.shape
    for box in boxes:
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Webcam Inference", orig_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
