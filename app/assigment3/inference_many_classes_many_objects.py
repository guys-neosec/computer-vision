import argparse

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torchvision import models

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions ---


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
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            SEBlock(in_channels // 4),
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
            weights=models.MobileNet_V3_Large_Weights.DEFAULT,
        )
        self.backbone = mobilenet.features
        self.conv_reduce = nn.Conv2d(960, 384, kernel_size=1)
        # The head predicts 5 box attributes + num_classes scores per anchor.
        self.head = YOLOHead(384, num_anchors * (5 + num_classes))
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        features = self.conv_reduce(features)
        out = self.head(features)
        return out


# --- Inference Utilities ---


def compute_giou_vectorized(pred_boxes, gt_boxes, eps=1e-6):
    # Minimal IoU computation used for NMS.
    p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    g_x1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
    g_y1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
    g_x2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
    g_y2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2
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
    return iou


def non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.5):
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    indices = scores.argsort(descending=True)
    keep = []
    while indices.numel() > 0:
        current = indices[0]
        keep.append(current)
        if indices.numel() == 1:
            break
        current_box = boxes[current].unsqueeze(0)
        other_boxes = boxes[indices[1:]]
        ious = compute_giou_vectorized(
            current_box.repeat(other_boxes.size(0), 1),
            other_boxes,
        )
        indices = indices[1:][ious < iou_threshold]
    return keep


def inference(
    model,
    image,
    anchors,
    num_classes,
    conf_threshold=0.5,
    iou_threshold=0.5,
):
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
    batch_size, _, grid_h, grid_w = output.shape
    num_anchors = anchors.shape[0]
    # Reshape and permute output to [grid_h, grid_w, num_anchors, 5+num_classes]
    output = output.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
    output = output.permute(0, 3, 4, 1, 2).contiguous()[0]
    boxes = []
    scores = []
    labels = []
    for i in range(grid_h):
        for j in range(grid_w):
            for a in range(num_anchors):
                pred = output[i, j, a]
                obj_score = torch.sigmoid(pred[0])
                if obj_score < conf_threshold:
                    continue
                tx = torch.sigmoid(pred[1])
                ty = torch.sigmoid(pred[2])
                tw = pred[3]
                th = pred[4]
                cx = (j + tx) / grid_w
                cy = (i + ty) / grid_h
                anchor_w, anchor_h = anchors[a]
                bw = anchor_w * torch.exp(tw)
                bh = anchor_h * torch.exp(th)
                boxes.append(torch.tensor([cx, cy, bw, bh]))
                scores.append(obj_score)
                cls_prob = pred[5:]
                label = torch.argmax(cls_prob)
                labels.append(label)
    if boxes:
        boxes = torch.stack(boxes)
        scores = torch.stack(scores)
        keep_idx = non_max_suppression(
            {"boxes": boxes, "scores": scores},
            conf_threshold,
            iou_threshold,
        )
        # Convert indices (which might be 0-dim tensors) to plain ints
        keep_idx = [int(idx) for idx in keep_idx]
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = torch.tensor(labels)[keep_idx]
    return boxes, scores, labels


def cxcywh_to_xyxy(box, img_width, img_height):
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return int(x1), int(y1), int(x2), int(y2)


# --- Label and Color Mappings ---

# Adjust these mappings as needed.
LABEL_NAMES = {
    0: "Ball",
    1: "ball",
    2: "made",
    3: "person",
    4: "rim",
    5: "shoot",
    6: "unknown",
}

# OpenCV uses BGR color order.
LABEL_COLORS = {
    0: (0, 0, 255),  # Red
    1: (0, 255, 0),  # Green
    2: (255, 0, 0),  # Blue
    3: (0, 255, 255),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (255, 255, 0),  # Cyan
    6: (128, 128, 128),  # Gray
}

# --- Main Inference Script ---


def main(args):
    # Define transforms:
    # One transform for inference (includes normalization) and one for drawing (resizing only)
    inference_transform = A.Compose(
        [
            A.Resize(400, 400),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
    )
    draw_transform = A.Resize(400, 400)

    # Read and prepare the image
    orig_image = cv2.imread(args.image)
    if orig_image is None:
        print("Error: Image not found or unable to load.")
        return
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    # Resize for drawing (but keep it in uint8)
    drawn_image = draw_transform(image=image_rgb)["image"]
    # For inference, apply normalization and conversion to tensor
    transformed = inference_transform(image=image_rgb)
    input_tensor = transformed["image"]

    # Define anchors (from your training output)
    anchors = torch.tensor(
        [
            [0.0345, 0.0513],
            [0.4671, 0.8831],
            [0.1763, 0.4546],
            [0.8256, 0.8913],
            [0.4828, 0.5786],
            [0.1012, 0.1713],
            [0.2491, 0.6914],
            [0.2815, 0.2879],
            [0.0883, 0.2786],
        ],
        device=device,
    )

    num_classes = 7
    num_anchors = anchors.shape[0]

    # Initialize model and load weights
    model = YOLOFaceDetector(num_anchors=num_anchors, num_classes=num_classes).to(
        device,
    )
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Run inference
    boxes, scores, labels = inference(
        model,
        input_tensor,
        anchors,
        num_classes,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )

    img_h, img_w = drawn_image.shape[:2]
    # Draw bounding boxes with different colors
    for box, score, label in zip(boxes, scores, labels, strict=False):
        x1, y1, x2, y2 = cxcywh_to_xyxy(box.cpu().numpy(), img_w, img_h)
        color = LABEL_COLORS.get(int(label.item()), (255, 255, 255))
        label_text = f"{LABEL_NAMES.get(int(label.item()), 'unknown')}:{score:.2f}"
        cv2.rectangle(drawn_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            drawn_image,
            label_text,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # Convert RGB back to BGR for OpenCV display or saving
    output_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)

    # Save and/or display the image
    if args.output:
        cv2.imwrite(args.output, output_image)
        print(f"Output saved to {args.output}")
    else:
        cv2.imshow("Detections", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for YOLO-based face detector.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="basketball_dynamic_anchors.pth",
        help="Path to the model weights file.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for non-max suppression.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save the output image (optional).",
    )
    args = parser.parse_args()
    main(args)
