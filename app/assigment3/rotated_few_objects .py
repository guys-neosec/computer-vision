import json
import os
import math
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import MobileNet_V3_Large_Weights
from sklearn.cluster import KMeans
from albumentations.core.transforms_interface import DualTransform
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
num_classes = 19

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def compute_anchors_from_dataset(dataset, num_anchors):
    all_wh = []

    for img_path, bboxes, _ in dataset.data:
        # Read image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape

        for bbox in bboxes:
            bw = bbox[2] / w  # Normalize width
            bh = bbox[3] / h  # Normalize height
            all_wh.append([bw, bh])

    kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(all_wh)
    anchors = kmeans.cluster_centers_

    return torch.tensor(anchors, dtype=torch.float32, device=device)

def normalize_angle_rad(angle):
    """Normalize angle to [-π, π]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def polygon_to_obb(line):
    parts = line.strip().split()
    coords = list(map(float, parts[:8]))
    category = parts[8]
    cx = sum(coords[::2]) / 4.0
    cy = sum(coords[1::2]) / 4.0
    w = math.hypot(coords[2] - coords[0], coords[3] - coords[1])
    h = math.hypot(coords[4] - coords[2], coords[5] - coords[3])
    angle = math.atan2(coords[3] - coords[1], coords[2] - coords[0])
    angle = normalize_angle_rad(angle)  # ensure in [-π, π]
    return cx, cy, w, h, angle, category

class FaceDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform, class_mapping):
        """
        Args:
            img_dir (str): Directory with images.
            ann_dir (str): Directory with annotation .txt files (DOTA format).
            transform: Albumentations transform with bbox_params(format="obb").
            class_mapping (dict): Maps class names (str) to integer category IDs.
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.class_mapping = class_mapping

        self.data = []
        for filename in os.listdir(ann_dir):
            if not filename.endswith(".txt"):
                continue
            ann_path = os.path.join(ann_dir, filename)
            img_id = os.path.splitext(filename)[0]
            img_path = os.path.join(img_dir, img_id + ".png")
            if not os.path.exists(img_path):
                continue

            with open(ann_path, "r") as f:
                lines = f.readlines()[2:]  # skip imagesource and gsd

            bboxes = []
            labels = []
            for line in lines:
                try:
                    cx, cy, w, h, angle, class_name = polygon_to_obb(line)
                    if class_name not in self.class_mapping:
                        continue
                    bboxes.append([cx, cy, w, h, angle])
                    labels.append(self.class_mapping[class_name])
                except Exception as e:
                    logger.warning(f"Failed parsing line in {ann_path}: {line.strip()}\n{e}")
                    continue

            if bboxes:
                self.data.append((img_path, bboxes, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, bboxes, labels = self.data[index]  # bboxes: [[cx, cy, w, h, angle], ...]
        image = cv2.imread(img_path)
        orig_h, orig_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform image only (without bbox_params)
        transform_img = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        transformed = transform_img(image=image)
        image_out = transformed["image"]

        # Manually resize the bboxes using the same scale factors.
        scale_x = 512 / orig_w
        scale_y = 512 / orig_h
        norm_bboxes = []
        for cx, cy, w, h, angle in bboxes:
            new_cx = cx * scale_x
            new_cy = cy * scale_y
            new_w  = w  * scale_x
            new_h  = h  * scale_y
            # Normalize to [0,1] relative to new image size (512x512)
            norm_bboxes.append([new_cx / 512, new_cy / 512, new_w / 512, new_h / 512, angle])
        # debug_print_scaling_info(img_path,orig_w,orig_h,bboxes,norm_bboxes)
        norm_bboxes = torch.tensor(norm_bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return image_out, norm_bboxes, labels



def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(boxes), list(labels)

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
        mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone = mobilenet.features
        self.conv_reduce = nn.Conv2d(960, 384, kernel_size=1)
        self.head = YOLOHead(384, num_anchors * (7 + num_classes))
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        features = self.conv_reduce(features)
        out = self.head(features)
        return out


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    prob = torch.sigmoid(inputs)
    p_t = targets * prob + (1 - targets) * (1 - prob)
    loss = alpha * (1 - p_t) ** gamma * bce_loss
    return loss.mean()


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2



def decode_predictions(pred, grid_x, grid_y, anchor, grid_w, grid_h):
    # Decode the bbox predictions.
    tx = torch.sigmoid(pred[1])
    ty = torch.sigmoid(pred[2])
    tw = pred[3]
    th = pred[4]
    t_angle = pred[5]  # you might apply a non-linearity if desired (e.g., tanh)

    cx = (grid_x + tx) / grid_w
    cy = (grid_y + ty) / grid_h
    bw = anchor[0] * torch.exp(tw)
    bh = anchor[1] * torch.exp(th)
    return torch.stack([cx, cy, bw, bh,t_angle])

def yolo_loss_function_oriented(
    output,
    bboxes,
    labels,
    device,
    anchors,
    num_classes,
    lambda_coord=5,
    lambda_noobj=2,
    lambda_cls=1,
    lambda_angle=1,   # new parameter for angle loss
):
    batch_size = output.shape[0]
    grid_h, grid_w = output.shape[2], output.shape[3]
    num_anchors = anchors.shape[0]
    # Now output is assumed to be [B, num_anchors, 6 + num_classes, grid_h, grid_w]
    output = output.view(batch_size, num_anchors, 7 + num_classes, grid_h, grid_w)
    output = output.permute(0, 3, 4, 1, 2).contiguous()

    # Build targets (ensure build_target now outputs a 6-dimensional target for boxes, where target[..., 5] is the angle)
    target_box, target_cls = build_target(
        bboxes, labels, grid_h, grid_w, batch_size, device, anchors, num_anchors, num_classes
    )

    # Objectness and classification predictions
    pred_obj = torch.sigmoid(output[..., 0])
    pred_cls = output[..., 7:]  # now classes start after 6 channels

    # Objectness loss:
    obj_mask = target_box[..., 0] == 1
    noobj_mask = target_box[..., 0] == 0
    obj_loss = F.binary_cross_entropy(pred_obj[obj_mask], target_box[..., 0][obj_mask], reduction="mean")
    noobj_loss = F.binary_cross_entropy(pred_obj[noobj_mask], target_box[..., 0][noobj_mask], reduction="mean")

    # -------------------------------
    # Vectorized coordinate loss (for cx, cy, w, h)
    indices = torch.nonzero(obj_mask, as_tuple=False)  # (N, 4)
    if indices.numel() > 0:
        t = target_box[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]  # (N, 6)
        preds = output[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]  # (N, 6+num_classes)

        grid_y = indices[:, 1].float()
        grid_x = indices[:, 2].float()

        # Decode predictions for coordinates:
        tx = torch.sigmoid(preds[:, 1])
        ty = torch.sigmoid(preds[:, 2])
        tw = preds[:, 3]
        th = preds[:, 4]

        decoded_cx = (grid_x + tx) / grid_w
        decoded_cy = (grid_y + ty) / grid_h
        anchor_selected = anchors[indices[:, 3]]
        tw = torch.clamp(tw, -10, 10)
        th = torch.clamp(th, -10, 10)
        decoded_w = anchor_selected[:, 0] * torch.exp(tw)
        decoded_h = anchor_selected[:, 1] * torch.exp(th)
        pred_sin = torch.tanh(preds[:, 5])  # channel 5
        pred_cos = torch.tanh(preds[:, 6])  # channel 6
        pred_angle = torch.atan2(pred_sin, pred_cos)

        decoded_preds = torch.stack([decoded_cx, decoded_cy, decoded_w, decoded_h,pred_angle], dim=1)

        # Build ground truth boxes (for cx, cy, w, h)
        gt_cx = (grid_x + t[:, 1]) / grid_w
        gt_cy = (grid_y + t[:, 2]) / grid_h
        gt_w = anchor_selected[:, 0] * torch.exp(t[:, 3])
        gt_h = anchor_selected[:, 1] * torch.exp(t[:, 4])
        gt_sin = t[:, 5]
        gt_cos = t[:, 6]
        gt_angle = torch.atan2(gt_sin, gt_cos)

        gt_boxes = torch.stack([gt_cx, gt_cy, gt_w, gt_h,gt_angle], dim=1)

        ious = probiou(decoded_preds, gt_boxes)
        coord_loss_giou = (1 - ious).mean()
        coord_loss_l1 = F.smooth_l1_loss(decoded_preds, gt_boxes)
        coord_loss = coord_loss_giou + 0.5 * coord_loss_l1

        # Angle loss: using cosine difference, where target angle is t[:, 5] and predicted angle is preds[:, 5]
        angle_loss = F.mse_loss(pred_sin, gt_sin) + F.mse_loss(pred_cos, gt_cos)

    else:
        coord_loss = 0.0
        angle_loss = 0.0

    # -------------------------------
    # Classification loss:
    if obj_mask.sum() > 0:
        pred_cls_obj = pred_cls[obj_mask]
        target_cls_obj = target_cls[obj_mask].long()
        class_loss = F.cross_entropy(pred_cls_obj, target_cls_obj, reduction="mean")
    else:
        class_loss = 0.0

    total_loss = (
        lambda_coord * coord_loss
        + obj_loss
        + lambda_noobj * noobj_loss
        + lambda_cls * class_loss
        + lambda_angle * angle_loss
    )
    return total_loss, coord_loss, obj_loss, noobj_loss, class_loss, angle_loss

def build_target(bboxes, labels, grid_h, grid_w, batch_size, device, anchors, num_anchors, num_classes):
    # target_box: (batch, grid_h, grid_w, num_anchors, 5) for objectness and bbox regression.
    target_box = torch.zeros((batch_size, grid_h, grid_w, num_anchors, 7), device=device)
    # target_cls: (batch, grid_h, grid_w, num_anchors) for classification.
    # Initialize with 0 (background). For anchors with a face, we will set the value to 1.
    target_cls = torch.zeros((batch_size, grid_h, grid_w, num_anchors), dtype=torch.long, device=device)
    for i in range(batch_size):
        for face, lab in zip(bboxes[i], labels[i]):
            cx, cy, bw, bh, angle = face.tolist()
            cell_x = int(cx * grid_w)
            cell_y = int(cy * grid_h)
            cell_x = min(cell_x, grid_w - 1)
            cell_y = min(cell_y, grid_h - 1)
            best_iou = 0.0
            best_anchor = 0
            assigned_anchors = []
            iou_treshold = 0.5
            for a in range(num_anchors):
                anchor_w, anchor_h = anchors[a].tolist()
                inter = min(bw, anchor_w) * min(bh, anchor_h)
                union = bw * bh + anchor_w * anchor_h - inter + 1e-6
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = a

                if iou > iou_treshold:
                    assigned_anchors.append(a)
            if not assigned_anchors:
                assigned_anchors.append(best_anchor)
            for anchor_idx in assigned_anchors:
                anchor_w, anchor_h = anchors[anchor_idx].tolist()
                t_x = cx * grid_w - cell_x
                t_y = cy * grid_h - cell_y
                t_w = math.log(bw / (anchor_w + 1e-6) + 1e-6)
                t_h = math.log(bh / (anchor_h + 1e-6) + 1e-6)
                target_box[i, cell_y, cell_x, anchor_idx, 0] = 1  # objectness
                target_box[i, cell_y, cell_x, anchor_idx, 1] = t_x
                target_box[i, cell_y, cell_x, anchor_idx, 2] = t_y
                target_box[i, cell_y, cell_x, anchor_idx, 3] = t_w
                target_box[i, cell_y, cell_x, anchor_idx, 4] = t_h
                target_box[i, cell_y, cell_x, anchor_idx, 5] = angle
                target_cls[i, cell_y, cell_x, anchor_idx] = lab
    return target_box, target_cls


def _get_covariance_matrix(boxes):
    # Generate covariance matrix from obbs
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def probiou(obb1, obb2, eps=1e-7):
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps) * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps) * 0.5
    ratio = ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (
        4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps
    )
    ratio = ratio.clamp(min=1e-6)
    t3 = ratio.log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd

def angle_diff(a, b):
    diff = (a - b + math.pi) % (2 * math.pi) - math.pi
    return diff



def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True

def s_unit(w,h,wtag,htag,s_inter):
    return w*h+htag*wtag-s_inter



def train(model, dataloader, optimizer, scheduler, epochs, device, writer, freeze_epochs=5):
    model.train()
    global_step = 0
    for epoch in range(epochs):
        if epoch < freeze_epochs:
            freeze_backbone(model)
        elif epoch == freeze_epochs:
            unfreeze_backbone(model)
            logger.info("Backbone unfrozen for fine-tuning.")
        total_loss_epoch = 0.0
        total_obj_loss = 0.0
        total_noobj_loss = 0.0
        total_cls_loss = 0.0
        for imgs, bboxes, labels in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            # total_loss, coord_loss, obj_loss, noobj_loss, class_loss, angle_loss
            loss, coord_loss, obj_loss, noobj_loss, cls_loss,angle_loss = yolo_loss_function_oriented(out, bboxes, labels, device, anchors, num_classes)
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            total_obj_loss += obj_loss.item()
            total_noobj_loss += noobj_loss if isinstance(noobj_loss, float) else noobj_loss.item()
            total_cls_loss += cls_loss if isinstance(cls_loss, float) else cls_loss.item()
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            global_step += 1
        avg_loss = total_loss_epoch / len(dataloader)
        avg_obj = total_obj_loss / len(dataloader)
        avg_noobj = total_noobj_loss / len(dataloader)
        avg_cls = total_cls_loss / len(dataloader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Total Loss: {avg_loss:.4f}, Obj Loss: {avg_obj:.4f},coords Loss: {coord_loss:.4f}, noobj Loss: {avg_noobj:.4f}, Cls Loss: {avg_cls:.4f}, angle loss {angle_loss:.4f}, LR: {current_lr:.6f}")
        writer.add_scalar("Train/EpochLoss", avg_loss, epoch)
        writer.add_scalar("Train/EpochObjLoss", avg_obj, epoch)
        writer.add_scalar("Train/EpochGiouLoss", avg_noobj, epoch)
        writer.add_scalar("Train/EpochClsLoss", avg_cls, epoch)
        writer.add_scalar("Train/LearningRate", current_lr, epoch)
        # # check_gradients(model)
        # if (epoch + 1) % 10 == 0:
        #   model_path = f"/content/trained_yolo_oriented_{epoch + 1}.pth"
        #   torch.save(model.state_dict(), model_path)
        #   logger.info(f"Saved model checkpoint at {model_path}")


def oriented_box_to_polygon(cx, cy, w, h, angle, degrees=False):
    rect = Polygon([
        (-w/2, -h/2),
        ( w/2, -h/2),
        ( w/2,  h/2),
        (-w/2,  h/2),
    ])
    # angle is either radians or degrees
    ang_deg = angle if degrees else math.degrees(angle)
    rect = rotate(rect, ang_deg, origin=(0, 0))
    rect = translate(rect, xoff=cx, yoff=cy)
    return rect

def compute_oriented_iou(pred_boxes, gt_boxes, degrees=False):
    """
    pred_boxes, gt_boxes: shape (N, 5) = [cx, cy, w, h, angle]
    Returns iou for each box pair (length N).
    """
    # We'll do this in a loop for clarity; shapely won't be GPU vectorized anyway
    N = pred_boxes.size(0)
    ious = []
    for i in range(N):
        cxp, cyp, wp, hp, ap = pred_boxes[i].tolist()
        cxg, cyg, wg, hg, ag = gt_boxes[i].tolist()

        poly_pred = oriented_box_to_polygon(cxp, cyp, wp, hp, ap, degrees)
        poly_gt   = oriented_box_to_polygon(cxg, cyg, wg, hg, ag, degrees)

        inter_area = poly_pred.intersection(poly_gt).area
        union_area = poly_pred.union(poly_gt).area
        iou = inter_area / union_area if union_area > 1e-6 else 0.0
        ious.append(iou)

    return torch.tensor(ious, device=pred_boxes.device)

def aabb_to_obb(bbox):
    x_min, y_min, w, h = bbox
    cx = x_min + w / 2
    cy = y_min + h / 2
    angle = 0  # initial angle is 0 for axis-aligned boxes
    return [cx, cy, w, h, math.radians(angle)]

def draw_obb_on_image(image, obbs, labels, class_mapping, image_size=512):
    # Assume image is a numpy array in RGB format.
    image = image.copy()
    index = 0
    for obb, label in zip(obbs, labels):
        # Multiply by image_size to convert normalized coordinates to pixel coordinates.
        cx, cy, w, h, angle = [coord * image_size if idx < 4 else coord
                                for idx, coord in enumerate(obb)]
        index+=1
        # Convert radians to degrees for OpenCV
        angle_deg = math.degrees(angle)

        # Ensure width/height are positive
        w, h = abs(w), abs(h)

        # Create the rotated rectangle
        rect = ((cx, cy), (w, h), angle_deg)
        box = cv2.boxPoints(rect).astype(int)

        # Draw bounding box
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # Add label text
        label_text = [k for k, v in class_mapping.items() if v == label][0]
        cv2.putText(image, label_text, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return image


def inference_vectorized_obb(model, image, anchors, num_classes, conf_threshold=0.5, iou_threshold=0.5):
    """
    Perform vectorized inference on an image for OBB outputs.
    Returns decoded boxes [cx, cy, bw, bh, angle], scores, and predicted classes.
    """
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
    # Expected output shape: (1, num_anchors*(6+num_classes), grid_h, grid_w)
    batch_size, _, grid_h, grid_w = output.shape
    num_anchors = anchors.shape[0]
    # Reshape: (1, num_anchors, 6+num_classes, grid_h, grid_w)
    output = output.view(batch_size, num_anchors, 6 + num_classes, grid_h, grid_w)
    # Permute to shape: (grid_h, grid_w, num_anchors, 6+num_classes)
    output = output.permute(0, 3, 4, 1, 2).contiguous()[0]

    # Compute objectness scores and build mask
    obj_scores = torch.sigmoid(output[..., 0])  # shape: (grid_h, grid_w, num_anchors)
    mask = obj_scores >= conf_threshold
    if mask.sum() == 0:
        return torch.empty((0, 5)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

    # Create grid indices (vectorized)
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h, device=device),
                                    torch.arange(grid_w, device=device), indexing='ij')
    grid_x = grid_x.unsqueeze(-1).expand(grid_h, grid_w, num_anchors).float()
    grid_y = grid_y.unsqueeze(-1).expand(grid_h, grid_w, num_anchors).float()

    # Flatten predictions and corresponding grid indices
    output_flat = output.view(-1, 6 + num_classes)
    obj_scores_flat = obj_scores.view(-1)
    grid_x_flat = grid_x.reshape(-1)
    grid_y_flat = grid_y.reshape(-1)
    mask_flat = mask.view(-1)

    valid_preds = output_flat[mask_flat]  # (N_valid, 6+num_classes)
    valid_scores = obj_scores_flat[mask_flat]  # (N_valid,)
    nonzero_indices = mask.nonzero(as_tuple=False)  # (N_valid, 3) indices: (i, j, anchor)
    a_idx = nonzero_indices[:, 2]
    selected_anchors = anchors[a_idx]  # (N_valid, 2)
    grid_x_valid = grid_x_flat[mask_flat]
    grid_y_valid = grid_y_flat[mask_flat]

    # Decode bounding box parameters (OBB)
    tx = torch.sigmoid(valid_preds[:, 1])
    ty = torch.sigmoid(valid_preds[:, 2])
    tw = valid_preds[:, 3]
    th = valid_preds[:, 4]
    t_angle = valid_preds[:, 5]  # Optionally, you might use tanh(t_angle)

    cx = (grid_x_valid + tx) / grid_w
    cy = (grid_y_valid + ty) / grid_h
    bw = selected_anchors[:, 0] * torch.exp(tw)
    bh = selected_anchors[:, 1] * torch.exp(th)

    # Stack decoded boxes including angle: shape (N_valid, 5)
    boxes = torch.stack([cx, cy, bw, bh, t_angle], dim=1)

    # Decode class predictions
    cls_logits = valid_preds[:, 6:]  # shape: (N_valid, num_classes)
    cls_probs = torch.softmax(cls_logits, dim=1)
    pred_classes = torch.argmax(cls_probs, dim=1)

    # (Optional) Filter out detections with a specific background class if needed.
    # For example, if your background is class index 6:
    # valid_class_mask = pred_classes != 6
    # boxes, valid_scores, pred_classes = boxes[valid_class_mask], valid_scores[valid_class_mask], pred_classes[valid_class_mask]

    # Convert boxes to axis-aligned for NMS:
    boxes_xyxy = torch.zeros((boxes.shape[0], 4), device=device)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    keep = torch.ops.torchvision.nms(boxes_xyxy, valid_scores, iou_threshold)
    boxes_final = boxes[keep]  # Still normalized [cx, cy, bw, bh, angle]
    scores_final = valid_scores[keep]
    classes_final = pred_classes[keep]

    return boxes_final, scores_final, classes_final

def draw_predicted_obb_on_image(image, boxes, labels, image_size=512, color=255):
    """
    Draw predicted oriented bounding boxes on an image.
    Args:
        image (np.ndarray): Image in RGB format.
        boxes (np.ndarray): Array of boxes in normalized [cx, cy, w, h, angle] format.
        labels (np.ndarray): Array of predicted class indices.
        image_size (int): The size (width/height) of the image (assumed square).
    Returns:
        np.ndarray: Image with drawn boxes.
    """
    image = image.copy()
    for box, label in zip(boxes, labels):
        # Unpack the normalized values.
        cx, cy, w, h, angle = box
        # Convert normalized coordinates to pixel values.
        cx, cy, w, h = cx * image_size, cy * image_size, w * image_size, h * image_size
        angle_deg = math.degrees(angle)
        # Explicitly convert values to native Python floats.
        rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))
        pts = cv2.boxPoints(rect)
        pts = np.int64(pts)
        # Draw the predicted OBB in blue.
        cv2.drawContours(image, [pts], 0, (255-color, color, 0), 2)
        # Draw the predicted class as text at the center.
        cv2.putText(image, str(int(label.item())), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 2)
    return image



def draw_obb_on_image(image, obbs, labels, class_mapping, image_size=512):
    # Assume image is a numpy array in RGB format.
    image = image.copy()
    index = 0
    for obb, label in zip(obbs, labels):
        # Multiply by image_size to convert normalized coordinates to pixel coordinates.
        cx, cy, w, h, angle = [coord * image_size if idx < 4 else coord
                                for idx, coord in enumerate(obb)]
        index+=1
        # Convert radians to degrees for OpenCV
        angle_deg = math.degrees(angle)

        # Ensure width/height are positive
        w, h = abs(w), abs(h)

        # Create the rotated rectangle
        rect = ((cx, cy), (w, h), angle_deg)
        box = cv2.boxPoints(rect).astype(int)

        # Draw bounding box
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # Add label text
        label_text = [k for k, v in class_mapping.items() if v == label][0]
        cv2.putText(image, label_text, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return image


def inference_vectorized_obb(model, image, anchors, num_classes, conf_threshold=0.5, iou_threshold=0.5):
    """
    Perform vectorized inference on an image for OBB outputs.
    Returns decoded boxes [cx, cy, bw, bh, angle], scores, and predicted classes.
    """
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
    # Expected output shape: (1, num_anchors*(6+num_classes), grid_h, grid_w)
    batch_size, _, grid_h, grid_w = output.shape
    num_anchors = anchors.shape[0]
    # Reshape: (1, num_anchors, 6+num_classes, grid_h, grid_w)
    output = output.view(batch_size, num_anchors, 7 + num_classes, grid_h, grid_w)
    # Permute to shape: (grid_h, grid_w, num_anchors, 6+num_classes)
    output = output.permute(0, 3, 4, 1, 2).contiguous()[0]

    # Compute objectness scores and build mask
    obj_scores = torch.sigmoid(output[..., 0])  # shape: (grid_h, grid_w, num_anchors)
    mask = obj_scores >= conf_threshold
    if mask.sum() == 0:
        return torch.empty((0, 5)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

    # Create grid indices (vectorized)
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h, device=device),
                                    torch.arange(grid_w, device=device), indexing='ij')
    grid_x = grid_x.unsqueeze(-1).expand(grid_h, grid_w, num_anchors).float()
    grid_y = grid_y.unsqueeze(-1).expand(grid_h, grid_w, num_anchors).float()

    # Flatten predictions and corresponding grid indices
    output_flat = output.view(-1, 7 + num_classes)
    obj_scores_flat = obj_scores.view(-1)
    grid_x_flat = grid_x.reshape(-1)
    grid_y_flat = grid_y.reshape(-1)
    mask_flat = mask.view(-1)

    valid_preds = output_flat[mask_flat]  # (N_valid, 6+num_classes)
    valid_scores = obj_scores_flat[mask_flat]  # (N_valid,)
    nonzero_indices = mask.nonzero(as_tuple=False)  # (N_valid, 3) indices: (i, j, anchor)
    a_idx = nonzero_indices[:, 2]
    selected_anchors = anchors[a_idx]  # (N_valid, 2)
    grid_x_valid = grid_x_flat[mask_flat]
    grid_y_valid = grid_y_flat[mask_flat]

    # Decode bounding box parameters (OBB)
    tx = torch.sigmoid(valid_preds[:, 1])
    ty = torch.sigmoid(valid_preds[:, 2])
    tw = valid_preds[:, 3]
    th = valid_preds[:, 4]
    sin = valid_preds[:, 5]  # Optionally, you might use tanh(t_angle)
    cos = valid_preds[:, 6]  # Optionally, you might use tanh(t_angle)
    t_angle =  torch.atan2(sin, cos)
    cx = (grid_x_valid + tx) / grid_w
    cy = (grid_y_valid + ty) / grid_h
    bw = selected_anchors[:, 0] * torch.exp(tw)
    bh = selected_anchors[:, 1] * torch.exp(th)

    # Stack decoded boxes including angle: shape (N_valid, 5)
    boxes = torch.stack([cx, cy, bw, bh, t_angle], dim=1)

    # Decode class predictions
    cls_logits = valid_preds[:, 6:]  # shape: (N_valid, num_classes)
    cls_probs = torch.softmax(cls_logits, dim=1)
    pred_classes = torch.argmax(cls_probs, dim=1)

    # Convert boxes to axis-aligned for NMS:
    boxes_xyxy = torch.zeros((boxes.shape[0], 4), device=device)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    keep = torch.ops.torchvision.nms(boxes_xyxy, valid_scores, iou_threshold)
    boxes_final = boxes[keep]  # Still normalized [cx, cy, bw, bh, angle]
    scores_final = valid_scores[keep]
    classes_final = pred_classes[keep]

    return boxes_final, scores_final, classes_final

def draw_predicted_obb_on_image(image, boxes, labels, image_size=512, color=255):
    """
    Draw predicted oriented bounding boxes on an image.
    Args:
        image (np.ndarray): Image in RGB format.
        boxes (np.ndarray): Array of boxes in normalized [cx, cy, w, h, angle] format.
        labels (np.ndarray): Array of predicted class indices.
        image_size (int): The size (width/height) of the image (assumed square).
    Returns:
        np.ndarray: Image with drawn boxes.
    """
    image = image.copy()
    for box, label in zip(boxes, labels):
        # Unpack the normalized values.
        cx, cy, w, h, angle = box
        # Convert normalized coordinates to pixel values.
        cx, cy, w, h = cx * image_size, cy * image_size, w * image_size, h * image_size
        angle_deg = math.degrees(angle)
        # Explicitly convert values to native Python floats.
        rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))
        pts = cv2.boxPoints(rect)
        pts = np.int64(pts)
        # Draw the predicted OBB in blue.
        cv2.drawContours(image, [pts], 0, (255-color, color, 0), 2)
        # Draw the predicted class as text at the center.
        cv2.putText(image, str(int(label.item())), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 2)
    return image

if __name__ == "__main__":
    transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    class_mapping = {
        "plane": 0,
        "large-vehicle": 1,
        "small-vehicle": 2,
        "ship": 3,
        "helicopter": 4,
        "roundabout": 5,
        "bridge": 6,
        "storage-tank": 7,
        "soccer-ball-field": 8,
        "tennis-court": 9,
        "swimming-pool": 10,
        "basketball-court": 11,
        "ground-track-field": 12,
        "harbor": 13,
        "container-crane": 14,
        "vehicle": 15,
        "baseball-diamond": 16,
        "helipad": 17,
        "airport": 18
    }

    train_dataset = FaceDataset(
        img_dir="dota_img_1/images/",
        ann_dir="labelTxt",
        transform=transform,
        class_mapping=class_mapping
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    train_dataset.data = [train_dataset.data[13]]  # override internal dataset
    anchors = compute_anchors_from_dataset(train_dataset, num_anchors=5)
    print(anchors)
    model = YOLOFaceDetector(anchors.shape[0], num_classes=len(class_mapping)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter()

    train(model, train_loader, optimizer, scheduler, epochs, device, writer, freeze_epochs=0)
    # Load original image + boxes
    image_tensor, boxes_tensor, labels_tensor = train_dataset[0]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * std + mean) * 255.0
    image_np = np.clip(image_np, 0, 255).astype("uint8")

    model.eval()
    with torch.no_grad():
        raw_output = model(image_tensor.unsqueeze(0).to(device))
        # Permute and reshape the output to shape [1, grid_h, grid_w, num_anchors, 6+num_classes]
        output = raw_output.permute(0, 2, 3, 1).contiguous()
        _, grid_h, grid_w, _ = output.shape
        num_anchors = anchors.shape[0]
        output = output.view(1, grid_h, grid_w, num_anchors, 7 + len(class_mapping))
        # Decode predictions.
        pred_boxes, pred_scores, pred_labels = inference_vectorized_obb(model,image_tensor, anchors, num_classes=len(class_mapping), conf_threshold=0.5)

            # Draw the predictions on the original (resized) image.
        pred_image = draw_predicted_obb_on_image(image_np, pred_boxes, pred_labels, image_size=512)

        # Convert the image back to BGR for OpenCV display/saving.
        pred_image_bgr = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)


    for i in range(len(train_dataset)):
        image_tensor, boxes_tensor, labels_tensor = train_dataset[i]
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * std + mean) * 255.0
        image_np = np.clip(image_np, 0, 255).astype("uint8")
        # Run inference on each image.
        with torch.no_grad():
            out = model(image_tensor.unsqueeze(0).to(device))
            out = out.permute(0, 2, 3, 1).contiguous()
            _, gh, gw, _ = out.shape
            out = out.view(1, gh, gw, num_anchors, 7 + len(class_mapping))
            boxes, scores, labels_pred = inference_vectorized_obb(model,image_tensor, anchors, num_classes=len(class_mapping), conf_threshold=0.5)
        pred_img = draw_predicted_obb_on_image(image_np, boxes, labels_pred)
        cv2.imwrite(f"predictions_{i}.png", cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        print(f"Saved predictions_{i}.png")

    torch.save(model.state_dict(), "app/assigment3/trained_yolo_face_detector_anchor_nms.pth")
    writer.close()
