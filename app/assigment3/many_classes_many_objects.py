import json
import math
import os

import albumentations as A
import cv2
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from loguru import logger
from sklearn.cluster import KMeans
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights

torch.backends.cudnn.benchmark = True

num_classes = 7
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu"),
)


def compute_anchors_from_dataset(dataset, num_anchors):
    all_wh = []
    for img_info, bboxes, labels in dataset.data:
        img_w, img_h = img_info["width"], img_info["height"]
        for bbox in bboxes:
            bw = bbox[2] / img_w
            bh = bbox[3] / img_h
            all_wh.append([bw, bh])
    kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(all_wh)
    anchors = kmeans.cluster_centers_
    return torch.tensor(anchors, dtype=torch.float32, device=device)


class FaceDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform):
        with open(ann_file) as f:
            ann_data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = {img["id"]: img for img in ann_data["images"]}
        annotations = {}
        for ann in ann_data["annotations"]:
            annotations.setdefault(ann["image_id"], []).append(ann)
        self.data = []
        for img in ann_data["images"]:
            if "file_name" not in img:
                continue
            img_id = img["id"]
            anns = annotations.get(img_id, [])
            valid_bboxes = []
            valid_labels = []
            for ann in anns:
                bbox = ann["bbox"]
                img_w, img_h = img["width"], img["height"]
                if (
                    bbox[0] < 0
                    or bbox[1] < 0
                    or bbox[0] + bbox[2] > img_w
                    or bbox[1] + bbox[3] > img_h
                ):
                    logger.warning(f"Skipping invalid bbox {bbox} in image {img_id}")
                    continue
                valid_bboxes.append(bbox)
                valid_labels.append(ann["category_id"])
            if valid_bboxes:
                self.data.append((img, valid_bboxes, valid_labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_info, bboxes, labels = self.data[index]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        image = transformed["image"]
        _, H, W = image.shape
        norm_bboxes = []
        for b in transformed["bboxes"]:
            cx = (b[0] + b[2] / 2) / W
            cy = (b[1] + b[3] / 2) / H
            bw = b[2] / W
            bh = b[3] / H
            norm_bboxes.append([cx, cy, bw, bh])
        norm_bboxes = torch.tensor(norm_bboxes, dtype=torch.float32)
        labels = torch.tensor(transformed["labels"], dtype=torch.long)
        return image, norm_bboxes, labels


def collate_fn(batch):
    images, boxes, labels = zip(*batch, strict=False)
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
        mobilenet = models.mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT,
        )
        self.backbone = mobilenet.features
        self.conv_reduce = nn.Conv2d(960, 384, kernel_size=1)
        self.head = YOLOHead(384, num_anchors * (5 + num_classes))
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        features = self.conv_reduce(features)
        out = self.head(features)
        return out


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def compute_giou_vectorized(pred_boxes, gt_boxes, eps=1e-6):
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


def build_target(
    bboxes,
    labels,
    grid_h,
    grid_w,
    batch_size,
    device,
    anchors,
    num_anchors,
    num_classes,
):
    target_box = torch.zeros(
        (batch_size, grid_h, grid_w, num_anchors, 5),
        device=device,
    )
    target_cls = torch.zeros(
        (batch_size, grid_h, grid_w, num_anchors),
        dtype=torch.long,
        device=device,
    )
    for i in range(batch_size):
        for face, lab in zip(bboxes[i], labels[i], strict=False):
            cx, cy, bw, bh = face.tolist()
            cell_x = int(cx * grid_w)
            cell_y = int(cy * grid_h)
            cell_x = min(cell_x, grid_w - 1)
            cell_y = min(cell_y, grid_h - 1)
            assigned_anchors = []
            iou_threshold = 0.5
            best_iou = 0.0
            best_anchor = None
            for a in range(num_anchors):
                anchor_w, anchor_h = anchors[a].tolist()
                inter = min(bw, anchor_w) * min(bh, anchor_h)
                union = bw * bh + anchor_w * anchor_h - inter + 1e-6
                current_iou = inter / union
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_anchor = a
                if current_iou >= iou_threshold:
                    assigned_anchors.append(a)
            if not assigned_anchors:
                assigned_anchors.append(best_anchor)
            for anchor_idx in assigned_anchors:
                t_x = cx * grid_w - cell_x
                t_y = cy * grid_h - cell_y
                anchor_w = anchors[anchor_idx][0].item()
                anchor_h = anchors[anchor_idx][1].item()
                t_w = math.log(bw / (anchor_w + 1e-6))
                t_h = math.log(bh / (anchor_h + 1e-6))
                target_box[i, cell_y, cell_x, anchor_idx, 0] = 1
                target_box[i, cell_y, cell_x, anchor_idx, 1] = t_x
                target_box[i, cell_y, cell_x, anchor_idx, 2] = t_y
                target_box[i, cell_y, cell_x, anchor_idx, 3] = t_w
                target_box[i, cell_y, cell_x, anchor_idx, 4] = t_h
                target_cls[i, cell_y, cell_x, anchor_idx] = lab
    return target_box, target_cls


def yolo_loss_function(
    output,
    bboxes,
    labels,
    device,
    anchors,
    num_classes,
    lambda_coord=5,
    lambda_noobj=1,
    lambda_cls=1,
    lambda_l1=1,
):
    batch_size = output.shape[0]
    grid_h, grid_w = output.shape[2], output.shape[3]
    num_anchors = anchors.shape[0]
    output = output.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
    output = output.permute(0, 3, 4, 1, 2).contiguous()
    target_box, target_cls = build_target(
        bboxes,
        labels,
        grid_h,
        grid_w,
        batch_size,
        device,
        anchors,
        num_anchors,
        num_classes,
    )
    pred_obj = torch.sigmoid(output[..., 0])
    pred_cls = output[..., 5:]
    obj_mask = target_box[..., 0] == 1
    noobj_mask = target_box[..., 0] == 0
    obj_loss = (
        F.binary_cross_entropy(
            pred_obj[obj_mask],
            target_box[..., 0][obj_mask],
            reduction="mean",
        )
        if obj_mask.sum() > 0
        else 0.0
    )
    noobj_loss = (
        F.binary_cross_entropy(
            pred_obj[noobj_mask],
            target_box[..., 0][noobj_mask],
            reduction="mean",
        )
        if noobj_mask.sum() > 0
        else 0.0
    )
    indices = torch.nonzero(obj_mask, as_tuple=False)
    if indices.numel() > 0:
        t = target_box[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]
        preds = output[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]
        grid_y = indices[:, 1].float()
        grid_x = indices[:, 2].float()
        tx = torch.sigmoid(preds[:, 1])
        ty = torch.sigmoid(preds[:, 2])
        tw = preds[:, 3]
        th = preds[:, 4]
        decoded_cx = (grid_x + tx) / grid_w
        decoded_cy = (grid_y + ty) / grid_h
        anchor_selected = anchors[indices[:, 3]]
        decoded_w = anchor_selected[:, 0] * torch.exp(tw)
        decoded_h = anchor_selected[:, 1] * torch.exp(th)
        decoded_preds = torch.stack(
            [decoded_cx, decoded_cy, decoded_w, decoded_h],
            dim=1,
        )
        gt_cx = (grid_x + t[:, 1]) / grid_w
        gt_cy = (grid_y + t[:, 2]) / grid_h
        gt_w = anchor_selected[:, 0] * torch.exp(t[:, 3])
        gt_h = anchor_selected[:, 1] * torch.exp(t[:, 4])
        gt_boxes = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1)
        giou_loss = (1 - compute_giou_vectorized(decoded_preds, gt_boxes)).mean()
        l1_loss = F.smooth_l1_loss(decoded_preds, gt_boxes, reduction="mean")
        coord_loss = giou_loss + lambda_l1 * l1_loss
    else:
        coord_loss = 0.0
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
    )
    return total_loss, coord_loss, obj_loss, noobj_loss, class_loss


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True


def train(
    model,
    dataloader,
    optimizer,
    epochs,
    device,
    writer,
    anchors,
    freeze_epochs=5,
):
    model.train()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5,
    )
    global_step = 0
    for epoch in range(epochs):
        if epoch < freeze_epochs:
            freeze_backbone(model)
        elif epoch == freeze_epochs:
            unfreeze_backbone(model)
            logger.info("Backbone unfrozen for fine-tuning.")
        total_loss_epoch = 0.0
        total_coord_loss = 0.0
        total_obj_loss = 0.0
        total_noobj_loss = 0.0
        total_cls_loss = 0.0
        for imgs, bboxes, labels in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss, coord_loss, obj_loss, noobj_loss, cls_loss = yolo_loss_function(
                out,
                bboxes,
                labels,
                device,
                anchors,
                num_classes,
            )
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            total_coord_loss += (
                coord_loss if isinstance(coord_loss, float) else coord_loss.item()
            )
            total_obj_loss += (
                obj_loss.item() if not isinstance(obj_loss, float) else obj_loss
            )
            total_noobj_loss += (
                noobj_loss.item() if not isinstance(noobj_loss, float) else noobj_loss
            )
            total_cls_loss += (
                cls_loss.item() if not isinstance(cls_loss, float) else cls_loss
            )
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            global_step += 1
        avg_loss = total_loss_epoch / len(dataloader)
        avg_coord = total_coord_loss / len(dataloader)
        avg_obj = total_obj_loss / len(dataloader)
        avg_noobj = total_noobj_loss / len(dataloader)
        avg_cls = total_cls_loss / len(dataloader)
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Total Loss: {avg_loss:.4f}, Coord Loss: {avg_coord:.4f}, Obj Loss: {avg_obj:.4f}, NoObj Loss: {avg_noobj:.4f}, Cls Loss: {avg_cls:.4f}",
        )
        writer.add_scalar("Train/EpochLoss", avg_loss, epoch)
        writer.add_scalar("Train/EpochCoordLoss", avg_coord, epoch)
        writer.add_scalar("Train/EpochObjLoss", avg_obj, epoch)
        writer.add_scalar("Train/EpochNoObjLoss", avg_noobj, epoch)
        writer.add_scalar("Train/EpochClsLoss", avg_cls, epoch)
        scheduler.step()


def evaluate_model(model, dataloader, device, anchors):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, bboxes, labels in dataloader:
            imgs = imgs.to(device)
            out = model(imgs)
            loss, _, _, _, _ = yolo_loss_function(
                out,
                bboxes,
                labels,
                device,
                anchors,
                num_classes,
            )
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


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


def inference(model, image, anchors, conf_threshold=0.5, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
    batch_size, _, grid_h, grid_w = output.shape
    num_anchors = anchors.shape[0]
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
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = torch.tensor(labels)[keep_idx]
    return boxes, scores, labels


if __name__ == "__main__":
    transform = A.Compose(
        [
            A.Resize(400, 400),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
    )
    train_dataset = FaceDataset(
        "/root/projects/computer-vision/Basketball.v2i/train",
        "/root/projects/computer-vision/Basketball.v2i/train/_annotations.coco.json",
        transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    anchors = compute_anchors_from_dataset(train_dataset, num_anchors=9)
    logger.info(anchors)
    model = YOLOFaceDetector(num_anchors=anchors.shape[0], num_classes=num_classes).to(
        device,
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 200
    writer = SummaryWriter()
    train(
        model,
        train_loader,
        optimizer,
        epochs,
        device,
        writer,
        anchors,
        freeze_epochs=1,
    )
    torch.save(model.state_dict(), "basketball_dynamic_anchors.pth")
    writer.close()
