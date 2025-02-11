import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import patches
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class SingleObjectCocoDataset(Dataset):
    def __init__(self, root, annFile, transform):
        self.coco = CocoDetection(root=root, annFile=annFile)
        self.transform = transform

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, targets = self.coco[idx]
        target = targets[0]
        bbox = target["bbox"]
        label = target["category_id"]
        img, bbox = self.transform(img, bbox)
        return (
            img,
            torch.tensor(bbox, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        # Extract feature layers excluding classifier head
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Get number of input features for the classifier
        in_features = mobilenet.classifier[0].in_features

        # Define classifier and bounding box regressor
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 4),
        )

        self.bb = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 4),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)


def train_epocs(model, optimizer, train_dl, val_dl, epochs=10, C=1000):
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y_bb, y_class in train_dl:
            batch = y_class.shape[0]
            x = x.to(device).float()
            y_class = y_class.to(device)
            y_bb = y_bb.to(device).float()
            out_class, out_bb = model(x)
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb / C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss / total
        val_loss, val_acc = val_metrics(model, valid_dl, C)
        print(
            "train_loss %.3f val_loss %.3f val_acc %.3f"
            % (train_loss, val_loss, val_acc)
        )
    return sum_loss / total


def val_metrics(model, valid_dl, C=1000):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x, y_bb, y_class in valid_dl:
        batch = y_class.shape[0]
        x = x.to(device).float()
        y_class = y_class.to(device)
        y_bb = y_bb.to(device).float()
        out_class, out_bb = model(x)
        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
        loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb / C
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
    return sum_loss / total, correct / total


def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


def resize_transform(image, bbox, size=(640, 640)):
    w, h = image.size
    image = T.Resize(size)(image)
    new_w, new_h = size
    scale_x = new_w / w
    scale_y = new_h / h
    bbox = [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    image = transform(image)  # Converts image to [0,1] range
    return image, bbox


def visualize_predictions(model, dataset, device, num_images=5):
    model.eval()
    for i in range(num_images):
        image, bbox, label = dataset[i]
        input_img = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_cls, pred_bbox = model(input_img)
        pred_cls = torch.argmax(pred_cls, dim=1).item()
        pred_bbox = pred_bbox.squeeze(0).cpu().numpy()
        ground_bbox = bbox.squeeze(0).cpu().numpy()
        img_np = image.cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        fig, ax = plt.subplots(1)
        rect_predict = patches.Rectangle(
            (pred_bbox[0], pred_bbox[1]),
            pred_bbox[2],
            pred_bbox[3],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        rect_ground = patches.Rectangle(
            (ground_bbox[0], ground_bbox[1]),
            ground_bbox[2],
            ground_bbox[3],
            linewidth=2,
            edgecolor="b",
            facecolor="none",
        )
        ax.add_patch(rect_ground)
        ax.add_patch(rect_predict)
        ax.text(
            pred_bbox[0],
            pred_bbox[1],
            f"Pred: {pred_cls}",
            bbox=dict(facecolor="yellow", alpha=0.5),
        )
        ax.imshow(img_np)
        plt.show()


base_path = "/Users/gstrauss/Downloads/chair detection 2"
train_path = os.path.join(base_path, "train")
valid_path = os.path.join(base_path, "valid")
test_path = os.path.join(base_path, "test")
train_ann = os.path.join(train_path, "_annotations.coco.json")
valid_ann = os.path.join(valid_path, "_annotations.coco.json")
test_ann = os.path.join(test_path, "_annotations.coco.json")

train_dataset = SingleObjectCocoDataset(
    root=train_path, annFile=train_ann, transform=resize_transform
)
valid_dataset = SingleObjectCocoDataset(
    root=valid_path, annFile=valid_ann, transform=resize_transform
)
test_dataset = SingleObjectCocoDataset(
    root=test_path, annFile=test_ann, transform=resize_transform
)
batch_size = 32
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
model = BB_model().to(device)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.006)
train_epocs(model, optimizer, train_dl, valid_dl, epochs=5)
visualize_predictions(model, test_dataset, device, num_images=5)
