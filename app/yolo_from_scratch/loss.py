import torch
from torch import nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 50)
        iou_b1 = intersection_over_union(predictions[..., 21:25], targets[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], targets[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_max, bestbox = torch.max(ious, dim=0)
        exists_box = targets[..., 20].unsqueeze(3)

        box_predictions = exists_box * (
            bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]
        )
        box_targets = exists_box * targets[..., 21:25]

        box_predictions[..., 2:4] = torch.sin(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6),
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse_loss(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        pred_box = (bestbox * predictions[..., 25:26]) + (1 - bestbox) * predictions[
            ...,
            20:21,
        ]
        object_loss = self.mse_loss(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * targets[..., 20:21]),
        )

        no_ob
