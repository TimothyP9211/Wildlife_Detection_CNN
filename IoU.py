import torch
import torch.nn as nn
from torch.nn import *

class IoUModule(nn.Module):
    def __init__(self):
        super(IoUModule, self).__init__()
    
    def xywh_to_xyxy(self, box):
        # Convert (x_min, y_min, width, height) to (x_min, y_min, x_max, y_max)
        x1, x2, y1, y2 = box[:, 0], box[:, 0] + box[:, 2], box[:, 1], box[:, 1] + box[:, 3]
        return torch.stack((x1, y1, x2, y2), dim=1)

    # Assume predicted boxes and target boxes are in (x_min, y_min, width, height) format
    def compute_iou(self, pred_boxes, target_boxes):
        pred_xyxy = self.xywh_to_xyxy(pred_boxes)
        target_xyxy = self.xywh_to_xyxy(target_boxes)

        # Reshape for broadcasting where pred_xyxy is (N, 4) and target_xyxy is (M, 4)
        # IoU[i, j] is the IoU between pred_boxes[i] and target_boxes[j]
        pred_xyxy = pred_xyxy[:, None, :]
        target_xyxy = target_xyxy[None, :, :]

        # Calculate corner coordinates of intersection rectangle
        x1 = torch.max(pred_xyxy[:, :, 0], target_xyxy[:, :, 0])
        y1 = torch.max(pred_xyxy[:, :, 1], target_xyxy[:, :, 1])
        x2 = torch.min(pred_xyxy[:, :, 2], target_xyxy[:, :, 2])
        y2 = torch.min(pred_xyxy[:, :, 3], target_xyxy[:, :, 3])

        # Calculate area of union rectangle, zero if no overlap
        intersect_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Calculate area of bounding box rectangles
        rect1_area = (pred_xyxy[:, :, 2] - pred_xyxy[:, :, 0]) * (pred_xyxy[:, :, 3] - pred_xyxy[:, :, 1])
        rect2_area = (target_xyxy[:, :, 2] - target_xyxy[:, :, 0]) * (target_xyxy[:, :, 3] - target_xyxy[:, :, 1])

        # Compute IoU, adding a small epsilon to avoid division by zero (although it should not happen in practice)
        IoU = intersect_area / (rect1_area + rect2_area - intersect_area + 1e-5)
        return IoU

    def forward(self, pred_boxes, target_boxes):
        return self.compute_iou(pred_boxes, target_boxes)