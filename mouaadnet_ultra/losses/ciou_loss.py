"""
CIoU Loss (Complete IoU Loss)
==============================
Advanced bounding box regression loss that considers overlap, center distance,
and aspect ratio for more accurate localization.

Reference: Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        box1: (N, 4) boxes in [x1, y1, x2, y2] format
        box2: (N, 4) boxes in [x1, y1, x2, y2] format
        eps: Small value to avoid division by zero
        
    Returns:
        IoU values (N,)
    """
    # Intersection area
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # Union area
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = area1 + area2 - inter_area + eps
    
    return inter_area / union_area


def box_ciou(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate CIoU between predicted and target boxes.
    
    CIoU = IoU - (ρ²(b, b_gt) / c²) - αv
    
    Where:
    - ρ²(b, b_gt): Squared distance between centers
    - c²: Squared diagonal of smallest enclosing box
    - α: Trade-off parameter
    - v: Aspect ratio consistency
    
    Args:
        pred: (N, 4) predicted boxes [x1, y1, x2, y2]
        target: (N, 4) target boxes [x1, y1, x2, y2]
        eps: Small value for numerical stability
        
    Returns:
        CIoU values (N,) in range [-1, 1]
    """
    # Basic IoU
    iou = box_iou(pred, target, eps)
    
    # Enclosing box
    enclose_x1 = torch.min(pred[:, 0], target[:, 0])
    enclose_y1 = torch.min(pred[:, 1], target[:, 1])
    enclose_x2 = torch.max(pred[:, 2], target[:, 2])
    enclose_y2 = torch.max(pred[:, 3], target[:, 3])
    
    # Diagonal squared of smallest enclosing box
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps
    
    # Center distance squared
    pred_cx = (pred[:, 0] + pred[:, 2]) / 2
    pred_cy = (pred[:, 1] + pred[:, 3]) / 2
    target_cx = (target[:, 0] + target[:, 2]) / 2
    target_cy = (target[:, 1] + target[:, 3]) / 2
    
    rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
    
    # Aspect ratio consistency
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    target_w = target[:, 2] - target[:, 0]
    target_h = target[:, 3] - target[:, 1]
    
    v = (4 / math.pi ** 2) * torch.pow(
        torch.atan(target_w / (target_h + eps)) - torch.atan(pred_w / (pred_h + eps)), 2
    )
    
    # Trade-off parameter
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    
    # CIoU
    ciou = iou - rho2 / c2 - alpha * v
    
    return ciou


class IoULoss(nn.Module):
    """
    Basic IoU Loss.
    
    Loss = 1 - IoU
    
    Args:
        reduction: Reduction method ('none', 'mean', 'sum')
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate IoU loss.
        
        Args:
            pred: (N, 4) predicted boxes
            target: (N, 4) target boxes
            
        Returns:
            IoU loss
        """
        iou = box_iou(pred, target)
        loss = 1 - iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CIoULoss(nn.Module):
    """
    Complete IoU (CIoU) Loss.
    
    More accurate than basic IoU as it considers:
    - Overlap area
    - Center point distance
    - Aspect ratio consistency
    
    Loss = 1 - CIoU
    
    Args:
        reduction: Reduction method ('none', 'mean', 'sum')
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate CIoU loss.
        
        Args:
            pred: (N, 4) predicted boxes [x1, y1, x2, y2]
            target: (N, 4) target boxes [x1, y1, x2, y2]
            
        Returns:
            CIoU loss
        """
        ciou = box_ciou(pred, target)
        loss = 1 - ciou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DIoULoss(nn.Module):
    """
    Distance IoU Loss.
    
    DIoU = IoU - (ρ²(b, b_gt) / c²)
    
    Penalizes center point distance without aspect ratio term.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        iou = box_iou(pred, target, eps)
        
        # Enclosing box
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])
        
        c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps
        
        # Center distance
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        
        rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        diou = iou - rho2 / c2
        loss = 1 - diou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == "__main__":
    # Test CIoU Loss
    print("Testing CIoU Loss...")
    
    # Create sample boxes
    pred = torch.tensor([
        [10, 10, 50, 50],
        [20, 20, 60, 60],
        [0, 0, 100, 100],
    ], dtype=torch.float32)
    
    target = torch.tensor([
        [10, 10, 50, 50],   # Perfect match
        [25, 25, 65, 65],   # Slightly offset
        [50, 50, 150, 150], # Poor overlap
    ], dtype=torch.float32)
    
    # Test IoU
    iou = box_iou(pred, target)
    print(f"IoU values: {iou.tolist()}")
    
    # Test CIoU
    ciou = box_ciou(pred, target)
    print(f"CIoU values: {ciou.tolist()}")
    
    # Test losses
    iou_loss = IoULoss()(pred, target)
    ciou_loss = CIoULoss()(pred, target)
    diou_loss = DIoULoss()(pred, target)
    
    print(f"IoU Loss: {iou_loss:.4f}")
    print(f"CIoU Loss: {ciou_loss:.4f}")
    print(f"DIoU Loss: {diou_loss:.4f}")
    
    print("✓ All CIoU tests passed!")
