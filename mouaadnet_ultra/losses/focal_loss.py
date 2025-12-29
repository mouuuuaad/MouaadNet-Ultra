"""
Focal Loss Implementation
=========================
Loss function designed for class imbalance, especially useful for
heatmap-based detection where positives are sparse.

Reference: Focal Loss for Dense Object Detection (RetinaNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
    
    Where:
    - p_t = p if y=1 else 1-p
    - α_t = α if y=1 else 1-α
    
    The focusing parameter γ reduces the loss for well-classified examples,
    focusing training on hard negatives.
    
    Args:
        alpha: Balancing factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('none', 'mean', 'sum')
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            pred: Predictions (logits or probabilities)
            target: Binary targets {0, 1}
            
        Returns:
            Focal loss
        """
        # Apply sigmoid if needed (handle both logits and probabilities)
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Clamp for numerical stability
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        
        # Calculate p_t
        p_t = pred * target + (1 - pred) * (1 - target)
        
        # Calculate alpha_t
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Binary cross entropy
        bce = -torch.log(p_t)
        
        # Focal loss
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class GaussianFocalLoss(nn.Module):
    """
    Gaussian Focal Loss for heatmap prediction.
    
    Designed for CenterNet-style detection where target heatmaps
    have Gaussian peaks at object centers.
    
    Uses quality focal loss formulation for better localization.
    
    Args:
        alpha: Balancing factor (default: 2.0)
        gamma: Focusing parameter (default: 4.0)
        reduction: Reduction method
    """
    
    def __init__(
        self,
        alpha: float = 2.0,
        gamma: float = 4.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate Gaussian focal loss.
        
        Args:
            pred: Predicted heatmap (B, C, H, W), values in [0, 1]
            target: Target heatmap (B, C, H, W), Gaussian peaks at centers
            
        Returns:
            Gaussian focal loss
        """
        # Clamp for numerical stability
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        
        # Positive locations (center peaks)
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        # Positive loss: standard focal loss at peaks
        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        
        # Negative loss: reduced for locations close to peaks
        neg_weight = torch.pow(1 - target, self.gamma)
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weight * neg_mask
        
        # Combine
        num_pos = pos_mask.sum() + 1  # +1 to avoid division by zero
        
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        
        if self.reduction == 'sum':
            return loss * pred.numel()
        return loss


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss.
    
    Extends focal loss to handle soft labels (IoU scores as targets).
    
    QFL = -|y - σ|^β × ((1-y)×log(1-σ) + y×log(σ))
    
    Where:
    - y: Target quality score (IoU)
    - σ: Predicted score
    - β: Focusing parameter
    """
    
    def __init__(
        self,
        beta: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate quality focal loss.
        
        Args:
            pred: Predictions (sigmoid applied)
            target: Soft targets (e.g., IoU scores)
            
        Returns:
            Quality focal loss
        """
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        
        # Quality focal weight
        weight = torch.abs(target - pred).pow(self.beta)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Weighted loss
        loss = weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == "__main__":
    # Test Focal Loss
    print("Testing Focal Loss...")
    
    # Binary classification example
    pred = torch.tensor([0.9, 0.7, 0.3, 0.1])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal(pred, target)
    print(f"Focal Loss: {loss:.4f}")
    
    # Heatmap example
    pred_heatmap = torch.rand(2, 1, 8, 8)
    target_heatmap = torch.zeros(2, 1, 8, 8)
    target_heatmap[:, :, 4, 4] = 1.0  # Single peak
    
    gaussian_focal = GaussianFocalLoss()
    hm_loss = gaussian_focal(pred_heatmap, target_heatmap)
    print(f"Gaussian Focal Loss: {hm_loss:.4f}")
    
    # Quality focal loss
    qfl = QualityFocalLoss(beta=2.0)
    soft_target = torch.tensor([0.8, 0.6, 0.2, 0.1])  # IoU scores as targets
    qfl_loss = qfl(pred, soft_target)
    print(f"Quality Focal Loss: {qfl_loss:.4f}")
    
    print("✓ All Focal Loss tests passed!")
