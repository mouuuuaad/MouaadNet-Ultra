"""
Multi-Task Loss Function
========================
Combined loss for simultaneous human detection and gender classification.

L_total = λ_det × L_detection + λ_gender × L_gender

Where L_detection = L_heatmap + L_size + L_offset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .ciou_loss import CIoULoss
from .focal_loss import GaussianFocalLoss, FocalLoss


class DetectionLoss(nn.Module):
    """
    Combined loss for anchor-free detection.
    
    Components:
    - Heatmap loss: Gaussian Focal Loss for center point prediction
    - Size loss: L1 loss for width/height prediction
    - Offset loss: L1 loss for sub-pixel center refinement
    
    Args:
        heatmap_weight: Weight for heatmap loss (default: 1.0)
        size_weight: Weight for size loss (default: 0.1)
        offset_weight: Weight for offset loss (default: 1.0)
    """
    
    def __init__(
        self,
        heatmap_weight: float = 1.0,
        size_weight: float = 0.1,
        offset_weight: float = 1.0,
    ):
        super().__init__()
        
        self.heatmap_loss = GaussianFocalLoss(alpha=2.0, gamma=4.0)
        self.size_loss = nn.L1Loss(reduction='none')
        self.offset_loss = nn.L1Loss(reduction='none')
        
        self.heatmap_weight = heatmap_weight
        self.size_weight = size_weight
        self.offset_weight = offset_weight
        
    def forward(
        self,
        pred_heatmap: torch.Tensor,
        pred_size: torch.Tensor,
        pred_offset: torch.Tensor,
        target_heatmap: torch.Tensor,
        target_size: torch.Tensor,
        target_offset: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate detection loss.
        
        Args:
            pred_heatmap: (B, C, H, W) predicted center heatmap
            pred_size: (B, 2, H, W) predicted width/height
            pred_offset: (B, 2, H, W) predicted x/y offset
            target_heatmap: (B, C, H, W) ground truth heatmap
            target_size: (B, 2, H, W) ground truth size
            target_offset: (B, 2, H, W) ground truth offset
            target_mask: (B, 1, H, W) mask for valid locations
            
        Returns:
            Dict with loss components and total
        """
        # Heatmap loss (all locations)
        loss_heatmap = self.heatmap_loss(pred_heatmap, target_heatmap)
        
        # Size and offset loss (only at positive locations)
        if target_mask is None:
            # Use peak locations from heatmap as mask
            target_mask = (target_heatmap == 1).float()
        
        num_pos = target_mask.sum() + 1  # +1 to avoid division by zero
        
        # Size loss
        loss_size = self.size_loss(pred_size, target_size)
        loss_size = (loss_size * target_mask).sum() / num_pos
        
        # Offset loss
        loss_offset = self.offset_loss(pred_offset, target_offset)
        loss_offset = (loss_offset * target_mask).sum() / num_pos
        
        # Weighted total
        total = (
            self.heatmap_weight * loss_heatmap +
            self.size_weight * loss_size +
            self.offset_weight * loss_offset
        )
        
        return {
            'loss_heatmap': loss_heatmap,
            'loss_size': loss_size,
            'loss_offset': loss_offset,
            'loss_detection': total,
        }


class GenderLoss(nn.Module):
    """
    Gender classification loss with class weighting.
    
    Uses Binary Cross Entropy with optional class weights
    to handle imbalanced data.
    
    Args:
        pos_weight: Weight for positive class (Female) (default: 1.0)
        label_smoothing: Label smoothing factor (default: 0.0)
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        self.pos_weight = torch.tensor([pos_weight])
        self.label_smoothing = label_smoothing
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate gender classification loss.
        
        Args:
            pred: (N, 1) predicted logits
            target: (N, 1) binary targets (0=Male, 1=Female)
            
        Returns:
            BCE loss
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Move pos_weight to same device
        pos_weight = self.pos_weight.to(pred.device)
        
        return F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight
        )


class MultiTaskLoss(nn.Module):
    """
    Complete Multi-Task Loss for MOUAADNET-ULTRA.
    
    Combines detection and gender classification losses with
    automatic weighting and task balancing.
    
    L_total = λ_det × L_detection + λ_gender × L_gender
    
    Args:
        det_weight: Weight for detection loss (default: 1.0)
        gender_weight: Weight for gender loss (default: 1.0)
        gender_pos_weight: Class weight for female (default: 5.0 for imbalance)
        use_uncertainty_weighting: Whether to use learned weights (default: False)
    """
    
    def __init__(
        self,
        det_weight: float = 1.0,
        gender_weight: float = 1.0,
        gender_pos_weight: float = 5.0,
        use_uncertainty_weighting: bool = False,
    ):
        super().__init__()
        
        self.detection_loss = DetectionLoss()
        self.gender_loss = GenderLoss(pos_weight=gender_pos_weight)
        
        self.det_weight = det_weight
        self.gender_weight = gender_weight
        
        # Uncertainty-based task weighting (learnable)
        if use_uncertainty_weighting:
            # log(σ²) for numerical stability
            self.log_var_det = nn.Parameter(torch.zeros(1))
            self.log_var_gender = nn.Parameter(torch.zeros(1))
        else:
            self.log_var_det = None
            self.log_var_gender = None
            
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-task loss.
        
        Args:
            predictions: Dict with 'heatmaps', 'sizes', 'offsets', 'gender'
            targets: Dict with 'heatmaps', 'sizes', 'offsets', 'gender_labels', 'masks'
            
        Returns:
            Dict with all loss components and total
        """
        losses = {}
        
        # Detection loss (multi-scale)
        total_det_loss = 0
        num_scales = len(predictions.get('heatmaps', []))
        
        for i in range(num_scales):
            scale_losses = self.detection_loss(
                pred_heatmap=predictions['heatmaps'][i],
                pred_size=predictions['sizes'][i],
                pred_offset=predictions['offsets'][i],
                target_heatmap=targets['heatmaps'][i],
                target_size=targets['sizes'][i],
                target_offset=targets['offsets'][i],
                target_mask=targets.get('masks', [None] * num_scales)[i],
            )
            
            total_det_loss = total_det_loss + scale_losses['loss_detection']
            
            # Store per-scale losses
            for k, v in scale_losses.items():
                losses[f'{k}_s{i}'] = v
        
        # Average across scales
        losses['loss_detection'] = total_det_loss / max(num_scales, 1)
        
        # Gender loss
        if 'gender' in predictions and 'gender_labels' in targets:
            losses['loss_gender'] = self.gender_loss(
                predictions['gender'],
                targets['gender_labels'],
            )
        else:
            losses['loss_gender'] = torch.tensor(0.0, device=predictions['heatmaps'][0].device)
        
        # Combine with weighting
        if self.log_var_det is not None:
            # Uncertainty weighting: L/σ² + log(σ²)
            det_weight = torch.exp(-self.log_var_det)
            gender_weight = torch.exp(-self.log_var_gender)
            
            losses['total'] = (
                det_weight * losses['loss_detection'] + self.log_var_det +
                gender_weight * losses['loss_gender'] + self.log_var_gender
            )
        else:
            losses['total'] = (
                self.det_weight * losses['loss_detection'] +
                self.gender_weight * losses['loss_gender']
            )
        
        return losses


if __name__ == "__main__":
    # Test Multi-Task Loss
    print("Testing Multi-Task Loss...")
    
    batch_size = 2
    h, w = 52, 52
    
    # Create mock predictions
    predictions = {
        'heatmaps': [torch.sigmoid(torch.randn(batch_size, 1, h, w))],
        'sizes': [torch.randn(batch_size, 2, h, w)],
        'offsets': [torch.randn(batch_size, 2, h, w)],
        'gender': torch.randn(batch_size, 1),
    }
    
    # Create mock targets
    targets = {
        'heatmaps': [torch.zeros(batch_size, 1, h, w)],
        'sizes': [torch.zeros(batch_size, 2, h, w)],
        'offsets': [torch.zeros(batch_size, 2, h, w)],
        'gender_labels': torch.randint(0, 2, (batch_size, 1)).float(),
    }
    
    # Add some positive locations
    targets['heatmaps'][0][:, :, 25, 25] = 1.0
    targets['sizes'][0][:, :, 25, 25] = 0.5
    
    # Test DetectionLoss
    det_loss = DetectionLoss()
    det_losses = det_loss(
        predictions['heatmaps'][0],
        predictions['sizes'][0],
        predictions['offsets'][0],
        targets['heatmaps'][0],
        targets['sizes'][0],
        targets['offsets'][0],
    )
    print(f"Detection losses: {', '.join(f'{k}={v:.4f}' for k, v in det_losses.items())}")
    
    # Test GenderLoss
    gender_loss = GenderLoss(pos_weight=5.0)
    g_loss = gender_loss(predictions['gender'], targets['gender_labels'])
    print(f"Gender loss: {g_loss:.4f}")
    
    # Test MultiTaskLoss
    multi_loss = MultiTaskLoss(det_weight=1.0, gender_weight=1.0)
    all_losses = multi_loss(predictions, targets)
    print(f"\nMulti-task losses:")
    for k, v in all_losses.items():
        if not k.startswith('loss_heatmap_s') and not k.startswith('loss_size_s'):
            print(f"  {k}: {v:.4f}")
    
    # Test with uncertainty weighting
    multi_loss_uw = MultiTaskLoss(use_uncertainty_weighting=True)
    all_losses_uw = multi_loss_uw(predictions, targets)
    print(f"\nWith uncertainty weighting:")
    print(f"  Total: {all_losses_uw['total']:.4f}")
    print(f"  log_var_det: {multi_loss_uw.log_var_det.item():.4f}")
    print(f"  log_var_gender: {multi_loss_uw.log_var_gender.item():.4f}")
    
    print("✓ All Multi-Task Loss tests passed!")
