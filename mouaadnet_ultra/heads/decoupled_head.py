"""
Decoupled Multi-Task Head
==========================
Coordinates detection and classification tasks with gradient isolation
to prevent interference between human localization and gender prediction.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from .detection_head import DetectionHead, AnchorFreeHead
from .gender_head import GenderHead, MultiScaleGenderHead


class GradientReversal(torch.autograd.Function):
    """
    Gradient Reversal Layer.
    
    Passes input unchanged during forward, but reverses gradient during backward.
    Used to prevent certain gradients from affecting shared features.
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientScaling(torch.autograd.Function):
    """
    Gradient Scaling Layer.
    
    Scales gradients during backward pass to balance multi-task learning.
    """
    
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class DecoupledHead(nn.Module):
    """
    Decoupled Multi-Task Head for Detection + Gender Classification.
    
    Key features:
    1. Separate detection and classification branches
    2. Gradient isolation to prevent task interference
    3. Optional gradient scaling for task balancing
    
    Architecture:
    - Shared neck features (N3, N4, N5)
    - Detection branch: Predicts person locations (heatmap, size, offset)
    - Gender branch: Classifies detected persons
    
    Args:
        in_channels: Input feature channels from neck
        det_channels: Detection head hidden channels
        cls_channels: Classification head hidden channels
        num_classes: Number of detection classes (default: 1 for person)
        gradient_scale_det: Gradient scale for detection branch
        gradient_scale_cls: Gradient scale for classification branch
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        det_channels: int = 64,
        cls_channels: int = 128,
        num_classes: int = 1,
        gradient_scale_det: float = 1.0,
        gradient_scale_cls: float = 1.0,
    ):
        super().__init__()
        
        # Separate feature adapters for each task
        self.det_adapter = nn.Sequential(
            nn.Conv2d(in_channels, det_channels, 1, bias=False),
            nn.BatchNorm2d(det_channels),
            nn.ReLU6(inplace=True),
        )
        
        self.cls_adapter = nn.Sequential(
            nn.Conv2d(in_channels, cls_channels, 1, bias=False),
            nn.BatchNorm2d(cls_channels),
            nn.ReLU6(inplace=True),
        )
        
        # Detection head (anchor-free)
        self.detection_head = AnchorFreeHead(det_channels, num_classes)
        
        # Gender classification head
        self.gender_head = GenderHead(cls_channels, hidden_dim=cls_channels)
        
        # Gradient scaling factors
        self.gradient_scale_det = gradient_scale_det
        self.gradient_scale_cls = gradient_scale_cls
        
    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        person_crops: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both detection and classification.
        
        Args:
            features: Tuple of (N3, N4, N5) feature maps from neck
            person_crops: Optional pre-cropped person features for classification
            
        Returns:
            Dict containing:
            - 'heatmaps': Detection heatmaps per scale
            - 'sizes': BBox size predictions per scale
            - 'offsets': Center offset predictions per scale
            - 'gender': Gender logits (if person_crops provided)
        """
        n3, n4, n5 = features
        
        outputs = {
            'heatmaps': [],
            'sizes': [],
            'offsets': [],
        }
        
        # Process each scale for detection
        for feat in [n3, n4, n5]:
            # Adapt features for detection
            det_feat = self.det_adapter(feat)
            
            # Apply gradient scaling
            if self.training and self.gradient_scale_det != 1.0:
                det_feat = GradientScaling.apply(det_feat, self.gradient_scale_det)
            
            # Detect at this scale
            hm, sz, off = self.detection_head(det_feat)
            
            outputs['heatmaps'].append(hm)
            outputs['sizes'].append(sz)
            outputs['offsets'].append(off)
        
        # Gender classification (on cropped features or pooled features)
        if person_crops is not None:
            # Classify provided crop features
            cls_feat = self.cls_adapter(person_crops)
            
            if self.training and self.gradient_scale_cls != 1.0:
                cls_feat = GradientScaling.apply(cls_feat, self.gradient_scale_cls)
            
            outputs['gender'] = self.gender_head(cls_feat)
        else:
            # Use highest resolution features for global classification
            # (useful for training when we don't have specific crops)
            cls_feat = self.cls_adapter(n3)
            
            if self.training and self.gradient_scale_cls != 1.0:
                cls_feat = GradientScaling.apply(cls_feat, self.gradient_scale_cls)
            
            outputs['gender'] = self.gender_head(cls_feat)
        
        return outputs
    
    def detect_only(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, list]:
        """Run detection only (no gender classification)."""
        outputs = {'heatmaps': [], 'sizes': [], 'offsets': []}
        
        for feat in features:
            det_feat = self.det_adapter(feat)
            hm, sz, off = self.detection_head(det_feat)
            
            outputs['heatmaps'].append(hm)
            outputs['sizes'].append(sz)
            outputs['offsets'].append(off)
        
        return outputs
    
    def classify_gender(self, crops: torch.Tensor) -> torch.Tensor:
        """Run gender classification only on cropped person features."""
        cls_feat = self.cls_adapter(crops)
        return self.gender_head(cls_feat)


class MultiScaleDecoupledHead(nn.Module):
    """
    Enhanced Decoupled Head with multi-scale gender classification.
    
    Uses features from all scales for more robust gender prediction.
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        det_channels: int = 64,
        cls_channels: int = 128,
        num_classes: int = 1,
    ):
        super().__init__()
        
        # Detection branch
        self.detection_head = DetectionHead(
            in_channels=det_channels,
            num_classes=num_classes,
            num_scales=3,
        )
        
        # Feature adapters
        self.det_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, det_channels, 1, bias=False),
                nn.BatchNorm2d(det_channels),
                nn.ReLU6(inplace=True),
            )
            for _ in range(3)
        ])
        
        self.cls_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
            )
            for _ in range(3)
        ])
        
        # Multi-scale gender classification
        self.gender_head = MultiScaleGenderHead(
            in_channels=in_channels,
            num_scales=3,
            hidden_dim=cls_channels,
        )
        
    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Multi-scale forward pass."""
        n3, n4, n5 = features
        
        # Adapt and detect
        det_features = tuple(
            adapter(feat) 
            for adapter, feat in zip(self.det_adapters, [n3, n4, n5])
        )
        det_outputs = self.detection_head(det_features)
        
        # Adapt and classify
        cls_features = tuple(
            adapter(feat)
            for adapter, feat in zip(self.cls_adapters, [n3, n4, n5])
        )
        gender_logits = self.gender_head(cls_features)
        
        return {
            **det_outputs,
            'gender': gender_logits,
        }


if __name__ == "__main__":
    # Test Decoupled Head
    print("Testing Decoupled Head...")
    
    # Simulate neck outputs
    n3 = torch.randn(2, 64, 52, 52)
    n4 = torch.randn(2, 64, 26, 26)
    n5 = torch.randn(2, 64, 13, 13)
    features = (n3, n4, n5)
    
    # Test DecoupledHead
    head = DecoupledHead(in_channels=64, det_channels=64, cls_channels=128)
    outputs = head(features)
    
    print("DecoupledHead outputs:")
    print(f"  Heatmaps: {[h.shape for h in outputs['heatmaps']]}")
    print(f"  Sizes: {[s.shape for s in outputs['sizes']]}")
    print(f"  Offsets: {[o.shape for o in outputs['offsets']]}")
    print(f"  Gender: {outputs['gender'].shape}")
    
    # Test with person crops
    crops = torch.randn(10, 64, 16, 16)  # 10 detected persons
    outputs_crop = head(features, person_crops=crops)
    print(f"  Gender (from crops): {outputs_crop['gender'].shape}")
    
    # Test detect_only
    det_only = head.detect_only(features)
    print(f"\nDetect only: {len(det_only['heatmaps'])} scales")
    
    # Test classify_gender
    gender_only = head.classify_gender(crops)
    print(f"Classify only: {gender_only.shape}")
    
    # Test MultiScaleDecoupledHead
    multi_head = MultiScaleDecoupledHead(in_channels=64)
    multi_outputs = multi_head(features)
    print(f"\nMultiScale gender: {multi_outputs['gender'].shape}")
    
    # Count parameters
    params = sum(p.numel() for p in head.parameters())
    print(f"\nDecoupledHead parameters: {params:,}")
    
    print("✓ All Decoupled Head tests passed!")
