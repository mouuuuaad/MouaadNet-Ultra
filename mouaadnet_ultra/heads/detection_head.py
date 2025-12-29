"""
Anchor-Free Detection Head
==========================
CenterNet-style detection head that predicts humans as center points
with associated bounding box dimensions.

No anchor boxes needed - simpler and faster than anchor-based methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class ConvBNReLU(nn.Module):
    """Standard Conv + BatchNorm + ReLU6."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HeatmapHead(nn.Module):
    """
    Heatmap prediction branch for center point detection.
    
    Outputs a heatmap where each pixel represents the probability
    of being a human center point (single class for person detection).
    
    Args:
        in_channels: Number of input feature channels
        num_classes: Number of detection classes (default: 1 for person)
    """
    
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        
        self.head = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, 3),
            ConvBNReLU(in_channels, in_channels // 2, 3),
            nn.Conv2d(in_channels // 2, num_classes, 1),
        )
        
        # Initialize bias for focal loss stability
        self._init_bias()
        
    def _init_bias(self):
        """Initialize output bias for stable focal loss training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == 1:
                # Bias initialization for rare positive samples
                nn.init.constant_(m.bias, -2.19)  # -log((1-0.1)/0.1)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns heatmap with sigmoid activation."""
        return torch.sigmoid(self.head(x))


class SizeHead(nn.Module):
    """
    Size prediction branch for bounding box dimensions.
    
    Predicts width and height of the bounding box at each location.
    
    Args:
        in_channels: Number of input feature channels
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.head = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, 3),
            ConvBNReLU(in_channels, in_channels // 2, 3),
            nn.Conv2d(in_channels // 2, 2, 1),  # 2 channels: width, height
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 2, H, W) with width and height predictions."""
        return self.head(x)


class OffsetHead(nn.Module):
    """
    Offset prediction branch for sub-pixel localization.
    
    Predicts local offset to refine center point location from
    quantized feature map coordinates to continuous image coordinates.
    
    Args:
        in_channels: Number of input feature channels
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.head = nn.Sequential(
            ConvBNReLU(in_channels, in_channels // 2, 3),
            nn.Conv2d(in_channels // 2, 2, 1),  # 2 channels: x_offset, y_offset
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 2, H, W) with x and y offset predictions."""
        return self.head(x)


class AnchorFreeHead(nn.Module):
    """
    Complete Anchor-Free Detection Head.
    
    Combines three branches:
    - Heatmap: Center point probability (person detection)
    - Size: Bounding box dimensions (width, height)
    - Offset: Sub-pixel center offset refinement
    
    Args:
        in_channels: Number of input feature channels
        num_classes: Number of classes to detect (default: 1 for person)
    """
    
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        
        self.heatmap_head = HeatmapHead(in_channels, num_classes)
        self.size_head = SizeHead(in_channels)
        self.offset_head = OffsetHead(in_channels)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all detection branches.
        
        Args:
            x: Input feature map (B, C, H, W)
            
        Returns:
            Tuple of:
            - heatmap: (B, num_classes, H, W) center probability
            - size: (B, 2, H, W) width and height
            - offset: (B, 2, H, W) x and y offset
        """
        heatmap = self.heatmap_head(x)
        size = self.size_head(x)
        offset = self.offset_head(x)
        
        return heatmap, size, offset


class DetectionHead(nn.Module):
    """
    Multi-scale Detection Head.
    
    Processes features from multiple scales (P3, P4, P5) and
    produces detections at each scale.
    
    Args:
        in_channels: Number of input channels (same for all scales)
        num_classes: Number of classes (default: 1 for person)
        num_scales: Number of feature scales (default: 3)
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 1,
        num_scales: int = 3,
    ):
        super().__init__()
        
        # Shared detection head for all scales
        self.head = AnchorFreeHead(in_channels, num_classes)
        
        # Scale-specific adapters (optional, can share weights)
        self.scale_adapters = nn.ModuleList([
            ConvBNReLU(in_channels, in_channels, 3)
            for _ in range(num_scales)
        ])
        
    def forward(
        self,
        features: Tuple[torch.Tensor, ...]
    ) -> Dict[str, Tuple[torch.Tensor, ...]]:
        """
        Multi-scale detection.
        
        Args:
            features: Tuple of feature maps (N3, N4, N5) from neck
            
        Returns:
            Dict with 'heatmaps', 'sizes', 'offsets' - each a tuple of tensors
        """
        heatmaps = []
        sizes = []
        offsets = []
        
        for i, feat in enumerate(features):
            # Apply scale adapter
            feat = self.scale_adapters[i](feat)
            
            # Detect at this scale
            hm, sz, off = self.head(feat)
            
            heatmaps.append(hm)
            sizes.append(sz)
            offsets.append(off)
            
        return {
            'heatmaps': tuple(heatmaps),
            'sizes': tuple(sizes),
            'offsets': tuple(offsets),
        }


def decode_detections(
    heatmap: torch.Tensor,
    size: torch.Tensor,
    offset: torch.Tensor,
    stride: int,
    threshold: float = 0.3,
    top_k: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode detections from head outputs.
    
    Args:
        heatmap: (B, 1, H, W) center probability
        size: (B, 2, H, W) width, height
        offset: (B, 2, H, W) x, y offset
        stride: Feature map stride (e.g., 8, 16, 32)
        threshold: Detection confidence threshold
        top_k: Maximum number of detections
        
    Returns:
        Tuple of:
        - boxes: (B, K, 4) as [x1, y1, x2, y2]
        - scores: (B, K) confidence scores
        - valid: (B, K) boolean mask for valid detections
    """
    batch_size, _, h, w = heatmap.shape
    
    # Flatten and get top-k peaks
    heatmap_flat = heatmap.view(batch_size, -1)
    scores, indices = torch.topk(heatmap_flat, min(top_k, h * w), dim=1)
    
    # Convert flat indices to coordinates
    y_indices = (indices // w).float()
    x_indices = (indices % w).float()
    
    # Gather size and offset at peak locations
    indices_2d = indices.unsqueeze(1).expand(-1, 2, -1)
    
    size_flat = size.view(batch_size, 2, -1)
    peak_sizes = torch.gather(size_flat, 2, indices_2d)  # (B, 2, K)
    
    offset_flat = offset.view(batch_size, 2, -1)
    peak_offsets = torch.gather(offset_flat, 2, indices_2d)  # (B, 2, K)
    
    # Calculate center coordinates with offset
    cx = (x_indices + peak_offsets[:, 0, :]) * stride
    cy = (y_indices + peak_offsets[:, 1, :]) * stride
    
    # Calculate box dimensions
    w_box = peak_sizes[:, 0, :].abs() * stride
    h_box = peak_sizes[:, 1, :].abs() * stride
    
    # Convert to [x1, y1, x2, y2]
    x1 = cx - w_box / 2
    y1 = cy - h_box / 2
    x2 = cx + w_box / 2
    y2 = cy + h_box / 2
    
    boxes = torch.stack([x1, y1, x2, y2], dim=2)
    
    # Valid mask
    valid = scores > threshold
    
    return boxes, scores, valid


if __name__ == "__main__":
    # Test Detection Head
    print("Testing Detection Head...")
    
    # Single scale input
    x = torch.randn(2, 64, 52, 52)
    
    # Test AnchorFreeHead
    head = AnchorFreeHead(64, num_classes=1)
    hm, sz, off = head(x)
    
    print(f"Input: {x.shape}")
    print(f"Heatmap: {hm.shape} (range: {hm.min():.3f} - {hm.max():.3f})")
    print(f"Size: {sz.shape}")
    print(f"Offset: {off.shape}")
    
    # Test multi-scale DetectionHead
    features = (
        torch.randn(2, 64, 52, 52),
        torch.randn(2, 64, 26, 26),
        torch.randn(2, 64, 13, 13),
    )
    
    det_head = DetectionHead(64, num_classes=1, num_scales=3)
    outputs = det_head(features)
    
    print("\nMulti-scale outputs:")
    for i, (hm, sz, off) in enumerate(zip(
        outputs['heatmaps'], outputs['sizes'], outputs['offsets']
    )):
        print(f"  Scale {i}: heatmap={hm.shape}, size={sz.shape}, offset={off.shape}")
    
    # Test decoding
    boxes, scores, valid = decode_detections(
        hm, sz, off, stride=32, threshold=0.3
    )
    print(f"\nDecoded: boxes={boxes.shape}, scores={scores.shape}")
    
    print("✓ All Detection Head tests passed!")
