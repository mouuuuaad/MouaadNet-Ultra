"""
SPP-Lite (Spatial Pyramid Pooling - Lightweight)
=================================================
Lightweight multi-scale context aggregation module that captures
features at different scales without changing resolution.

Uses smaller pooling kernels than standard SPP for efficiency.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class SPPLite(nn.Module):
    """
    Lightweight Spatial Pyramid Pooling block.
    
    Applies multiple max pooling operations with different kernel sizes
    to capture multi-scale context, then concatenates the results.
    
    This variant uses smaller kernel sizes (3, 5, 7) compared to
    standard SPP (5, 9, 13) for faster inference on edge devices.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        pool_sizes: Tuple of pooling kernel sizes (default: (3, 5, 7))
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        pool_sizes: Tuple[int, ...] = (3, 5, 7),
    ):
        super().__init__()
        
        out_channels = out_channels or in_channels
        
        # Pre-convolution to reduce channels before pooling
        hidden_channels = in_channels // 2
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
        )
        
        # Max pooling layers with same padding to preserve resolution
        self.pools = nn.ModuleList([
            nn.MaxPool2d(
                kernel_size=size,
                stride=1,
                padding=size // 2,
            )
            for size in pool_sizes
        ])
        
        # Post-convolution to fuse pooled features
        # Concatenates: original + len(pool_sizes) pooled versions
        total_channels = hidden_channels * (1 + len(pool_sizes))
        
        self.post_conv = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale pooling.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor with multi-scale context
        """
        # Reduce channels
        x = self.pre_conv(x)
        
        # Apply pooling at multiple scales
        pooled = [x] + [pool(x) for pool in self.pools]
        
        # Concatenate all scales
        x = torch.cat(pooled, dim=1)
        
        # Fuse features
        x = self.post_conv(x)
        
        return x


class SPPFast(nn.Module):
    """
    Fast SPP using sequential 3x3 max pooling.
    
    Instead of parallel pooling with different sizes, uses sequential
    3x3 pooling which achieves similar receptive field expansion
    but with better GPU utilization.
    
    3x3 -> 5x5 effective -> 7x7 effective (through cascade)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        
        out_channels = out_channels or in_channels
        hidden_channels = in_channels // 2
        
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
        )
        
        # Single pooling operation, applied sequentially
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # 4 scales: original + 3 pooled (3x3, 5x5 effective, 7x7 effective)
        self.post_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)
        
        # Sequential pooling for cascaded receptive field
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        
        # Concatenate all scales
        x = torch.cat([x, y1, y2, y3], dim=1)
        
        return self.post_conv(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (lite version).
    
    Uses dilated convolutions instead of max pooling to capture
    multi-scale context with learnable parameters.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dilations: Tuple of dilation rates (default: (1, 2, 4, 8))
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        
        out_channels = out_channels or in_channels
        hidden_channels = in_channels // 4
        
        # Parallel dilated convolutions
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels, hidden_channels, 3,
                    padding=d, dilation=d, bias=False
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
            )
            for d in dilations
        ])
        
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
        )
        
        # Fusion convolution
        total_channels = hidden_channels * (len(dilations) + 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        
        # Apply dilated convolutions
        features = [branch(x) for branch in self.branches]
        
        # Global pooling and upsample
        global_feat = self.global_pool(x)
        global_feat = torch.nn.functional.interpolate(
            global_feat, size=size, mode='bilinear', align_corners=False
        )
        features.append(global_feat)
        
        # Concatenate and fuse
        x = torch.cat(features, dim=1)
        x = self.fusion(x)
        
        return x


if __name__ == "__main__":
    # Test SPP modules
    print("Testing SPP-Lite modules...")
    
    x = torch.randn(2, 128, 13, 13)
    
    # Test SPPLite
    spp = SPPLite(128, 128)
    out = spp(x)
    print(f"SPPLite input: {x.shape} -> output: {out.shape}")
    
    # Test SPPFast
    sppf = SPPFast(128, 128)
    out_fast = sppf(x)
    print(f"SPPFast input: {x.shape} -> output: {out_fast.shape}")
    
    # Test ASPP
    aspp = ASPP(128, 128)
    out_aspp = aspp(x)
    print(f"ASPP input: {x.shape} -> output: {out_aspp.shape}")
    
    # Compare parameters
    print(f"SPPLite params: {sum(p.numel() for p in spp.parameters()):,}")
    print(f"SPPFast params: {sum(p.numel() for p in sppf.parameters()):,}")
    print(f"ASPP params: {sum(p.numel() for p in aspp.parameters()):,}")
    
    print("✓ All SPP tests passed!")
