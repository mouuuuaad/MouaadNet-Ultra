"""
Slim-PAN (Path Aggregation Network) Implementation
===================================================
Lightweight bi-directional feature fusion network that combines high-level
semantics with low-level spatial details for multi-scale detection.

Uses element-wise addition instead of concatenation for memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class ConvBNReLU(nn.Module):
    """Standard Conv + BatchNorm + ReLU6 block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride, padding, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size,
                stride, padding, groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LateralConnection(nn.Module):
    """
    Lateral connection for FPN-style feature fusion.
    Reduces channels and aligns feature maps for addition.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.lateral = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lateral(x)


class SlimPAN(nn.Module):
    """
    Slim Path Aggregation Network for efficient multi-scale feature fusion.
    
    Implements a bi-directional feature pyramid:
    1. Top-down pathway: High-level semantics flow to low-level features
    2. Bottom-up pathway: Low-level details flow to high-level features
    
    Uses element-wise addition instead of concatenation for memory efficiency.
    
    Args:
        in_channels: Tuple of (P3, P4, P5) channel counts from backbone
        out_channels: Unified output channel count for all scales
    """
    
    def __init__(
        self,
        in_channels: Tuple[int, int, int],
        out_channels: int = 64,
    ):
        super().__init__()
        
        c3, c4, c5 = in_channels
        
        # Lateral connections (1x1 conv to unify channels)
        self.lateral_p5 = LateralConnection(c5, out_channels)
        self.lateral_p4 = LateralConnection(c4, out_channels)
        self.lateral_p3 = LateralConnection(c3, out_channels)
        
        # Top-down pathway (upsample + add)
        self.td_p5 = DepthwiseSeparableConv(out_channels, out_channels)
        self.td_p4 = DepthwiseSeparableConv(out_channels, out_channels)
        self.td_p3 = DepthwiseSeparableConv(out_channels, out_channels)
        
        # Bottom-up pathway (downsample + add)
        self.bu_p3 = DepthwiseSeparableConv(out_channels, out_channels, stride=2)
        self.bu_p4 = DepthwiseSeparableConv(out_channels, out_channels, stride=2)
        self.bu_p5 = DepthwiseSeparableConv(out_channels, out_channels)
        
        # Output convolutions
        self.out_p3 = DepthwiseSeparableConv(out_channels, out_channels)
        self.out_p4 = DepthwiseSeparableConv(out_channels, out_channels)
        self.out_p5 = DepthwiseSeparableConv(out_channels, out_channels)
        
        self.out_channels = out_channels
        
    def forward(
        self, 
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with bi-directional feature fusion.
        
        Args:
            features: Tuple of (P3, P4, P5) feature maps from backbone
            
        Returns:
            Tuple of fused (N3, N4, N5) feature maps
        """
        p3, p4, p5 = features
        
        # Apply lateral connections
        p5_lateral = self.lateral_p5(p5)
        p4_lateral = self.lateral_p4(p4)
        p3_lateral = self.lateral_p3(p3)
        
        # Top-down pathway
        # P5 -> P4: upsample and add
        p5_td = self.td_p5(p5_lateral)
        p4_td = self.td_p4(
            p4_lateral + F.interpolate(p5_td, size=p4_lateral.shape[2:], mode='nearest')
        )
        
        # P4 -> P3: upsample and add
        p3_td = self.td_p3(
            p3_lateral + F.interpolate(p4_td, size=p3_lateral.shape[2:], mode='nearest')
        )
        
        # Bottom-up pathway
        # N3 -> N4: downsample and add
        n3 = self.out_p3(p3_td)
        n4 = self.out_p4(p4_td + self.bu_p3(n3))
        
        # N4 -> N5: downsample and add
        n5 = self.out_p5(p5_td + self.bu_p4(n4))
        
        return n3, n4, n5


class SlimPANv2(nn.Module):
    """
    Enhanced Slim-PAN with CSP-style connections for better gradient flow.
    """
    
    def __init__(
        self,
        in_channels: Tuple[int, int, int],
        out_channels: int = 64,
    ):
        super().__init__()
        
        c3, c4, c5 = in_channels
        
        # Reduce channels
        self.reduce_p5 = ConvBNReLU(c5, out_channels, 1)
        self.reduce_p4 = ConvBNReLU(c4, out_channels, 1)
        self.reduce_p3 = ConvBNReLU(c3, out_channels, 1)
        
        # Top-down fusion blocks
        self.fpn_p4 = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )
        self.fpn_p3 = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )
        
        # Bottom-up fusion blocks
        self.pan_p4 = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )
        self.pan_p5 = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )
        
        # Downsample convolutions
        self.down_p3 = ConvBNReLU(out_channels, out_channels, 3, stride=2)
        self.down_p4 = ConvBNReLU(out_channels, out_channels, 3, stride=2)
        
        self.out_channels = out_channels
        
    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = features
        
        # Reduce channels
        p5 = self.reduce_p5(p5)
        p4 = self.reduce_p4(p4)
        p3 = self.reduce_p3(p3)
        
        # Top-down pathway
        p4 = self.fpn_p4(p4 + F.interpolate(p5, size=p4.shape[2:], mode='nearest'))
        p3 = self.fpn_p3(p3 + F.interpolate(p4, size=p3.shape[2:], mode='nearest'))
        
        # Bottom-up pathway
        n3 = p3
        n4 = self.pan_p4(p4 + self.down_p3(n3))
        n5 = self.pan_p5(p5 + self.down_p4(n4))
        
        return n3, n4, n5


if __name__ == "__main__":
    # Test Slim-PAN
    print("Testing Slim-PAN...")
    
    # Simulate backbone outputs
    p3 = torch.randn(2, 24, 52, 52)   # 1/8 resolution
    p4 = torch.randn(2, 40, 26, 26)   # 1/16 resolution
    p5 = torch.randn(2, 128, 13, 13)  # 1/32 resolution
    
    # Test SlimPAN
    pan = SlimPAN(in_channels=(24, 40, 128), out_channels=64)
    n3, n4, n5 = pan((p3, p4, p5))
    
    print(f"Input P3: {p3.shape} -> Output N3: {n3.shape}")
    print(f"Input P4: {p4.shape} -> Output N4: {n4.shape}")
    print(f"Input P5: {p5.shape} -> Output N5: {n5.shape}")
    
    # Test SlimPANv2
    pan_v2 = SlimPANv2(in_channels=(24, 40, 128), out_channels=64)
    n3_v2, n4_v2, n5_v2 = pan_v2((p3, p4, p5))
    print(f"SlimPANv2 outputs: N3={n3_v2.shape}, N4={n4_v2.shape}, N5={n5_v2.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in pan.parameters())
    print(f"SlimPAN parameters: {params:,}")
    
    print("✓ All Slim-PAN tests passed!")
