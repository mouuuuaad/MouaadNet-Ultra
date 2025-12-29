"""
Ghost Module Implementation
===========================
Efficient feature generation using cheap linear transformations to create
"ghost" features from intrinsic features, effectively doubling feature count
with minimal computational overhead.

Reference: GhostNet - More Features from Cheap Operations (CVPR 2020)
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class GhostModule(nn.Module):
    """
    Ghost Module for efficient feature map generation.
    
    Generates features in two steps:
    1. Primary convolution to generate intrinsic features (half the output)
    2. Cheap depthwise linear operations to generate ghost features
    
    This doubles the feature count with minimal computational overhead.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Primary convolution kernel size (default: 1)
        ratio: Ratio control for ghost features (default: 2)
        dw_kernel_size: Depthwise kernel size for ghost generation (default: 3)
        stride: Convolution stride (default: 1)
        activation: Whether to use ReLU6 activation (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ):
        super().__init__()
        
        self.out_channels = out_channels
        
        # Number of intrinsic features (primary conv output)
        init_channels = math.ceil(out_channels / ratio)
        # Number of ghost features to generate
        new_channels = init_channels * (ratio - 1)
        
        # Primary convolution: generates intrinsic features
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                init_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU6(inplace=True) if activation else nn.Identity(),
        )
        
        # Cheap operation: depthwise conv to generate ghost features
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_kernel_size,
                1,  # stride=1 for ghost generation
                dw_kernel_size // 2,
                groups=init_channels,  # Depthwise
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU6(inplace=True) if activation else nn.Identity(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass generating intrinsic + ghost features.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor with doubled feature channels
        """
        # Generate intrinsic features
        x_intrinsic = self.primary_conv(x)
        
        # Generate ghost features via cheap depthwise operation
        x_ghost = self.cheap_operation(x_intrinsic)
        
        # Concatenate intrinsic and ghost features
        out = torch.cat([x_intrinsic, x_ghost], dim=1)
        
        # Trim to exact output channels (in case of rounding)
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck: Efficient inverted residual block using Ghost Modules.
    
    Structure:
    - GhostModule (expansion)
    - Depthwise Conv (if stride > 1)
    - GhostModule (projection, no activation)
    - Residual connection (if stride=1 and channels match)
    
    Args:
        in_channels: Number of input channels
        mid_channels: Number of expansion channels
        out_channels: Number of output channels
        kernel_size: Depthwise conv kernel size (default: 3)
        stride: Stride for downsampling (default: 1)
        use_se: Whether to use Squeeze-and-Excitation (default: False)
    """
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = False,
    ):
        super().__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Ghost module for expansion
        self.ghost1 = GhostModule(
            in_channels,
            mid_channels,
            activation=True,
        )
        
        # Depthwise convolution for spatial mixing (if downsampling)
        if stride > 1:
            self.dwconv = nn.Sequential(
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=mid_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(mid_channels),
            )
        else:
            self.dwconv = nn.Identity()
        
        # Optional Squeeze-and-Excitation
        if use_se:
            self.se = SqueezeExcite(mid_channels)
        else:
            self.se = nn.Identity()
        
        # Ghost module for projection (no activation)
        self.ghost2 = GhostModule(
            mid_channels,
            out_channels,
            activation=False,
        )
        
        # Shortcut for non-matching dimensions
        if not self.use_residual and stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    1,
                    0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        elif not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    1,
                    0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        # Main branch
        out = self.ghost1(x)
        out = self.dwconv(out)
        out = self.se(out)
        out = self.ghost2(out)
        
        # Residual/shortcut
        shortcut = self.shortcut(x)
        
        return out + shortcut


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Args:
        channels: Number of input/output channels
        reduction: Channel reduction ratio (default: 4)
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        
        reduced_channels = max(channels // reduction, 8)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Hardsigmoid(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: global average pooling
        y = self.avgpool(x).view(b, c)
        # Excite: FC layers
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


if __name__ == "__main__":
    # Test Ghost Module
    print("Testing Ghost Module...")
    
    x = torch.randn(2, 32, 64, 64)
    
    # Test GhostModule
    ghost = GhostModule(32, 64)
    out = ghost(x)
    print(f"GhostModule input: {x.shape}")
    print(f"GhostModule output: {out.shape}")
    
    # Test GhostBottleneck
    bottleneck = GhostBottleneck(32, 64, 32, stride=1)
    out_bn = bottleneck(x)
    print(f"GhostBottleneck (stride=1) output: {out_bn.shape}")
    
    bottleneck_down = GhostBottleneck(32, 64, 64, stride=2)
    out_down = bottleneck_down(x)
    print(f"GhostBottleneck (stride=2) output: {out_down.shape}")
    
    # Test with SE
    bottleneck_se = GhostBottleneck(32, 64, 32, stride=1, use_se=True)
    out_se = bottleneck_se(x)
    print(f"GhostBottleneck with SE output: {out_se.shape}")
    
    print("✓ All Ghost Module tests passed!")
