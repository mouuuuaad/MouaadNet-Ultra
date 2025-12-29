"""
Inverted Residual Block (IRB) Implementation
=============================================
MobileNetV2-style inverted residual block with expansion-depthwise-projection
pattern for efficient feature extraction on edge devices.

Reference: MobileNetV2 - Inverted Residuals and Linear Bottlenecks (CVPR 2018)
"""

import torch
import torch.nn as nn
from typing import Optional


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block (IRB)
    
    Structure:
    1. Expansion: 1x1 conv to expand channels
    2. Depthwise: 3x3 depthwise separable conv for spatial features
    3. Projection: 1x1 conv to reduce channels (linear, no activation)
    
    Uses residual connection when input/output dimensions match.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Convolution stride (1 or 2)
        expansion_ratio: Channel expansion ratio (default: 6)
        kernel_size: Depthwise conv kernel size (default: 3)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion_ratio: int = 6,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expanded channel count
        expanded_channels = in_channels * expansion_ratio
        
        layers = []
        
        # Expansion phase (skip if expansion_ratio == 1)
        if expansion_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(inplace=True),
            ])
        
        # Depthwise convolution phase
        padding = kernel_size // 2
        layers.extend([
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size,
                stride,
                padding,
                groups=expanded_channels,  # Depthwise
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
        ])
        
        # Projection phase (linear - no activation)
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        out = self.conv(x)
        
        if self.use_residual:
            out = out + x
            
        return out


class IRBStack(nn.Module):
    """
    Stack of Inverted Residual Blocks.
    
    First block can have stride for downsampling, remaining blocks have stride=1.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_blocks: Number of IRB blocks
        stride: Stride for first block (default: 1)
        expansion_ratio: Channel expansion ratio (default: 6)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        expansion_ratio: int = 6,
    ):
        super().__init__()
        
        blocks = []
        
        # First block (may downsample)
        blocks.append(
            InvertedResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                expansion_ratio=expansion_ratio,
            )
        )
        
        # Remaining blocks (stride=1)
        for _ in range(1, num_blocks):
            blocks.append(
                InvertedResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    expansion_ratio=expansion_ratio,
                )
            )
            
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class LightweightIRB(nn.Module):
    """
    Lightweight Inverted Residual Block with reduced expansion.
    
    Uses smaller expansion ratio (3 instead of 6) and optional
    Squeeze-and-Excitation for better efficiency.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Convolution stride
        expansion_ratio: Channel expansion ratio (default: 3)
        use_se: Whether to use Squeeze-and-Excitation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion_ratio: int = 3,
        use_se: bool = False,
    ):
        super().__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        expanded_channels = in_channels * expansion_ratio
        
        # Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
        ) if expansion_ratio != 1 else nn.Identity()
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                3,
                stride,
                1,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
        )
        
        # Squeeze-and-Excitation (optional)
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, expanded_channels // 4, 1),
                nn.ReLU6(inplace=True),
                nn.Conv2d(expanded_channels // 4, expanded_channels, 1),
                nn.Hardsigmoid(inplace=True),
            )
        else:
            self.se = None
            
        # Projection (linear)
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.expanded_channels = expanded_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x) if hasattr(self.expand, '__call__') else x
        out = self.depthwise(out)
        
        if self.se is not None:
            out = out * self.se(out)
            
        out = self.project(out)
        
        if self.use_residual:
            out = out + x
            
        return out


if __name__ == "__main__":
    # Test Inverted Residual Block
    print("Testing Inverted Residual Block...")
    
    x = torch.randn(2, 32, 64, 64)
    
    # Test basic IRB
    irb = InvertedResidualBlock(32, 32, stride=1)
    out = irb(x)
    print(f"IRB (stride=1, same channels) output: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch for stride=1"
    
    # Test IRB with stride
    irb_stride = InvertedResidualBlock(32, 64, stride=2)
    out_stride = irb_stride(x)
    print(f"IRB (stride=2) output: {out_stride.shape}")
    assert out_stride.shape == (2, 64, 32, 32), "Shape mismatch for stride=2"
    
    # Test IRB stack
    stack = IRBStack(32, 64, num_blocks=3, stride=2)
    out_stack = stack(x)
    print(f"IRB Stack (3 blocks, stride=2) output: {out_stack.shape}")
    
    # Test Lightweight IRB
    light_irb = LightweightIRB(32, 32, stride=1, use_se=True)
    out_light = light_irb(x)
    print(f"Lightweight IRB with SE output: {out_light.shape}")
    
    print("✓ All IRB tests passed!")
