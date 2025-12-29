"""
Cross-Stage Partial (CSP) Connections
=====================================
Partitions feature maps to reduce redundant gradient information
during training while maintaining feature richness.

Reference: CSPNet - A New Backbone that can Enhance Learning Capability of CNN
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvBNReLU(nn.Module):
    """Standard Conv + BatchNorm + ReLU6."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
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


class BottleneckCSP(nn.Module):
    """
    CSP Bottleneck block.
    
    Splits input into two paths:
    - Path 1: Goes through bottleneck blocks
    - Path 2: Direct connection (bypass)
    
    Both paths are concatenated and fused at the end.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_blocks: Number of bottleneck blocks in main branch
        expansion: Hidden channel expansion ratio (default: 0.5)
        shortcut: Whether to use residual in bottlenecks (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        expansion: float = 0.5,
        shortcut: bool = True,
    ):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        # Transition 1: Input to main branch
        self.conv1 = ConvBNReLU(in_channels, hidden_channels, 1)
        
        # Transition 2: Input to bypass branch  
        self.conv2 = ConvBNReLU(in_channels, hidden_channels, 1)
        
        # Main branch: stack of bottleneck blocks
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut)
            for _ in range(num_blocks)
        ])
        
        # Transition 3: After bottlenecks
        self.conv3 = ConvBNReLU(hidden_channels, hidden_channels, 1)
        
        # Fusion: Concatenated channels -> output
        self.conv4 = ConvBNReLU(hidden_channels * 2, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main branch through bottlenecks
        x1 = self.conv1(x)
        x1 = self.bottlenecks(x1)
        x1 = self.conv3(x1)
        
        # Bypass branch
        x2 = self.conv2(x)
        
        # Concatenate and fuse
        x = torch.cat([x1, x2], dim=1)
        x = self.conv4(x)
        
        return x


class Bottleneck(nn.Module):
    """
    Standard bottleneck block with optional residual.
    
    Structure: 1x1 conv -> 3x3 conv -> (optional residual)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
    ):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvBNReLU(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNReLU(hidden_channels, out_channels, 3)
        
        self.use_shortcut = shortcut and in_channels == out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        
        if self.use_shortcut:
            out = out + x
            
        return out


class CSPBlock(nn.Module):
    """
    Cross-Stage Partial block for efficient gradient flow.
    
    Partitions the feature map channel-wise:
    - Part 1 (ratio): Goes through transform blocks
    - Part 2 (1-ratio): Bypasses directly
    
    This reduces computation while maintaining gradient diversity.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_blocks: Number of transform blocks
        part_ratio: Ratio of channels for main branch (default: 0.5)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        part_ratio: float = 0.5,
    ):
        super().__init__()
        
        self.part_channels = int(in_channels * part_ratio)
        self.bypass_channels = in_channels - self.part_channels
        
        # Transform branch
        transform_layers = []
        for i in range(num_blocks):
            in_ch = self.part_channels if i == 0 else out_channels // 2
            transform_layers.append(
                nn.Sequential(
                    ConvBNReLU(in_ch, out_channels // 2, 1),
                    ConvBNReLU(out_channels // 2, out_channels // 2, 3),
                )
            )
        self.transform = nn.Sequential(*transform_layers)
        
        # Bypass projection (if needed)
        if self.bypass_channels != out_channels // 2:
            self.bypass_proj = ConvBNReLU(self.bypass_channels, out_channels // 2, 1)
        else:
            self.bypass_proj = nn.Identity()
            
        # Fusion
        self.fusion = ConvBNReLU(out_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split channels
        x1 = x[:, :self.part_channels, :, :]
        x2 = x[:, self.part_channels:, :, :]
        
        # Transform branch
        x1 = self.transform(x1)
        
        # Bypass branch
        x2 = self.bypass_proj(x2)
        
        # Concatenate and fuse
        x = torch.cat([x1, x2], dim=1)
        x = self.fusion(x)
        
        return x


class CSPStage(nn.Module):
    """
    Complete CSP stage with optional downsampling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_blocks: Number of internal blocks
        stride: Downsampling stride (1 or 2)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        stride: int = 1,
    ):
        super().__init__()
        
        # Downsampling (if stride > 1)
        if stride > 1:
            self.downsample = ConvBNReLU(in_channels, in_channels, 3, stride=stride)
        else:
            self.downsample = nn.Identity()
            
        # CSP block
        self.csp = BottleneckCSP(
            in_channels, out_channels, num_blocks,
            expansion=0.5, shortcut=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.csp(x)
        return x


if __name__ == "__main__":
    # Test CSP modules
    print("Testing CSP modules...")
    
    x = torch.randn(2, 64, 32, 32)
    
    # Test CSP Block
    csp = CSPBlock(64, 64, num_blocks=2)
    out = csp(x)
    print(f"CSPBlock input: {x.shape} -> output: {out.shape}")
    
    # Test BottleneckCSP
    bcsp = BottleneckCSP(64, 64, num_blocks=3)
    out_bcsp = bcsp(x)
    print(f"BottleneckCSP input: {x.shape} -> output: {out_bcsp.shape}")
    
    # Test CSP Stage with downsampling
    stage = CSPStage(64, 128, num_blocks=2, stride=2)
    out_stage = stage(x)
    print(f"CSPStage (stride=2) input: {x.shape} -> output: {out_stage.shape}")
    
    # Count parameters
    print(f"CSPBlock params: {sum(p.numel() for p in csp.parameters()):,}")
    print(f"BottleneckCSP params: {sum(p.numel() for p in bcsp.parameters()):,}")
    
    print("✓ All CSP tests passed!")
