"""
Partial Convolution (PConv) Implementation
==========================================
Core building block for MOUAADNET-ULTRA that applies convolutions to only
a subset of input channels (default 25%), significantly reducing FLOPs
and memory access overhead.

Reference: FasterNet - Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks
"""

import torch
import torch.nn as nn
from typing import Optional


class PConv(nn.Module):
    """
    Partial Convolution Block
    
    Applies standard convolution to only a fraction of input channels,
    leaving the rest untouched. This reduces redundant memory access
    and computational overhead while maintaining feature richness.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (for the partial portion)
        kernel_size: Convolution kernel size (default: 3)
        stride: Convolution stride (default: 1)
        partial_ratio: Fraction of channels to convolve (default: 0.25)
        bias: Whether to use bias in convolution (default: False)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        partial_ratio: float = 0.25,
        bias: bool = False,
    ):
        super().__init__()
        
        # Calculate partial channel count
        self.partial_channels = int(in_channels * partial_ratio)
        self.untouched_channels = in_channels - self.partial_channels
        
        # Output channels defaults to partial channels if not specified
        out_channels = out_channels or self.partial_channels
        
        # Padding for same spatial size
        padding = kernel_size // 2
        
        # Convolution applied only to partial channels
        self.conv = nn.Conv2d(
            self.partial_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with partial convolution.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor with convolved partial channels + untouched channels
        """
        # Split channels into partial and untouched
        x_partial = x[:, :self.partial_channels, :, :]
        x_untouched = x[:, self.partial_channels:, :, :]
        
        # Apply convolution only to partial channels
        x_partial = self.conv(x_partial)
        
        # Handle stride for untouched channels (downsample if needed)
        if self.stride > 1:
            x_untouched = x_untouched[:, :, ::self.stride, ::self.stride]
        
        # Concatenate convolved and untouched channels
        return torch.cat([x_partial, x_untouched], dim=1)


class PConvBlock(nn.Module):
    """
    Complete PConv Block with normalization and activation.
    
    Structure: PConv -> BatchNorm -> ReLU6
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        partial_ratio: Fraction of channels to convolve
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        partial_ratio: float = 0.25,
    ):
        super().__init__()
        
        self.pconv = PConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            partial_ratio=partial_ratio,
        )
        
        # Calculate actual output channels
        partial_out = out_channels or int(in_channels * partial_ratio)
        untouched = in_channels - int(in_channels * partial_ratio)
        total_out = partial_out + untouched
        
        self.bn = nn.BatchNorm2d(total_out)
        self.act = nn.ReLU6(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


if __name__ == "__main__":
    # Test PConv
    print("Testing Partial Convolution...")
    
    # Create sample input: batch=2, channels=64, height=32, width=32
    x = torch.randn(2, 64, 32, 32)
    
    # Test basic PConv
    pconv = PConv(in_channels=64, partial_ratio=0.25)
    out = pconv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Partial channels: {pconv.partial_channels}")
    print(f"Untouched channels: {pconv.untouched_channels}")
    
    # Test PConv with stride
    pconv_stride = PConv(in_channels=64, stride=2, partial_ratio=0.25)
    out_stride = pconv_stride(x)
    print(f"Output with stride=2: {out_stride.shape}")
    
    # Test PConvBlock
    block = PConvBlock(in_channels=64, partial_ratio=0.25)
    out_block = block(x)
    print(f"PConvBlock output: {out_block.shape}")
    
    print("✓ All PConv tests passed!")
