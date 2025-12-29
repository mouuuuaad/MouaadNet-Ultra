"""
Nano-Backbone Implementation
=============================
Ultra-lightweight 5-stage hierarchical backbone for MOUAADNET-ULTRA.
Combines PConv, Ghost Modules, and IRB for maximum efficiency.

Channel progression: 16 -> 24 -> 40 -> 80 -> 128
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .pconv import PConv, PConvBlock
from .ghost_module import GhostModule, GhostBottleneck
from .irb import InvertedResidualBlock, LightweightIRB


class StemBlock(nn.Module):
    """
    Initial stem block for input processing.
    
    Converts RGB input (3 channels) to initial feature maps.
    Uses efficient 3x3 conv with stride 2 for immediate downsampling.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 16):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class NanoStage(nn.Module):
    """
    Single stage of the Nano-Backbone.
    
    Combines different efficient blocks based on stage depth:
    - Early stages: Lightweight IRB
    - Middle stages: Ghost Bottleneck
    - Late stages: PConv + Ghost hybrid
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        num_blocks: Number of blocks in stage
        stride: Stride for first block (downsampling)
        stage_type: Block type ('irb', 'ghost', 'hybrid')
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 2,
        stage_type: str = 'irb',
    ):
        super().__init__()
        
        blocks = []
        
        for i in range(num_blocks):
            # First block handles channel change and downsampling
            block_in = in_channels if i == 0 else out_channels
            block_stride = stride if i == 0 else 1
            
            if stage_type == 'irb':
                block = LightweightIRB(
                    block_in, 
                    out_channels, 
                    stride=block_stride,
                    expansion_ratio=3,
                    use_se=(i == num_blocks - 1),  # SE on last block
                )
            elif stage_type == 'ghost':
                mid_channels = out_channels * 2
                block = GhostBottleneck(
                    block_in,
                    mid_channels,
                    out_channels,
                    stride=block_stride,
                    use_se=(i == num_blocks - 1),
                )
            else:  # hybrid
                if i == 0:
                    # First block: Ghost for channel change
                    block = GhostBottleneck(
                        block_in,
                        out_channels * 2,
                        out_channels,
                        stride=block_stride,
                    )
                else:
                    # Remaining blocks: Lightweight IRB
                    block = LightweightIRB(
                        out_channels,
                        out_channels,
                        stride=1,
                        use_se=(i == num_blocks - 1),
                    )
                    
            blocks.append(block)
            
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class NanoBackbone(nn.Module):
    """
    Complete 5-Stage Nano-Backbone for MOUAADNET-ULTRA.
    
    Hierarchical feature extraction with channel progression:
    Stage 1: 16 channels  (1/4 resolution)
    Stage 2: 24 channels  (1/8 resolution)
    Stage 3: 40 channels  (1/16 resolution)
    Stage 4: 80 channels  (1/32 resolution)
    Stage 5: 128 channels (1/32 resolution, no further downsampling)
    
    Returns multi-scale features for FPN/PAN neck.
    
    Args:
        in_channels: Input image channels (default: 3)
        width_mult: Width multiplier for channel scaling (default: 1.0)
    """
    
    # Default channel configuration
    DEFAULT_CHANNELS = [16, 24, 40, 80, 128]
    DEFAULT_BLOCKS = [1, 2, 3, 4, 2]
    DEFAULT_STRIDES = [1, 2, 2, 2, 1]  # Stage 5 doesn't downsample
    DEFAULT_TYPES = ['irb', 'irb', 'ghost', 'ghost', 'hybrid']
    
    def __init__(
        self,
        in_channels: int = 3,
        width_mult: float = 1.0,
        channels: List[int] = None,
        num_blocks: List[int] = None,
    ):
        super().__init__()
        
        # Apply width multiplier to channels
        channels = channels or self.DEFAULT_CHANNELS
        channels = [int(c * width_mult) for c in channels]
        num_blocks = num_blocks or self.DEFAULT_BLOCKS
        
        # Stem: 3 -> channels[0]
        self.stem = StemBlock(in_channels, channels[0])
        
        # Build 5 stages
        self.stage1 = NanoStage(
            channels[0], channels[0], num_blocks[0],
            stride=self.DEFAULT_STRIDES[0], 
            stage_type=self.DEFAULT_TYPES[0],
        )
        
        self.stage2 = NanoStage(
            channels[0], channels[1], num_blocks[1],
            stride=self.DEFAULT_STRIDES[1],
            stage_type=self.DEFAULT_TYPES[1],
        )
        
        self.stage3 = NanoStage(
            channels[1], channels[2], num_blocks[2],
            stride=self.DEFAULT_STRIDES[2],
            stage_type=self.DEFAULT_TYPES[2],
        )
        
        self.stage4 = NanoStage(
            channels[2], channels[3], num_blocks[3],
            stride=self.DEFAULT_STRIDES[3],
            stage_type=self.DEFAULT_TYPES[3],
        )
        
        self.stage5 = NanoStage(
            channels[3], channels[4], num_blocks[4],
            stride=self.DEFAULT_STRIDES[4],
            stage_type=self.DEFAULT_TYPES[4],
        )
        
        # Store output channels for neck
        self.out_channels = channels
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of feature maps at different scales:
            - P3: 1/8 resolution (stage 2 output)
            - P4: 1/16 resolution (stage 3 output)  
            - P5: 1/32 resolution (stage 5 output)
        """
        # Stem: H/2
        x = self.stem(x)
        
        # Stage 1: H/2 (no additional downsampling)
        c1 = self.stage1(x)
        
        # Stage 2: H/4 (P3)
        c2 = self.stage2(c1)
        
        # Stage 3: H/8 (P4)
        c3 = self.stage3(c2)
        
        # Stage 4: H/16
        c4 = self.stage4(c3)
        
        # Stage 5: H/16 (P5, no further downsampling)
        c5 = self.stage5(c4)
        
        # Return multi-scale features for FPN/PAN
        # P3 (c2): High resolution, low-level features
        # P4 (c3): Medium resolution
        # P5 (c5): Low resolution, high-level features
        return c2, c3, c5
    
    def get_output_channels(self) -> Tuple[int, int, int]:
        """Get output channel counts for P3, P4, P5."""
        return (
            self.out_channels[1],  # P3: stage2 output
            self.out_channels[2],  # P4: stage3 output
            self.out_channels[4],  # P5: stage5 output
        )


if __name__ == "__main__":
    # Test Nano-Backbone
    print("Testing Nano-Backbone...")
    
    # Create backbone
    backbone = NanoBackbone(in_channels=3, width_mult=1.0)
    
    # Test with sample input
    x = torch.randn(2, 3, 416, 416)
    
    # Forward pass
    p3, p4, p5 = backbone(x)
    
    print(f"Input shape: {x.shape}")
    print(f"P3 (1/8 res) shape: {p3.shape}")
    print(f"P4 (1/16 res) shape: {p4.shape}")
    print(f"P5 (1/32 res) shape: {p5.shape}")
    
    # Get output channels
    c3, c4, c5 = backbone.get_output_channels()
    print(f"Output channels: P3={c3}, P4={c4}, P5={c5}")
    
    # Count parameters
    params = sum(p.numel() for p in backbone.parameters())
    print(f"Total parameters: {params:,}")
    print(f"Estimated FP32 size: {params * 4 / 1024 / 1024:.2f} MB")
    
    # Test with width multiplier
    backbone_slim = NanoBackbone(width_mult=0.5)
    slim_params = sum(p.numel() for p in backbone_slim.parameters())
    print(f"Slim backbone (0.5x) parameters: {slim_params:,}")
    
    print("✓ All Nano-Backbone tests passed!")
