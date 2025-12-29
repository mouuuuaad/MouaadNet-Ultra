"""
Structural Re-parameterization (RepVGG-style)
==============================================
Multi-branch training blocks that fuse into single 3x3 convolution
for ultra-fast inference without accuracy loss.

Training: 3x3 conv + 1x1 conv + identity (if channels match)
Inference: Single fused 3x3 conv

Reference: RepVGG - Making VGG-style ConvNets Great Again (CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import copy


class RepVGGBlock(nn.Module):
    """
    RepVGG Block with structural re-parameterization.
    
    Training mode: Multi-branch (3x3 + 1x1 + identity)
    Inference mode: Single fused 3x3 convolution
    
    Call .fuse() to convert to inference mode.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Convolution stride (default: 1)
        groups: Groups for grouped convolution (default: 1)
        deploy: Whether to use fused inference mode (default: False)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        deploy: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy
        
        # Padding for 3x3 conv to preserve spatial size
        padding = 1
        
        if deploy:
            # Inference mode: single fused conv
            self.rbr_reparam = nn.Conv2d(
                in_channels, out_channels, 3, stride, padding,
                groups=groups, bias=True
            )
        else:
            # Training mode: multi-branch
            self.rbr_reparam = None
            
            # Identity branch (only when channels match and stride=1)
            self.rbr_identity = (
                nn.BatchNorm2d(in_channels)
                if in_channels == out_channels and stride == 1
                else None
            )
            
            # 3x3 branch
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            
            # 1x1 branch
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        self.act = nn.ReLU6(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        
        # Multi-branch forward (training)
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        
        return self.act(out)
    
    def _get_equivalent_kernel_bias(self):
        """
        Compute equivalent 3x3 kernel and bias by fusing all branches.
        
        Returns:
            Tuple of (kernel, bias) tensors
        """
        # Get 3x3 branch kernel and bias
        kernel_3x3, bias_3x3 = self._fuse_bn_tensor(self.rbr_dense)
        
        # Get 1x1 branch kernel and bias (pad to 3x3)
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_1x1 = F.pad(kernel_1x1, [1, 1, 1, 1])  # Pad 1x1 to 3x3
        
        # Get identity branch (if exists)
        if self.rbr_identity is not None:
            kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
            kernel_id = F.pad(kernel_id, [1, 1, 1, 1])
        else:
            kernel_id = torch.zeros_like(kernel_3x3)
            bias_id = torch.zeros_like(bias_3x3)
        
        # Sum all branches
        return kernel_3x3 + kernel_1x1 + kernel_id, bias_3x3 + bias_1x1 + bias_id
    
    def _fuse_bn_tensor(self, branch) -> tuple:
        """
        Fuse Conv + BatchNorm into single Conv with bias.
        
        For identity branch (BN only), creates equivalent 3x3 identity kernel.
        """
        if branch is None:
            return 0, 0
        
        if isinstance(branch, nn.BatchNorm2d):
            # Identity branch: BN only
            # Create identity kernel
            kernel = torch.zeros(
                self.in_channels, self.in_channels // self.groups, 1, 1,
                device=branch.weight.device
            )
            for i in range(self.in_channels):
                kernel[i, i % (self.in_channels // self.groups), 0, 0] = 1
            
            # Fuse with BN
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        else:
            # Conv + BN branch
            if isinstance(branch, nn.Sequential):
                conv = branch[0]
                bn = branch[1]
            else:
                conv = branch
                bn = None
            
            kernel = conv.weight
            
            if bn is None:
                return kernel, conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
            
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
        
        # Fuse: W_fused = W * gamma / sqrt(var + eps)
        #       b_fused = beta - gamma * mean / sqrt(var + eps)
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        fused_kernel = kernel * t
        fused_bias = beta - gamma * running_mean / std
        
        return fused_kernel, fused_bias
    
    def fuse(self):
        """
        Convert training mode to inference mode.
        
        Fuses all branches into single 3x3 convolution.
        Call this before deployment for faster inference.
        """
        if self.deploy:
            return
        
        kernel, bias = self._get_equivalent_kernel_bias()
        
        self.rbr_reparam = nn.Conv2d(
            self.in_channels, self.out_channels, 3, self.stride, 1,
            groups=self.groups, bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        
        # Remove training branches
        for para in self.parameters():
            para.detach_()
        
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity') and self.rbr_identity is not None:
            self.__delattr__('rbr_identity')
        
        self.deploy = True


class RepBlock(nn.Module):
    """
    Simpler Rep Block with optional SE attention.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Convolution stride
        use_se: Whether to use Squeeze-and-Excitation
        deploy: Whether to use fused inference mode
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = False,
        deploy: bool = False,
    ):
        super().__init__()
        
        self.rep_block = RepVGGBlock(
            in_channels, out_channels, stride, deploy=deploy
        )
        
        if use_se:
            self.se = SqueezeExcite(out_channels)
        else:
            self.se = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rep_block(x)
        x = self.se(x)
        return x
    
    def fuse(self):
        self.rep_block.fuse()


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        reduced = max(channels // reduction, 8)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Hardsigmoid(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


def fuse_model(model: nn.Module) -> nn.Module:
    """
    Fuse all RepVGG blocks in a model for inference.
    
    Recursively finds and fuses all RepVGGBlock and RepBlock modules.
    
    Args:
        model: Model to fuse
        
    Returns:
        Fused model (modified in-place)
    """
    for module in model.modules():
        if hasattr(module, 'fuse'):
            module.fuse()
    return model


if __name__ == "__main__":
    # Test RepVGG Block
    print("Testing RepVGG Block...")
    
    x = torch.randn(2, 64, 32, 32)
    
    # Training mode
    block_train = RepVGGBlock(64, 64, stride=1, deploy=False)
    block_train.eval()
    out_train = block_train(x)
    print(f"Training mode output: {out_train.shape}")
    
    # Fuse to inference mode
    block_infer = copy.deepcopy(block_train)
    block_infer.fuse()
    out_infer = block_infer(x)
    print(f"Inference mode output: {out_infer.shape}")
    
    # Verify outputs match
    diff = (out_train - out_infer).abs().max()
    print(f"Max difference after fusion: {diff:.6f}")
    assert diff < 1e-4, "Fusion error!"
    
    # Compare parameter counts
    train_params = sum(p.numel() for p in block_train.parameters())
    infer_params = sum(p.numel() for p in block_infer.parameters())
    print(f"Training params: {train_params:,}")
    print(f"Inference params: {infer_params:,}")
    
    # Test with stride
    block_stride = RepVGGBlock(64, 128, stride=2, deploy=False)
    out_stride = block_stride(x)
    print(f"Stride=2 output: {out_stride.shape}")
    
    # Test RepBlock with SE
    rep_se = RepBlock(64, 64, use_se=True, deploy=False)
    out_se = rep_se(x)
    print(f"RepBlock with SE: {out_se.shape}")
    
    print("✓ All RepVGG tests passed!")
