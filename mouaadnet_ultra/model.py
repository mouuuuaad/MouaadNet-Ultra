"""
MOUAADNET-ULTRA: Complete Neural Network
==========================================
High-Efficiency Human Detection and Gender Classification

Lead Architect: MOUAAD IDOUFKIR
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List

from .backbone import NanoBackbone
from .neck import SlimPAN, SPPLite
from .heads import DecoupledHead
from .optim.rep_block import fuse_model


class MouaadNetUltra(nn.Module):
    """
    MOUAADNET-ULTRA: Unified Human Detection + Gender Classification.
    
    Architecture:
    - Backbone: 5-stage Nano-Backbone (PConv + Ghost + IRB)
    - Neck: Slim-PAN with SPP-Lite for multi-scale fusion
    - Head: Decoupled detection + classification heads
    
    Targets:
    - Inference: < 10ms on GPU
    - Model size: ~2-3MB (INT8 quantized)
    - Accuracy: State-of-the-art for lightweight models
    
    Args:
        in_channels: Input image channels (default: 3)
        width_mult: Width multiplier for scaling (default: 1.0)
        num_classes: Number of detection classes (default: 1 for person)
        neck_channels: Unified neck channel count (default: 64)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        width_mult: float = 1.0,
        num_classes: int = 1,
        neck_channels: int = 64,
    ):
        super().__init__()
        
        # Backbone: Multi-scale feature extraction
        self.backbone = NanoBackbone(
            in_channels=in_channels,
            width_mult=width_mult,
        )
        
        # Get backbone output channels
        p3_ch, p4_ch, p5_ch = self.backbone.get_output_channels()
        
        # SPP at backbone end for multi-scale context
        self.spp = SPPLite(p5_ch, p5_ch)
        
        # Neck: Feature fusion
        self.neck = SlimPAN(
            in_channels=(p3_ch, p4_ch, p5_ch),
            out_channels=neck_channels,
        )
        
        # Head: Detection + Gender Classification
        self.head = DecoupledHead(
            in_channels=neck_channels,
            det_channels=neck_channels,
            cls_channels=128,
            num_classes=num_classes,
        )
        
        # Store config
        self.config = {
            'in_channels': in_channels,
            'width_mult': width_mult,
            'num_classes': num_classes,
            'neck_channels': neck_channels,
        }
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
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
    
    def forward(
        self,
        x: torch.Tensor,
        person_crops: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for detection and classification.
        
        Args:
            x: Input images (B, 3, H, W)
            person_crops: Optional cropped person features for gender classification
            
        Returns:
            Dict containing:
            - heatmaps: List of detection heatmaps per scale
            - sizes: List of bbox size predictions per scale
            - offsets: List of offset predictions per scale
            - gender: Gender logits (B, 1)
        """
        # Backbone: Extract multi-scale features
        p3, p4, p5 = self.backbone(x)
        
        # SPP on highest level
        p5 = self.spp(p5)
        
        # Neck: Feature fusion
        n3, n4, n5 = self.neck((p3, p4, p5))
        
        # Head: Detection + Classification
        outputs = self.head((n3, n4, n5), person_crops)
        
        return outputs
    
    def detect(
        self,
        x: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Detection-only forward pass (faster for inference).
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Detection outputs (heatmaps, sizes, offsets)
        """
        # Backbone
        p3, p4, p5 = self.backbone(x)
        p5 = self.spp(p5)
        
        # Neck
        n3, n4, n5 = self.neck((p3, p4, p5))
        
        # Detection only
        return self.head.detect_only((n3, n4, n5))
    
    def classify_gender(
        self,
        crops: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gender classification on cropped person regions.
        
        Args:
            crops: Cropped person features (N, C, H, W)
            
        Returns:
            Gender logits (N, 1)
        """
        # Process crops through backbone stages
        # For efficiency, we extract from the adapter in head
        return self.head.classify_gender(crops)
    
    def fuse_for_inference(self):
        """
        Fuse RepVGG blocks for faster inference.
        
        Call this before deployment to convert multi-branch
        structures into single convolutions.
        """
        fuse_model(self)
        return self
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self, precision: str = 'fp32') -> float:
        """
        Estimate model size in MB.
        
        Args:
            precision: 'fp32', 'fp16', or 'int8'
            
        Returns:
            Estimated model size in MB
        """
        params = self.count_parameters()
        
        bytes_per_param = {
            'fp32': 4,
            'fp16': 2,
            'int8': 1,
        }
        
        bytes_count = params * bytes_per_param.get(precision, 4)
        return bytes_count / (1024 ** 2)


class MouaadNetUltraLite(MouaadNetUltra):
    """
    Lightweight variant of MOUAADNET-ULTRA.
    
    Uses 0.5x width multiplier and reduced neck channels
    for even faster inference on edge devices.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
    ):
        super().__init__(
            in_channels=in_channels,
            width_mult=0.5,
            num_classes=num_classes,
            neck_channels=48,
        )


class MouaadNetUltraPro(MouaadNetUltra):
    """
    High-accuracy variant of MOUAADNET-ULTRA.
    
    Uses 1.5x width multiplier and increased neck channels
    for better accuracy at the cost of speed.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
    ):
        super().__init__(
            in_channels=in_channels,
            width_mult=1.5,
            num_classes=num_classes,
            neck_channels=96,
        )


def load_model(
    checkpoint_path: str,
    device: str = 'cuda',
    fuse: bool = True,
) -> MouaadNetUltra:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        fuse: Whether to fuse for inference
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    config = checkpoint.get('config', {})
    
    # Create model
    model = MouaadNetUltra(**config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Fuse for inference
    if fuse:
        model.fuse_for_inference()
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("MOUAADNET-ULTRA: Complete Model Test")
    print("Lead Architect: MOUAAD IDOUFKIR")
    print("=" * 60)
    
    # Create model
    model = MouaadNetUltra()
    
    # Test input
    x = torch.randn(1, 3, 416, 416)
    
    # Forward pass
    print("\n1. Forward Pass Test:")
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"   Heatmaps: {[h.shape for h in outputs['heatmaps']]}")
    print(f"   Sizes: {[s.shape for s in outputs['sizes']]}")
    print(f"   Offsets: {[o.shape for o in outputs['offsets']]}")
    print(f"   Gender: {outputs['gender'].shape}")
    
    # Model statistics
    print("\n2. Model Statistics:")
    params = model.count_parameters()
    print(f"   Parameters: {params:,}")
    print(f"   FP32 Size: {model.get_model_size_mb('fp32'):.2f} MB")
    print(f"   FP16 Size: {model.get_model_size_mb('fp16'):.2f} MB")
    print(f"   INT8 Size: {model.get_model_size_mb('int8'):.2f} MB")
    
    # Benchmark (CPU)
    print("\n3. CPU Inference Benchmark:")
    model.eval()
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(x)
    
    # Benchmark
    times = []
    for _ in range(20):
        start = time.time()
        with torch.no_grad():
            _ = model(x)
        times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    print(f"   Average: {avg_time:.2f} ms")
    print(f"   Min: {min(times):.2f} ms")
    print(f"   Max: {max(times):.2f} ms")
    
    # Test variants
    print("\n4. Model Variants:")
    
    lite = MouaadNetUltraLite()
    print(f"   LITE: {lite.count_parameters():,} params, {lite.get_model_size_mb('int8'):.2f} MB (INT8)")
    
    pro = MouaadNetUltraPro()
    print(f"   PRO: {pro.count_parameters():,} params, {pro.get_model_size_mb('int8'):.2f} MB (INT8)")
    
    print("\n" + "=" * 60)
    print("✓ All MOUAADNET-ULTRA tests passed!")
    print("=" * 60)
