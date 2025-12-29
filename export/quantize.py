"""
INT8 Quantization for MOUAADNET-ULTRA
======================================
Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
for extreme model compression.

Target: ~2-3MB model size with minimal accuracy loss.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import torch
import torch.nn as nn
from torch.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    prepare_qat,
    get_default_qat_qconfig,
)

sys.path.insert(0, str(Path(__file__).parent.parent))


def prepare_model_for_quantization(
    model: nn.Module,
    backend: str = 'qnnpack',
) -> nn.Module:
    """
    Prepare model for quantization.
    
    Args:
        model: Model to prepare
        backend: Quantization backend ('qnnpack' for mobile, 'fbgemm' for server)
        
    Returns:
        Prepared model
    """
    # Set backend
    torch.backends.quantized.engine = backend
    
    # Fuse modules for better quantization
    model = fuse_model_for_quantization(model)
    
    # Set qconfig
    model.qconfig = get_default_qconfig(backend)
    
    return model


def fuse_model_for_quantization(model: nn.Module) -> nn.Module:
    """
    Fuse Conv-BN-ReLU modules for quantization.
    
    Quantization works better with fused modules.
    """
    # Find fusable patterns
    modules_to_fuse = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            # Look for Conv-BN-ReLU patterns
            children = list(module.children())
            for i in range(len(children) - 2):
                if (isinstance(children[i], nn.Conv2d) and
                    isinstance(children[i+1], nn.BatchNorm2d) and
                    isinstance(children[i+2], (nn.ReLU, nn.ReLU6))):
                    modules_to_fuse.append([
                        f"{name}.{i}",
                        f"{name}.{i+1}",
                        f"{name}.{i+2}",
                    ])
    
    if modules_to_fuse:
        try:
            model = torch.quantization.fuse_modules(model, modules_to_fuse)
        except Exception as e:
            print(f"Warning: Module fusion failed: {e}")
    
    return model


def calibrate_model(
    model: nn.Module,
    calibration_dataloader,
    num_batches: int = 100,
) -> nn.Module:
    """
    Calibrate model for PTQ by collecting activation statistics.
    
    Args:
        model: Prepared model
        calibration_dataloader: DataLoader with representative data
        num_batches: Number of batches for calibration
        
    Returns:
        Calibrated model
    """
    model.eval()
    
    print(f"Calibrating with {num_batches} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(calibration_dataloader):
            if i >= num_batches:
                break
            
            if isinstance(batch, dict):
                images = batch['images']
            else:
                images = batch[0]
            
            model(images)
            
            if (i + 1) % 10 == 0:
                print(f"  Calibrated {i + 1}/{num_batches} batches")
    
    print("Calibration complete!")
    return model


def quantize_ptq(
    model: nn.Module,
    calibration_dataloader,
    backend: str = 'qnnpack',
    num_calibration_batches: int = 100,
) -> nn.Module:
    """
    Post-Training Quantization (PTQ).
    
    Quantizes weights to INT8 with minimal accuracy loss.
    
    Args:
        model: Trained FP32 model
        calibration_dataloader: DataLoader for calibration
        backend: Quantization backend
        num_calibration_batches: Number of batches for calibration
        
    Returns:
        Quantized INT8 model
    """
    print("Starting Post-Training Quantization...")
    print(f"Backend: {backend}")
    
    # Prepare
    model = prepare_model_for_quantization(model, backend)
    model = prepare(model)
    
    # Calibrate
    model = calibrate_model(model, calibration_dataloader, num_calibration_batches)
    
    # Convert to quantized
    model = convert(model)
    
    print("PTQ complete!")
    return model


def prepare_qat(
    model: nn.Module,
    backend: str = 'qnnpack',
) -> nn.Module:
    """
    Prepare model for Quantization-Aware Training (QAT).
    
    Inserts fake quantization nodes for training.
    
    Args:
        model: Model to prepare
        backend: Quantization backend
        
    Returns:
        Model ready for QAT
    """
    torch.backends.quantized.engine = backend
    
    # Use QAT-specific qconfig
    model.qconfig = get_default_qat_qconfig(backend)
    
    # Fuse modules
    model = fuse_model_for_quantization(model)
    
    # Prepare for QAT
    model = prepare_qat(model)
    
    return model


def finalize_qat(model: nn.Module) -> nn.Module:
    """
    Finalize QAT model by converting to actual INT8.
    
    Call this after QAT training is complete.
    """
    model.eval()
    model = convert(model)
    return model


def compare_model_sizes(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
):
    """Compare model sizes before and after quantization."""
    
    def get_size_mb(model):
        # Save to temp file to get actual size
        temp_path = "/tmp/temp_model.pt"
        torch.save(model.state_dict(), temp_path)
        size = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return size
    
    fp32_size = get_size_mb(fp32_model)
    quant_size = get_size_mb(quantized_model)
    
    print(f"\nModel Size Comparison:")
    print(f"  FP32:      {fp32_size:.2f} MB")
    print(f"  Quantized: {quant_size:.2f} MB")
    print(f"  Reduction: {(1 - quant_size/fp32_size) * 100:.1f}%")


def export_quantized_for_mobile(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, int] = (416, 416),
):
    """
    Export quantized model for mobile deployment.
    
    Uses TorchScript for mobile deployment.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Trace with example input
    example_input = torch.randn(1, 3, *input_size)
    
    # Script the model
    scripted = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    optimized = torch.utils.mobile_optimizer.optimize_for_mobile(scripted)
    
    # Save
    optimized._save_for_lite_interpreter(str(output_path))
    
    print(f"Exported to {output_path}")
    print(f"Size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    from mouaadnet_ultra.model import MouaadNetUltra
    from torch.utils.data import DataLoader, TensorDataset
    
    print("MOUAADNET-ULTRA Quantization")
    print("=" * 50)
    
    # Create model
    print("\n1. Creating model...")
    model = MouaadNetUltra()
    model.eval()
    
    print(f"   FP32 Parameters: {model.count_parameters():,}")
    print(f"   FP32 Size: {model.get_model_size_mb('fp32'):.2f} MB")
    print(f"   Target INT8 Size: {model.get_model_size_mb('int8'):.2f} MB")
    
    # Create dummy calibration data
    print("\n2. Creating calibration data...")
    dummy_data = torch.randn(100, 3, 416, 416)
    calibration_loader = DataLoader(
        TensorDataset(dummy_data),
        batch_size=8,
    )
    
    # Note: Full quantization requires model changes for compatibility
    # This is a demonstration of the quantization workflow
    
    print("\n3. Quantization workflow ready!")
    print("   - Use quantize_ptq() for post-training quantization")
    print("   - Use prepare_qat() + finalize_qat() for QAT")
    print("   - Use export_quantized_for_mobile() for mobile deployment")
    
    print("\n✓ Quantization module ready!")
