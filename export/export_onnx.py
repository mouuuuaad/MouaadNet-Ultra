"""
ONNX Export and Optimization
==============================
Export MOUAADNET-ULTRA to ONNX format with layer fusion
and graph optimization for production deployment.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mouaadnet_ultra.model import MouaadNetUltra


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, int] = (416, 416),
    batch_size: int = 1,
    opset_version: int = 12,
    dynamic_axes: bool = True,
    simplify: bool = True,
) -> str:
    """
    Export MOUAADNET-ULTRA to ONNX format.
    
    Args:
        model: Trained model
        output_path: Path to save ONNX file
        input_size: Input image size (H, W)
        batch_size: Batch size for export
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic input shapes
        simplify: Whether to simplify the graph
        
    Returns:
        Path to exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare model
    model.eval()
    
    # Fuse RepVGG blocks for better performance
    if hasattr(model, 'fuse_for_inference'):
        model.fuse_for_inference()
    
    # Create dummy input
    h, w = input_size
    dummy_input = torch.randn(batch_size, 3, h, w)
    
    # Dynamic axes configuration
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'heatmap_0': {0: 'batch', 2: 'h0', 3: 'w0'},
            'heatmap_1': {0: 'batch', 2: 'h1', 3: 'w1'},
            'heatmap_2': {0: 'batch', 2: 'h2', 3: 'w2'},
            'size_0': {0: 'batch', 2: 'h0', 3: 'w0'},
            'size_1': {0: 'batch', 2: 'h1', 3: 'w1'},
            'size_2': {0: 'batch', 2: 'h2', 3: 'w2'},
            'offset_0': {0: 'batch', 2: 'h0', 3: 'w0'},
            'offset_1': {0: 'batch', 2: 'h1', 3: 'w1'},
            'offset_2': {0: 'batch', 2: 'h2', 3: 'w2'},
            'gender': {0: 'batch'},
        }
    else:
        dynamic_axes_dict = None
    
    # Create wrapper for multiple outputs
    class ONNXWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            outputs = self.model(x)
            # Flatten output dict to tuple
            return (
                outputs['heatmaps'][0],
                outputs['heatmaps'][1],
                outputs['heatmaps'][2],
                outputs['sizes'][0],
                outputs['sizes'][1],
                outputs['sizes'][2],
                outputs['offsets'][0],
                outputs['offsets'][1],
                outputs['offsets'][2],
                outputs['gender'],
            )
    
    wrapped_model = ONNXWrapper(model)
    
    # Export
    output_names = [
        'heatmap_0', 'heatmap_1', 'heatmap_2',
        'size_0', 'size_1', 'size_2',
        'offset_0', 'offset_1', 'offset_2',
        'gender',
    ]
    
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"Exported to {output_path}")
    
    # Simplify graph (optional)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            onnx_model = onnx.load(str(output_path))
            onnx_model_simp, check = onnx_simplify(onnx_model)
            
            if check:
                onnx.save(onnx_model_simp, str(output_path))
                print("Graph simplified successfully")
            else:
                print("Simplification check failed, keeping original")
        except ImportError:
            print("onnxsim not installed, skipping simplification")
    
    # Print model info
    print_onnx_info(str(output_path))
    
    return str(output_path)


def print_onnx_info(onnx_path: str):
    """Print ONNX model information."""
    try:
        import onnx
        
        model = onnx.load(onnx_path)
        
        # Get file size
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        
        print(f"\nONNX Model Info:")
        print(f"  File: {onnx_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Opset: {model.opset_import[0].version}")
        print(f"  Inputs: {[i.name for i in model.graph.input]}")
        print(f"  Outputs: {[o.name for o in model.graph.output]}")
        
    except ImportError:
        print("onnx not installed, skipping info print")


def verify_onnx(
    onnx_path: str,
    pytorch_model: nn.Module,
    input_size: Tuple[int, int] = (416, 416),
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Verify ONNX model matches PyTorch output.
    
    Args:
        onnx_path: Path to ONNX file
        pytorch_model: Original PyTorch model
        input_size: Input size to test
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if outputs match
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("onnxruntime not installed, skipping verification")
        return True
    
    # Create test input
    h, w = input_size
    test_input = torch.randn(1, 3, h, w)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_outputs = pytorch_model(test_input)
    
    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path)
    ort_inputs = {'input': test_input.numpy()}
    ort_outputs = session.run(None, ort_inputs)
    
    # Compare outputs
    pytorch_flat = [
        pytorch_outputs['heatmaps'][0].numpy(),
        pytorch_outputs['heatmaps'][1].numpy(),
        pytorch_outputs['heatmaps'][2].numpy(),
        pytorch_outputs['sizes'][0].numpy(),
        pytorch_outputs['sizes'][1].numpy(),
        pytorch_outputs['sizes'][2].numpy(),
        pytorch_outputs['offsets'][0].numpy(),
        pytorch_outputs['offsets'][1].numpy(),
        pytorch_outputs['offsets'][2].numpy(),
        pytorch_outputs['gender'].numpy(),
    ]
    
    all_close = True
    for i, (pt, ort) in enumerate(zip(pytorch_flat, ort_outputs)):
        if not np.allclose(pt, ort, rtol=rtol, atol=atol):
            max_diff = np.abs(pt - ort).max()
            print(f"Output {i} mismatch! Max diff: {max_diff}")
            all_close = False
    
    if all_close:
        print("✓ ONNX verification passed!")
    
    return all_close


def benchmark_onnx(
    onnx_path: str,
    input_size: Tuple[int, int] = (416, 416),
    num_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """
    Benchmark ONNX model inference speed.
    
    Args:
        onnx_path: Path to ONNX file
        input_size: Input size
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs
        
    Returns:
        Dict with timing statistics
    """
    try:
        import onnxruntime as ort
        import numpy as np
        import time
    except ImportError:
        print("onnxruntime not installed")
        return {}
    
    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    used_provider = session.get_providers()[0]
    print(f"Using provider: {used_provider}")
    
    # Create input
    h, w = input_size
    test_input = np.random.randn(1, 3, h, w).astype(np.float32)
    ort_inputs = {'input': test_input}
    
    # Warmup
    for _ in range(warmup):
        _ = session.run(None, ort_inputs)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, ort_inputs)
        times.append((time.perf_counter() - start) * 1000)
    
    results = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
    }
    
    print(f"\nONNX Benchmark Results ({num_runs} runs):")
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std:  {results['std_ms']:.2f} ms")
    print(f"  Min:  {results['min_ms']:.2f} ms")
    print(f"  Max:  {results['max_ms']:.2f} ms")
    print(f"  P50:  {results['p50_ms']:.2f} ms")
    print(f"  P95:  {results['p95_ms']:.2f} ms")
    
    return results


if __name__ == "__main__":
    print("MOUAADNET-ULTRA ONNX Export")
    print("=" * 50)
    
    # Create and prepare model
    model = MouaadNetUltra()
    model.eval()
    
    # Export
    output_path = "./exports/mouaadnet_ultra.onnx"
    export_onnx(
        model,
        output_path,
        input_size=(416, 416),
        dynamic_axes=True,
        simplify=False,  # Set True if onnxsim installed
    )
    
    # Verify
    verify_onnx(output_path, model)
    
    # Benchmark
    benchmark_onnx(output_path, num_runs=50)
    
    print("\n✓ ONNX export complete!")
