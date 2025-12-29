#!/usr/bin/env python3
"""
Example: Model Benchmarking
============================
Benchmark MOUAADNET-ULTRA inference speed on different configurations.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from mouaadnet_ultra.model import MouaadNetUltra, MouaadNetUltraLite


def benchmark_model(model, input_size, device, num_warmup=10, num_runs=100):
    """Benchmark model inference speed."""
    model = model.to(device)
    model.eval()
    
    x = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    # Sync before timing (for CUDA)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    
    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
        'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def main():
    print("=" * 70)
    print("MOUAADNET-ULTRA: Benchmark Suite")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Models to benchmark
    models = {
        'Ultra': MouaadNetUltra(),
        'Lite': MouaadNetUltraLite(),
    }
    
    # Input sizes
    input_sizes = [320, 416, 640]
    
    # Results table
    print("\n" + "=" * 70)
    print(f"{'Model':<10} | {'Size':<6} | {'Mean (ms)':<12} | {'Min (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 70)
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{name} ({params:,} params)")
        
        for size in input_sizes:
            try:
                results = benchmark_model(model, size, device, num_runs=50)
                print(f"  {size}x{size:<4} | "
                      f"Mean: {results['mean']:>6.2f}ms | "
                      f"Min: {results['min']:>6.2f}ms | "
                      f"Max: {results['max']:>6.2f}ms")
            except Exception as e:
                print(f"  {size}x{size:<4} | Error: {e}")
    
    print("\n" + "=" * 70)
    print("✓ Benchmark complete!")


if __name__ == "__main__":
    main()
