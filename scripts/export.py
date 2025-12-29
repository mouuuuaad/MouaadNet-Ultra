#!/usr/bin/env python3
"""
MOUAADNET-ULTRA: Export Script
===============================
Export trained models to production formats.

Usage:
    python scripts/export.py --weights checkpoints/best.pt --format onnx
    python scripts/export.py --format onnx --quantize int8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from mouaadnet_ultra.model import MouaadNetUltra, MouaadNetUltraLite


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export MOUAADNET-ULTRA to production formats"
    )
    
    parser.add_argument("--weights", type=str, help="Path to trained weights")
    parser.add_argument("--model", type=str, default="ultra",
                        choices=["ultra", "lite", "pro"], help="Model variant")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "torchscript", "tflite"],
                        help="Export format")
    parser.add_argument("--output", type=str, default="exports/model",
                        help="Output path (without extension)")
    parser.add_argument("--img-size", type=int, default=416, help="Input size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX")
    parser.add_argument("--quantize", type=str, choices=["none", "fp16", "int8"],
                        default="none", help="Quantization")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    
    return parser.parse_args()


def export_onnx(model, output_path, args):
    """Export to ONNX format."""
    import torch.onnx
    
    output_path = Path(output_path).with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dummy input
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size)
    
    # Dynamic axes
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
        }
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    
    print(f"Exported to: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Simplify
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
            
            model_onnx = onnx.load(str(output_path))
            model_simp, check = simplify(model_onnx)
            if check:
                onnx.save(model_simp, str(output_path))
                print("Graph simplified successfully")
        except ImportError:
            print("onnxsim not installed, skipping simplification")
    
    return output_path


def export_torchscript(model, output_path, args):
    """Export to TorchScript."""
    output_path = Path(output_path).with_suffix(".pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size)
    
    traced = torch.jit.trace(model, dummy_input)
    traced.save(str(output_path))
    
    print(f"Exported to: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return output_path


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MOUAADNET-ULTRA Export")
    print("=" * 60)
    
    # Create model
    model_classes = {
        "ultra": MouaadNetUltra,
        "lite": MouaadNetUltraLite,
    }
    
    model = model_classes.get(args.model, MouaadNetUltra)()
    
    # Load weights if provided
    if args.weights and Path(args.weights).exists():
        checkpoint = torch.load(args.weights, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded weights: {args.weights}")
    
    # Prepare model
    model.eval()
    
    # Fuse for inference
    if hasattr(model, "fuse_for_inference"):
        model.fuse_for_inference()
        print("Model fused for inference")
    
    # Export
    print(f"\nExport format: {args.format.upper()}")
    print(f"Input size: {args.img_size}x{args.img_size}")
    
    if args.format == "onnx":
        export_onnx(model, args.output, args)
    elif args.format == "torchscript":
        export_torchscript(model, args.output, args)
    else:
        print(f"Format {args.format} not yet implemented")
    
    print("\n✓ Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
