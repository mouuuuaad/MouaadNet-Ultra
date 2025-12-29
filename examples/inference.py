#!/usr/bin/env python3
"""
Example: Basic Inference with MOUAADNET-ULTRA
==============================================
Demonstrates how to load the model and run inference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from mouaadnet_ultra.model import MouaadNetUltra


def main():
    print("=" * 60)
    print("MOUAADNET-ULTRA: Basic Inference Example")
    print("=" * 60)
    
    # 1. Create model
    print("\n1. Creating model...")
    model = MouaadNetUltra()
    model.eval()
    
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Size: {model.get_model_size_mb():.2f} MB")
    
    # 2. Prepare input
    print("\n2. Preparing input...")
    # Random image tensor (replace with actual image preprocessing)
    image = torch.randn(1, 3, 416, 416)
    print(f"   Input shape: {image.shape}")
    
    # 3. Run inference
    print("\n3. Running inference...")
    with torch.no_grad():
        outputs = model(image)
    
    # 4. Parse outputs
    print("\n4. Parsing outputs...")
    
    # Detection outputs (multi-scale)
    heatmaps = outputs['heatmaps']
    sizes = outputs['sizes']
    offsets = outputs['offsets']
    
    print(f"   Heatmaps: {len(heatmaps)} scales")
    for i, hm in enumerate(heatmaps):
        print(f"     Scale {i}: {hm.shape}, max={hm.max():.4f}")
    
    # Gender prediction
    gender_logits = outputs['gender']
    gender_prob = torch.sigmoid(gender_logits)
    gender = "Female" if gender_prob > 0.5 else "Male"
    
    print(f"\n   Gender prediction:")
    print(f"     Logits: {gender_logits.item():.4f}")
    print(f"     Probability: {gender_prob.item():.4f}")
    print(f"     Prediction: {gender}")
    
    # 5. Decode detections (simplified example)
    print("\n5. Decoding detections...")
    
    # Get peak from highest resolution heatmap
    hm = heatmaps[0][0, 0]  # First batch, first channel
    peak_val, peak_idx = hm.view(-1).max(0)
    peak_y = peak_idx // hm.shape[1]
    peak_x = peak_idx % hm.shape[1]
    
    print(f"   Peak location: ({peak_x}, {peak_y})")
    print(f"   Peak confidence: {peak_val:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
