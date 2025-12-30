#!/usr/bin/env python3
"""
MOUAADNET-ULTRA: Real-Time Webcam Demo
========================================
Test the model with your computer camera using OpenCV.

Usage:
    python examples/webcam_demo.py
    python examples/webcam_demo.py --model lite
    python examples/webcam_demo.py --camera 1  # Use different camera
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mouaadnet_ultra.model import MouaadNetUltra, MouaadNetUltraLite


def preprocess(frame, input_size=416):
    """
    Preprocess frame for model input.
    
    Args:
        frame: BGR image from OpenCV
        input_size: Model input size
        
    Returns:
        Preprocessed tensor and scale info
    """
    h, w = frame.shape[:2]
    
    # Calculate scale to fit in input_size
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to input_size
    pad_h = input_size - new_h
    pad_w = input_size - new_w
    top = pad_h // 2
    left = pad_w // 2
    
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # Convert to tensor
    img = padded[:, :, ::-1]  # BGR to RGB
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = img / 255.0  # Normalize to [0, 1]
    
    tensor = torch.from_numpy(img).unsqueeze(0)
    
    return tensor, (scale, top, left, h, w)


def decode_heatmap(heatmap, threshold=0.3, top_k=20):
    """
    Decode detection peaks from heatmap.
    
    Args:
        heatmap: (1, 1, H, W) heatmap tensor
        threshold: Detection threshold
        top_k: Maximum detections
        
    Returns:
        List of (x, y, confidence) tuples
    """
    hm = heatmap[0, 0]  # (H, W)
    
    # Simple peak detection with max pooling
    hm_max = F.max_pool2d(
        hm.unsqueeze(0).unsqueeze(0), 
        kernel_size=3, stride=1, padding=1
    )[0, 0]
    
    # Keep only peaks
    peaks = (hm == hm_max) & (hm > threshold)
    
    # Get peak coordinates
    coords = peaks.nonzero()
    
    if len(coords) == 0:
        return []
    
    # Get scores
    scores = hm[coords[:, 0], coords[:, 1]]
    
    # Sort by score and take top_k
    order = scores.argsort(descending=True)[:top_k]
    
    detections = []
    for idx in order:
        y, x = coords[idx]
        conf = scores[idx].item()
        detections.append((x.item(), y.item(), conf))
    
    return detections


def draw_detections(frame, detections, sizes, scale_info, stride=4):
    """
    Draw detection boxes on frame.
    
    Args:
        frame: Original BGR frame
        detections: List of (x, y, conf) from heatmap
        sizes: Size predictions tensor
        scale_info: (scale, top, left, orig_h, orig_w)
        stride: Heatmap stride
    """
    scale, top, left, orig_h, orig_w = scale_info
    
    for x, y, conf in detections:
        # Get size at this location
        if sizes is not None:
            w_box = abs(sizes[0, 0, int(y), int(x)].item()) * stride * 4
            h_box = abs(sizes[0, 1, int(y), int(x)].item()) * stride * 4
        else:
            w_box = 80
            h_box = 160
        
        # Convert to original image coordinates
        cx = (x * stride - left) / scale
        cy = (y * stride - top) / scale
        
        x1 = int(cx - w_box / 2 / scale)
        y1 = int(cy - h_box / 2 / scale)
        x2 = int(cx + w_box / 2 / scale)
        y2 = int(cy + h_box / 2 / scale)
        
        # Clip to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(orig_w, x2)
        y2 = min(orig_h, y2)
        
        # Draw box
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"Person {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="MOUAADNET-ULTRA Webcam Demo")
    parser.add_argument("--model", type=str, default="ultra", 
                        choices=["ultra", "lite"], help="Model variant")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to trained weights (.pt file)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--size", type=int, default=416, help="Input size")
    parser.add_argument("--threshold", type=float, default=0.3, 
                        help="Detection threshold")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MOUAADNET-ULTRA: Real-Time Webcam Demo")
    print("=" * 60)
    
    # Device
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" 
        else "cpu"
    )
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading {args.model.upper()} model...")
    if args.model == "lite":
        model = MouaadNetUltraLite()
    else:
        model = MouaadNetUltra()
    
    # Load trained weights if provided
    if args.weights:
        print(f"Loading weights: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Epoch: {checkpoint.get('epoch', '?')}")
            print(f"  Best Acc: {checkpoint.get('best_acc', '?'):.2f}%" if 'best_acc' in checkpoint else "")
        else:
            model.load_state_dict(checkpoint)
        print("✓ Weights loaded!")
    else:
        print("⚠️ Using random weights (untrained)")
    
    model = model.to(device)
    model.eval()
    
    # Fuse for faster inference
    if hasattr(model, 'fuse_for_inference'):
        model.fuse_for_inference()
    
    print(f"Parameters: {model.count_parameters():,}")
    
    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return
    
    # Get camera info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    print("\nPress 'q' to quit, 's' to save screenshot")
    print("=" * 60)
    
    # FPS counter
    fps_counter = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame!")
                break
            
            start_time = time.time()
            
            # Preprocess
            tensor, scale_info = preprocess(frame, args.size)
            tensor = tensor.to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(tensor)
            
            # Get highest resolution heatmap
            heatmap = outputs['heatmaps'][0].cpu()
            sizes = outputs['sizes'][0].cpu()
            gender = torch.sigmoid(outputs['gender']).cpu().item()
            
            # Decode detections
            detections = decode_heatmap(heatmap, threshold=args.threshold)
            
            # Calculate FPS
            inference_time = time.time() - start_time
            fps_counter.append(1.0 / inference_time)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = sum(fps_counter) / len(fps_counter)
            
            # Draw detections
            frame = draw_detections(frame, detections, sizes, scale_info)
            
            # Draw info panel
            info_lines = [
                f"FPS: {avg_fps:.1f}",
                f"Detections: {len(detections)}",
                f"Model: {args.model.upper()}",
                f"Input: {args.size}x{args.size}",
            ]
            
            # Gender prediction (global)
            gender_text = f"Gender: {'Female' if gender > 0.5 else 'Male'} ({gender:.2f})"
            info_lines.append(gender_text)
            
            # Draw info
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 25 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow("MOUAADNET-ULTRA Demo", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo ended.")


if __name__ == "__main__":
    main()
