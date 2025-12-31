#!/usr/bin/env python3
"""
MOUAADNET-ULTRA V2: Webcam Demo
================================
Test MouaadNetUltraV2 model with webcam.

Usage:
    python examples/webcam_v2_demo.py
    python examples/webcam_v2_demo.py --weights training/mouaadnet_v2.pt --threshold 0.3
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


def load_v2_model(weights_path: str, device: torch.device):
    """Load MouaadNetUltraV2 from training script."""
    # Import model from training script
    sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
    from train_detection_v2 import MouaadNetUltraV2
    
    model = MouaadNetUltraV2()
    
    if weights_path and Path(weights_path).exists():
        print(f"Loading weights: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Epoch: {checkpoint.get('epoch', '?')}")
        else:
            model.load_state_dict(checkpoint)
        print("✓ Weights loaded!")
    else:
        print("⚠️ Using random weights (untrained)")
    
    return model


def preprocess(frame, input_size=256):
    """Preprocess frame with ImageNet normalization."""
    h, w = frame.shape[:2]
    
    # Scale to fit
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad
    pad_h = input_size - new_h
    pad_w = input_size - new_w
    top = pad_h // 2
    left = pad_w // 2
    
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    
    # RGB + ImageNet normalize
    img = canvas[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    
    return tensor, (scale, top, left, h, w)


def decode_v2(heatmap, wh, offset, threshold=0.3, top_k=5, input_size=256, stride=4):
    """Decode V2 model outputs to detections with proper NMS."""
    hm = torch.sigmoid(heatmap[0, 0])  # Apply sigmoid
    
    # NMS with max pooling (5x5 for better suppression)
    hm_max = F.max_pool2d(hm.unsqueeze(0).unsqueeze(0), 5, 1, 2)[0, 0]
    peaks = (hm == hm_max) & (hm > threshold)
    
    coords = peaks.nonzero()
    if len(coords) == 0:
        return []
    
    scores = hm[coords[:, 0], coords[:, 1]]
    order = scores.argsort(descending=True)[:top_k]
    
    detections = []
    for idx in order:
        cy, cx = coords[idx]
        conf = scores[idx].item()
        
        # Get WH (normalized 0-1)
        bw = abs(wh[0, 0, cy, cx].item()) * input_size
        bh = abs(wh[0, 1, cy, cx].item()) * input_size
        
        # Get offset
        ox = offset[0, 0, cy, cx].item()
        oy = offset[0, 1, cy, cx].item()
        
        # Center in input coords
        center_x = (cx.item() + ox) * stride
        center_y = (cy.item() + oy) * stride
        
        # Skip if WH too small (likely noise)
        if bw < 20 or bh < 30:
            bw = max(bw, 50)
            bh = max(bh, 80)
        
        detections.append({
            'cx': center_x,
            'cy': center_y,
            'w': bw,
            'h': bh,
            'conf': conf,
        })
    
    # Apply box-level NMS to remove overlapping boxes
    detections = nms_boxes(detections, iou_threshold=0.3)
    
    return detections


def nms_boxes(detections, iou_threshold=0.3):
    """Apply NMS to detection boxes."""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        # Remove overlapping boxes
        detections = [
            d for d in detections
            if box_iou(best, d) < iou_threshold
        ]
    
    return keep


def box_iou(a, b):
    """Calculate IoU between two detection dicts."""
    ax1, ay1 = a['cx'] - a['w']/2, a['cy'] - a['h']/2
    ax2, ay2 = a['cx'] + a['w']/2, a['cy'] + a['h']/2
    bx1, by1 = b['cx'] - b['w']/2, b['cy'] - b['h']/2
    bx2, by2 = b['cx'] + b['w']/2, b['cy'] + b['h']/2
    
    # Intersection
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    
    # Union
    area_a = a['w'] * a['h']
    area_b = b['w'] * b['h']
    union = area_a + area_b - inter
    
    return inter / union if union > 0 else 0


def draw_boxes(frame, detections, scale_info, input_size=256):
    """Draw detection boxes on frame."""
    scale, top, left, orig_h, orig_w = scale_info
    
    for det in detections:
        # Convert to original image coords
        cx = (det['cx'] - left) / scale
        cy = (det['cy'] - top) / scale
        w = det['w'] / scale
        h = det['h'] / scale
        
        x1 = int(max(0, cx - w / 2))
        y1 = int(max(0, cy - h / 2))
        x2 = int(min(orig_w, cx + w / 2))
        y2 = int(min(orig_h, cy + h / 2))
        
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        
        # Draw green box
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Corner accents
        corner = 15
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, 4)
        cv2.line(frame, (x2, y1), (x2 - corner, y1), color, 4)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), color, 4)
        cv2.line(frame, (x1, y2), (x1 + corner, y2), color, 4)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), color, 4)
        cv2.line(frame, (x2, y2), (x2 - corner, y2), color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), color, 4)
        
        # Label
        label = f"Person {det['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="MouaadNet-Ultra V2 Webcam Demo")
    parser.add_argument("--weights", type=str, default="training/mouaadnet_v2.pt",
                        help="Path to V2 weights")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--size", type=int, default=256, help="Input size")
    parser.add_argument("--threshold", type=float, default=0.3, 
                        help="Detection threshold")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 MOUAADNET-ULTRA V2: Webcam Demo")
    print("=" * 60)
    
    # Device
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" 
        else "cpu"
    )
    print(f"Device: {device}")
    
    # Load model
    model = load_v2_model(args.weights, device)
    model = model.to(device)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    # Camera
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {width}x{height}")
    
    print("\n" + "=" * 60)
    print("🎮 Controls:")
    print("   [Q] Quit")
    print("   [S] Screenshot")
    print("   [+/-] Adjust threshold")
    print("=" * 60)
    
    fps_counter = []
    threshold = args.threshold
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            
            # Preprocess
            tensor, scale_info = preprocess(frame, args.size)
            tensor = tensor.to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(tensor)
            
            # Decode
            detections = decode_v2(
                outputs['heatmap'].cpu(),
                outputs['wh'].cpu(),
                outputs['offset'].cpu(),
                threshold=threshold,
                input_size=args.size,
            )
            
            # FPS
            elapsed = time.time() - start
            fps_counter.append(1.0 / elapsed)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            fps = sum(fps_counter) / len(fps_counter)
            
            # Draw
            frame = draw_boxes(frame, detections, scale_info, args.size)
            
            # Info
            info = [
                f"FPS: {fps:.1f}",
                f"Detections: {len(detections)}",
                f"Threshold: {threshold:.2f}",
                f"Model: V2 ({params//1000}K)",
            ]
            for i, line in enumerate(info):
                cv2.putText(frame, line, (10, 25 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("MouaadNet-Ultra V2 Demo", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Saved: {filename}")
            elif key in (ord('+'), ord('=')):
                threshold = min(0.9, threshold + 0.05)
                print(f"🔼 Threshold: {threshold:.2f}")
            elif key == ord('-'):
                threshold = max(0.05, threshold - 0.05)
                print(f"🔽 Threshold: {threshold:.2f}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Demo ended.")


if __name__ == "__main__":
    main()
