#!/usr/bin/env python3
"""
MOUAADNET-ULTRA: ONNX Webcam Demo (CenterNet)
=============================================
Test the ONNX detection model with your webcam using ONNX Runtime.

Model outputs:
    - heatmap: (batch, 1, H, W) - detection confidence
    - wh: (batch, 2, H, W) - width/height predictions
    - offset: (batch, 2, H, W) - center offset refinement

Usage:
    python examples/webcam_onnx_demo.py
    python examples/webcam_onnx_demo.py --model detection.onnx --threshold 0.3
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
from pathlib import Path


def preprocess(frame, input_size=256):
    """
    Preprocess frame for ONNX model input.
    
    Args:
        frame: BGR image from OpenCV
        input_size: Model input size (256 for this model)
        
    Returns:
        Preprocessed numpy array and scale info
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
    
    # Convert to NCHW float32 input with ImageNet normalization
    img = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR to RGB, normalize to [0,1]
    
    # ImageNet normalization (REQUIRED for this model!)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    return img, (scale, top, left, h, w, new_h, new_w)


def sigmoid(x):
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def decode_centernet(heatmap, wh, offset, threshold=0.1, top_k=1, input_size=256):
    """
    Decode CenterNet detection outputs.
    
    Args:
        heatmap: (1, 1, H, W) - detection confidence after sigmoid
        wh: (1, 2, H, W) - width/height predictions
        offset: (1, 2, H, W) - center offset refinement
        threshold: Detection threshold
        top_k: Maximum detections (default: 1 = only best detection)
        input_size: Original input image size
        
    Returns:
        List of (x1, y1, x2, y2, confidence) in input image coordinates
    """
    hm = heatmap[0, 0]  # (H, W)
    H, W = hm.shape
    stride = input_size // H  # Should be 4 (256/64)
    
    # Simple NMS using max pooling
    from scipy.ndimage import maximum_filter
    hm_max = maximum_filter(hm, size=3, mode='constant')
    
    # Keep only peaks above threshold
    peaks = (hm == hm_max) & (hm > threshold)
    
    # Get peak coordinates
    coords = np.argwhere(peaks)
    
    if len(coords) == 0:
        return []
    
    # Get scores at peak locations
    scores = hm[coords[:, 0], coords[:, 1]]
    
    # Sort by score and take top_k
    order = np.argsort(scores)[::-1][:top_k]
    
    detections = []
    for idx in order:
        cy_idx, cx_idx = coords[idx]  # y, x in heatmap
        conf = float(scores[idx])
        
        # Get width/height at this location
        w = float(wh[0, 0, cy_idx, cx_idx])
        h = float(wh[0, 1, cy_idx, cx_idx])
        
        # Get offset refinement
        ox = float(offset[0, 0, cy_idx, cx_idx])
        oy = float(offset[0, 1, cy_idx, cx_idx])
        
        # Convert to input image coordinates
        # Center = (heatmap_coord + offset) * stride
        cx = (cx_idx + ox) * stride
        cy = (cy_idx + oy) * stride
        
        # Width/height - these might be in different formats depending on training
        # Try interpreting as direct pixel values first
        w_px = abs(w) * stride  # Scale by stride
        h_px = abs(h) * stride
        
        # If values seem too small, they might be normalized
        if w_px < 10:
            w_px = abs(w) * input_size
            h_px = abs(h) * input_size
        
        # Minimum size constraints
        w_px = max(w_px, 30)
        h_px = max(h_px, 40)
        
        # Calculate box corners
        x1 = cx - w_px / 2
        y1 = cy - h_px / 2
        x2 = cx + w_px / 2
        y2 = cy + h_px / 2
        
        # Clip to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(input_size, x2)
        y2 = min(input_size, y2)
        
        # Skip if too small
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue
            
        detections.append((x1, y1, x2, y2, conf))
    
    return detections


def draw_detections(frame, detections, scale_info, input_size=256):
    """
    Draw detection boxes on frame.
    
    Args:
        frame: Original BGR frame
        detections: List of (x1, y1, x2, y2, conf) in input coords
        scale_info: (scale, top, left, orig_h, orig_w, new_h, new_w)
        input_size: Model input size
    """
    scale, top, left, orig_h, orig_w, new_h, new_w = scale_info
    
    for x1, y1, x2, y2, conf in detections:
        # Convert from input (padded) coords to original frame coords
        # First remove padding offset
        x1_unpad = x1 - left
        y1_unpad = y1 - top
        x2_unpad = x2 - left
        y2_unpad = y2 - top
        
        # Then undo scaling
        x1_orig = int(x1_unpad / scale)
        y1_orig = int(y1_unpad / scale)
        x2_orig = int(x2_unpad / scale)
        y2_orig = int(y2_unpad / scale)
        
        # Clip to frame bounds
        x1_orig = max(0, x1_orig)
        y1_orig = max(0, y1_orig)
        x2_orig = min(orig_w, x2_orig)
        y2_orig = min(orig_h, y2_orig)
        
        # Skip if box is too small or invalid
        if x2_orig - x1_orig < 15 or y2_orig - y1_orig < 15:
            continue
        
        # Draw thick green box
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 3)
        
        # Draw corner accents for better visibility
        corner_len = min(15, (x2_orig - x1_orig) // 3, (y2_orig - y1_orig) // 3)
        cv2.line(frame, (x1_orig, y1_orig), (x1_orig + corner_len, y1_orig), color, 4)
        cv2.line(frame, (x1_orig, y1_orig), (x1_orig, y1_orig + corner_len), color, 4)
        cv2.line(frame, (x2_orig, y1_orig), (x2_orig - corner_len, y1_orig), color, 4)
        cv2.line(frame, (x2_orig, y1_orig), (x2_orig, y1_orig + corner_len), color, 4)
        cv2.line(frame, (x1_orig, y2_orig), (x1_orig + corner_len, y2_orig), color, 4)
        cv2.line(frame, (x1_orig, y2_orig), (x1_orig, y2_orig - corner_len), color, 4)
        cv2.line(frame, (x2_orig, y2_orig), (x2_orig - corner_len, y2_orig), color, 4)
        cv2.line(frame, (x2_orig, y2_orig), (x2_orig, y2_orig - corner_len), color, 4)
        
        # Draw label with background
        label = f"Person {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1_orig, y1_orig - th - 8), (x1_orig + tw + 4, y1_orig), color, -1)
        cv2.putText(frame, label, (x1_orig + 2, y1_orig - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="MOUAADNET-ULTRA ONNX Webcam Demo")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to ONNX model (default: detection.onnx in project root)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--size", type=int, default=256, help="Input size (256 for this model)")
    parser.add_argument("--threshold", type=float, default=0.1, 
                        help="Detection threshold (default: 0.1)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 MOUAADNET-ULTRA: ONNX Webcam Demo")
    print("=" * 60)
    
    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        # Default: look for detection.onnx in project root
        project_root = Path(__file__).parent.parent
        model_path = project_root / "detection.onnx"
        
        if not model_path.exists():
            model_path = project_root / "mouaadnet_ultra.onnx"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run: python examples/webcam_onnx_demo.py --model /path/to/your.onnx")
        return
    
    print(f"📦 Model: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Load ONNX model
    print("\n🔧 Loading ONNX Runtime session...")
    
    # Choose providers (prefer CUDA if available)
    providers = []
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
        print("   ✓ CUDA available - using GPU acceleration!")
    providers.append('CPUExecutionProvider')
    
    session = ort.InferenceSession(str(model_path), providers=providers)
    
    # Print model info
    print(f"\n📊 Model Info:")
    print(f"   Inputs:")
    for inp in session.get_inputs():
        print(f"      - {inp.name}: {inp.shape} ({inp.type})")
    print(f"   Outputs:")
    for out in session.get_outputs():
        print(f"      - {out.name}: {out.shape}")
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    
    # Open camera
    print(f"\n📹 Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        return
    
    # Get camera info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Resolution: {width}x{height}")
    
    print("\n" + "=" * 60)
    print("🎮 Controls:")
    print("   [Q] Quit")
    print("   [S] Save screenshot")
    print("   [+] Increase threshold")
    print("   [-] Decrease threshold")
    print("=" * 60)
    
    # FPS counter
    fps_counter = []
    threshold = args.threshold
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame!")
                break
            
            start_time = time.time()
            
            # Preprocess
            input_tensor, scale_info = preprocess(frame, args.size)
            
            # Inference
            outputs = session.run(output_names, {input_name: input_tensor})
            
            # Parse CenterNet outputs by name
            heatmap = None
            wh = None
            offset = None
            
            for name, out in zip(output_names, outputs):
                if name == 'heatmap':
                    heatmap = sigmoid(out)
                elif name == 'wh':
                    wh = out
                elif name == 'offset':
                    offset = out
            
            # Decode detections
            detections = []
            if heatmap is not None and wh is not None and offset is not None:
                detections = decode_centernet(heatmap, wh, offset, 
                                               threshold=threshold, 
                                               input_size=args.size)
            
            # Calculate FPS
            inference_time = time.time() - start_time
            fps_counter.append(1.0 / inference_time)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = sum(fps_counter) / len(fps_counter)
            
            # Draw detections
            frame = draw_detections(frame, detections, scale_info, input_size=args.size)
            
            # Draw info panel
            info_lines = [
                f"FPS: {avg_fps:.1f}",
                f"Detections: {len(detections)}",
                f"Threshold: {threshold:.2f}",
                f"Input: {args.size}x{args.size}",
            ]
            
            # Draw info
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 25 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show ONNX badge
            cv2.putText(frame, "ONNX Runtime", (width - 150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            # Show frame
            cv2.imshow("MOUAADNET-ULTRA ONNX Demo", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Saved: {filename}")
            elif key == ord('+') or key == ord('='):
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
