"""
Rust Bindings for MouaadNet-Ultra
=================================
Python wrapper with automatic fallback to pure Python.

Usage:
    from mouaadnet_ultra.rust_bindings import preprocess_image, decode_centernet, VideoPipeline

If Rust bindings are not available, falls back to NumPy/SciPy implementations.
"""

import numpy as np
from typing import Tuple, List, Optional, NamedTuple

# Try to import Rust bindings
try:
    from ultra_core import (
        preprocess_image as _rust_preprocess_image,
        preprocess_bgr as _rust_preprocess_bgr,
        decode_centernet as _rust_decode_centernet,
        sigmoid as _rust_sigmoid,
        transform_detections as _rust_transform_detections,
        Detection as RustDetection,
        ScaleInfo as RustScaleInfo,
        VideoPipeline as RustVideoPipeline,
    )
    RUST_AVAILABLE = True
    print("✓ Ultra-Core Rust bindings loaded successfully!")
except ImportError:
    RUST_AVAILABLE = False
    print("⚠ Rust bindings not available, using pure Python fallback")


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Detection(NamedTuple):
    """Detection result."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def to_tuple(self) -> Tuple[float, float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2, self.confidence)

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class ScaleInfo(NamedTuple):
    """Preprocessing scale information."""
    scale: float
    pad_top: int
    pad_left: int
    orig_height: int
    orig_width: int
    new_height: int
    new_width: int


def _python_preprocess(
    image: np.ndarray,
    input_size: int = 256,
    is_bgr: bool = False,
) -> Tuple[np.ndarray, ScaleInfo]:
    """Pure Python preprocessing fallback."""
    import cv2

    h, w = image.shape[:2]

    # Calculate scale
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad
    pad_h = input_size - new_h
    pad_w = input_size - new_w
    top = pad_h // 2
    left = pad_w // 2

    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # Convert to float and normalize
    if is_bgr:
        img = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR to RGB
    else:
        img = padded.astype(np.float32) / 255.0

    # ImageNet normalization
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # HWC to CHW
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)

    scale_info = ScaleInfo(
        scale=scale,
        pad_top=top,
        pad_left=left,
        orig_height=h,
        orig_width=w,
        new_height=new_h,
        new_width=new_w,
    )

    return img.flatten(), scale_info


def _python_decode_centernet(
    heatmap: np.ndarray,
    wh: np.ndarray,
    offset: np.ndarray,
    threshold: float = 0.3,
    top_k: int = 10,
    input_size: int = 256,
) -> List[Detection]:
    """Pure Python CenterNet decoding fallback."""
    from scipy.ndimage import maximum_filter

    hm = heatmap  # (H, W)
    H, W = hm.shape
    stride = input_size // H

    # NMS using max pooling
    hm_max = maximum_filter(hm, size=3, mode='constant')
    peaks = (hm == hm_max) & (hm > threshold)

    # Get peak coordinates
    coords = np.argwhere(peaks)

    if len(coords) == 0:
        return []

    # Get scores
    scores = hm[coords[:, 0], coords[:, 1]]

    # Sort and take top_k
    order = np.argsort(scores)[::-1][:top_k]

    detections = []
    for idx in order:
        cy_idx, cx_idx = coords[idx]
        conf = float(scores[idx])

        # Get width/height
        w_pred = float(wh[0, cy_idx, cx_idx])
        h_pred = float(wh[1, cy_idx, cx_idx])

        # Get offset
        ox = float(offset[0, cy_idx, cx_idx])
        oy = float(offset[1, cy_idx, cx_idx])

        # Calculate center
        cx = (cx_idx + ox) * stride
        cy = (cy_idx + oy) * stride

        # Calculate size
        w_px = abs(w_pred) * stride
        h_px = abs(h_pred) * stride

        if w_px < 10:
            w_px = abs(w_pred) * input_size
            h_px = abs(h_pred) * input_size

        w_px = max(w_px, 30)
        h_px = max(h_px, 40)

        # Calculate box
        x1 = max(0, cx - w_px / 2)
        y1 = max(0, cy - h_px / 2)
        x2 = min(input_size, cx + w_px / 2)
        y2 = min(input_size, cy + h_px / 2)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        detections.append(Detection(x1, y1, x2, y2, conf))

    return detections


def _python_transform_detections(
    detections: List[Detection],
    scale_info: ScaleInfo,
) -> List[Detection]:
    """Pure Python coordinate transformation fallback."""
    transformed = []

    for det in detections:
        # Remove padding
        x1 = det.x1 - scale_info.pad_left
        y1 = det.y1 - scale_info.pad_top
        x2 = det.x2 - scale_info.pad_left
        y2 = det.y2 - scale_info.pad_top

        # Undo scaling
        x1 = x1 / scale_info.scale
        y1 = y1 / scale_info.scale
        x2 = x2 / scale_info.scale
        y2 = y2 / scale_info.scale

        # Clip to bounds
        x1 = max(0, min(scale_info.orig_width, x1))
        y1 = max(0, min(scale_info.orig_height, y1))
        x2 = max(0, min(scale_info.orig_width, x2))
        y2 = max(0, min(scale_info.orig_height, y2))

        if (x2 - x1) < 15 or (y2 - y1) < 15:
            continue

        transformed.append(Detection(x1, y1, x2, y2, det.confidence))

    return transformed


# ============================================================================
# Public API - Use Rust if available, otherwise Python fallback
# ============================================================================

def preprocess_image(
    image: np.ndarray,
    input_size: int = 256,
) -> Tuple[np.ndarray, ScaleInfo]:
    """
    Preprocess RGB image for model input.

    Args:
        image: RGB image (H, W, 3) uint8
        input_size: Target input size

    Returns:
        Tuple of (CHW tensor as flat array, ScaleInfo)
    """
    if RUST_AVAILABLE:
        tensor, scale_info = _rust_preprocess_image(image, input_size)
        return np.asarray(tensor), ScaleInfo(
            scale_info.scale,
            scale_info.pad_top,
            scale_info.pad_left,
            scale_info.orig_height,
            scale_info.orig_width,
            scale_info.new_height,
            scale_info.new_width,
        )
    else:
        return _python_preprocess(image, input_size, is_bgr=False)


def preprocess_bgr(
    image: np.ndarray,
    input_size: int = 256,
) -> Tuple[np.ndarray, ScaleInfo]:
    """
    Preprocess BGR image (OpenCV format) for model input.

    Args:
        image: BGR image (H, W, 3) uint8
        input_size: Target input size

    Returns:
        Tuple of (CHW tensor as flat array, ScaleInfo)
    """
    if RUST_AVAILABLE:
        tensor, scale_info = _rust_preprocess_bgr(image, input_size)
        return np.asarray(tensor), ScaleInfo(
            scale_info.scale,
            scale_info.pad_top,
            scale_info.pad_left,
            scale_info.orig_height,
            scale_info.orig_width,
            scale_info.new_height,
            scale_info.new_width,
        )
    else:
        return _python_preprocess(image, input_size, is_bgr=True)


def decode_centernet(
    heatmap: np.ndarray,
    wh: np.ndarray,
    offset: np.ndarray,
    threshold: float = 0.3,
    top_k: int = 10,
    input_size: int = 256,
) -> List[Detection]:
    """
    Decode CenterNet outputs to detections.

    Args:
        heatmap: Detection heatmap (H, W) after sigmoid
        wh: Width/height predictions (2, H, W)
        offset: Offset predictions (2, H, W)
        threshold: Detection threshold
        top_k: Maximum detections
        input_size: Input image size

    Returns:
        List of Detection objects
    """
    if RUST_AVAILABLE:
        rust_dets = _rust_decode_centernet(heatmap, wh, offset, threshold, top_k, input_size)
        return [Detection(d.x1, d.y1, d.x2, d.y2, d.confidence) for d in rust_dets]
    else:
        return _python_decode_centernet(heatmap, wh, offset, threshold, top_k, input_size)


def transform_detections(
    detections: List[Detection],
    scale_info: ScaleInfo,
) -> List[Detection]:
    """
    Transform detections from input coords to original frame coords.

    Args:
        detections: Detections in input coordinates
        scale_info: Scale info from preprocessing

    Returns:
        Detections in original frame coordinates
    """
    if RUST_AVAILABLE:
        # Convert to Rust Detection objects
        rust_dets = [
            RustDetection(d.x1, d.y1, d.x2, d.y2, d.confidence)
            for d in detections
        ]
        rust_scale = RustScaleInfo(
            scale_info.scale,
            scale_info.pad_top,
            scale_info.pad_left,
            scale_info.orig_height,
            scale_info.orig_width,
            scale_info.new_height,
            scale_info.new_width,
        )
        result = _rust_transform_detections(rust_dets, rust_scale)
        return [Detection(d.x1, d.y1, d.x2, d.y2, d.confidence) for d in result]
    else:
        return _python_transform_detections(detections, scale_info)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation."""
    if RUST_AVAILABLE:
        return np.asarray(_rust_sigmoid(x.flatten())).reshape(x.shape)
    else:
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


class VideoPipeline:
    """
    Real-time video processing pipeline.

    Wraps Rust VideoPipeline if available, otherwise uses Python.
    """

    def __init__(self, input_size: int = 256, threshold: float = 0.3, top_k: int = 10):
        self.input_size = input_size
        self._threshold = threshold
        self.top_k = top_k

        if RUST_AVAILABLE:
            self._rust_pipeline = RustVideoPipeline(input_size, threshold, top_k)
        else:
            self._rust_pipeline = None

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = max(0.01, min(0.99, value))
        if self._rust_pipeline:
            self._rust_pipeline.threshold = self._threshold

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, ScaleInfo]:
        """Preprocess BGR frame."""
        return preprocess_bgr(frame, self.input_size)

    def decode(
        self,
        heatmap: np.ndarray,
        wh: np.ndarray,
        offset: np.ndarray,
    ) -> List[Detection]:
        """Decode model outputs."""
        return decode_centernet(heatmap, wh, offset, self.threshold, self.top_k, self.input_size)

    def transform(self, detections: List[Detection], scale_info: ScaleInfo) -> List[Detection]:
        """Transform to original coordinates."""
        return transform_detections(detections, scale_info)

    def postprocess(
        self,
        heatmap: np.ndarray,
        wh: np.ndarray,
        offset: np.ndarray,
        scale_info: ScaleInfo,
    ) -> List[Detection]:
        """Full postprocessing: decode + transform."""
        detections = self.decode(heatmap, wh, offset)
        return self.transform(detections, scale_info)

    def __repr__(self) -> str:
        backend = "Rust" if RUST_AVAILABLE else "Python"
        return f"VideoPipeline(input_size={self.input_size}, threshold={self.threshold:.2f}, backend={backend})"


# Export symbols
__all__ = [
    "RUST_AVAILABLE",
    "Detection",
    "ScaleInfo",
    "preprocess_image",
    "preprocess_bgr",
    "decode_centernet",
    "transform_detections",
    "sigmoid",
    "VideoPipeline",
]
