//! Coordinate Transformation Module
//! ==================================
//! Transform detection coordinates between different spaces.
//!
//! Operations:
//! - Input coords -> Original frame coords
//! - Clipping to bounds
//! - Batch transformations

use pyo3::prelude::*;

use crate::{Detection, ScaleInfo};

/// Transform detections from input (padded) coordinates to original frame coordinates.
///
/// # Arguments
/// * `detections` - Vector of detections in input image coordinates
/// * `scale_info` - Scale and padding information from preprocessing
///
/// # Returns
/// * Vector of detections in original frame coordinates
#[pyfunction]
pub fn transform_detections(detections: Vec<Detection>, scale_info: &ScaleInfo) -> Vec<Detection> {
    let mut transformed: Vec<Detection> = Vec::with_capacity(detections.len());

    for det in detections {
        // Remove padding offset
        let x1_unpad = det.x1 - scale_info.pad_left as f32;
        let y1_unpad = det.y1 - scale_info.pad_top as f32;
        let x2_unpad = det.x2 - scale_info.pad_left as f32;
        let y2_unpad = det.y2 - scale_info.pad_top as f32;

        // Undo scaling
        let x1_orig = x1_unpad / scale_info.scale;
        let y1_orig = y1_unpad / scale_info.scale;
        let x2_orig = x2_unpad / scale_info.scale;
        let y2_orig = y2_unpad / scale_info.scale;

        // Clip to original frame bounds
        let x1_clipped = x1_orig.max(0.0).min(scale_info.orig_width as f32);
        let y1_clipped = y1_orig.max(0.0).min(scale_info.orig_height as f32);
        let x2_clipped = x2_orig.max(0.0).min(scale_info.orig_width as f32);
        let y2_clipped = y2_orig.max(0.0).min(scale_info.orig_height as f32);

        // Skip if box is too small after transformation
        if (x2_clipped - x1_clipped) < 15.0 || (y2_clipped - y1_clipped) < 15.0 {
            continue;
        }

        transformed.push(Detection::new(
            x1_clipped,
            y1_clipped,
            x2_clipped,
            y2_clipped,
            det.confidence,
        ));
    }

    transformed
}

/// Clip detection coordinates to specified bounds.
///
/// # Arguments
/// * `detection` - Detection to clip
/// * `max_width` - Maximum x coordinate
/// * `max_height` - Maximum y coordinate
///
/// # Returns
/// * Clipped detection (or None if too small)
#[pyfunction]
pub fn clip_to_bounds(
    detection: &Detection,
    max_width: f32,
    max_height: f32,
) -> Option<Detection> {
    let x1 = detection.x1.max(0.0).min(max_width);
    let y1 = detection.y1.max(0.0).min(max_height);
    let x2 = detection.x2.max(0.0).min(max_width);
    let y2 = detection.y2.max(0.0).min(max_height);

    // Check minimum size
    if (x2 - x1) < 10.0 || (y2 - y1) < 10.0 {
        return None;
    }

    Some(Detection::new(x1, y1, x2, y2, detection.confidence))
}

/// Calculate IoU (Intersection over Union) between two detections.
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = a.area();
    let area_b = b.area();
    let union_area = area_a + area_b - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}

/// Apply NMS (Non-Maximum Suppression) to detections.
pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }

    // Sort by confidence (descending)
    let mut sorted: Vec<&Detection> = detections.iter().collect();
    sorted.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep: Vec<Detection> = Vec::new();
    let mut suppressed = vec![false; sorted.len()];

    for i in 0..sorted.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(sorted[i].clone());

        for j in (i + 1)..sorted.len() {
            if !suppressed[j] && iou(sorted[i], sorted[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_same_box() {
        let det = Detection::new(0.0, 0.0, 100.0, 100.0, 0.9);
        let result = iou(&det, &det);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = Detection::new(0.0, 0.0, 50.0, 50.0, 0.9);
        let b = Detection::new(100.0, 100.0, 150.0, 150.0, 0.8);
        let result = iou(&a, &b);
        assert!(result < 0.001);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let a = Detection::new(0.0, 0.0, 100.0, 100.0, 0.9);
        let b = Detection::new(50.0, 50.0, 150.0, 150.0, 0.8);
        // Intersection: 50x50 = 2500
        // Union: 10000 + 10000 - 2500 = 17500
        // IoU: 2500 / 17500 ≈ 0.143
        let result = iou(&a, &b);
        assert!((result - 0.143).abs() < 0.01);
    }
}
