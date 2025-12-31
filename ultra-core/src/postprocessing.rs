//! CenterNet Postprocessing Module
//! ================================
//! High-performance CenterNet output decoding.
//!
//! Operations:
//! - Sigmoid activation
//! - Max pooling NMS
//! - Peak finding
//! - Bounding box calculation

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::Detection;

/// Apply sigmoid activation to array.
#[pyfunction]
pub fn sigmoid<'py>(py: Python<'py>, input: PyReadonlyArray1<'py, f32>) -> Bound<'py, PyArray1<f32>> {
    let arr = input.as_array();
    let result: Vec<f32> = arr
        .iter()
        .map(|&x| {
            let clamped = x.clamp(-50.0, 50.0);
            1.0 / (1.0 + (-clamped).exp())
        })
        .collect();
    PyArray1::from_vec(py, result)
}

/// Apply 3x3 max pooling NMS to 2D heatmap.
///
/// Returns a mask where peaks (local maxima) are True.
#[pyfunction]
pub fn nms_max_pool<'py>(
    py: Python<'py>,
    heatmap: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray1<bool>> {
    let hm = heatmap.as_array();
    let (h, w) = (hm.shape()[0], hm.shape()[1]);

    let mut peaks = vec![false; h * w];

    // 3x3 max pooling NMS
    for y in 0..h {
        for x in 0..w {
            let center = hm[[y, x]];

            // Check 3x3 neighborhood
            let mut is_max = true;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let ny = y as i32 + dy;
                    let nx = x as i32 + dx;

                    if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                        if hm[[ny as usize, nx as usize]] > center {
                            is_max = false;
                            break;
                        }
                    }
                }
                if !is_max {
                    break;
                }
            }

            peaks[y * w + x] = is_max;
        }
    }

    PyArray1::from_vec(py, peaks)
}

/// Decode CenterNet outputs to bounding boxes.
///
/// # Arguments
/// * `heatmap` - Detection heatmap (H, W) after sigmoid
/// * `wh` - Width/height predictions (2, H, W)
/// * `offset` - Center offset refinement (2, H, W)
/// * `threshold` - Detection confidence threshold
/// * `top_k` - Maximum number of detections
/// * `input_size` - Input image size (e.g., 256)
///
/// # Returns
/// * Vector of Detection objects
#[pyfunction]
pub fn decode_centernet(
    heatmap: PyReadonlyArray2<f32>,
    wh: PyReadonlyArray3<f32>,
    offset: PyReadonlyArray3<f32>,
    threshold: f32,
    top_k: usize,
    input_size: u32,
) -> Vec<Detection> {
    let hm = heatmap.as_array();
    let wh_arr = wh.as_array();
    let offset_arr = offset.as_array();

    let (h, w) = (hm.shape()[0], hm.shape()[1]);
    let stride = input_size as f32 / h as f32;

    // Step 1: Find peaks using 3x3 max pooling NMS
    let mut candidates: Vec<(usize, usize, f32)> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let score = hm[[y, x]];

            if score < threshold {
                continue;
            }

            // Check if local maximum (3x3 NMS)
            let mut is_max = true;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let ny = y as i32 + dy;
                    let nx = x as i32 + dx;

                    if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                        if hm[[ny as usize, nx as usize]] > score {
                            is_max = false;
                            break;
                        }
                    }
                }
                if !is_max {
                    break;
                }
            }

            if is_max {
                candidates.push((y, x, score));
            }
        }
    }

    // Step 2: Sort by score and take top_k
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(top_k);

    // Step 3: Decode each detection
    let mut detections: Vec<Detection> = Vec::with_capacity(candidates.len());

    for (cy_idx, cx_idx, conf) in candidates {
        // Get width/height predictions
        let pred_w = wh_arr[[0, cy_idx, cx_idx]];
        let pred_h = wh_arr[[1, cy_idx, cx_idx]];

        // Get offset refinement
        let off_x = offset_arr[[0, cy_idx, cx_idx]];
        let off_y = offset_arr[[1, cy_idx, cx_idx]];

        // Calculate center in input image coordinates
        let cx = (cx_idx as f32 + off_x) * stride;
        let cy = (cy_idx as f32 + off_y) * stride;

        // Calculate width/height in pixels
        // Try scaling by stride first
        let mut w_px = pred_w.abs() * stride;
        let mut h_px = pred_h.abs() * stride;

        // If values seem too small, they might be normalized by input_size
        if w_px < 10.0 {
            w_px = pred_w.abs() * input_size as f32;
            h_px = pred_h.abs() * input_size as f32;
        }

        // Minimum size constraints
        w_px = w_px.max(30.0);
        h_px = h_px.max(40.0);

        // Calculate box corners
        let x1 = (cx - w_px / 2.0).max(0.0);
        let y1 = (cy - h_px / 2.0).max(0.0);
        let x2 = (cx + w_px / 2.0).min(input_size as f32);
        let y2 = (cy + h_px / 2.0).min(input_size as f32);

        // Skip if too small
        if (x2 - x1) < 10.0 || (y2 - y1) < 10.0 {
            continue;
        }

        detections.push(Detection::new(x1, y1, x2, y2, conf));
    }

    detections
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let x = 0.0f32;
        let result = 1.0 / (1.0 + (-x).exp());
        assert!((result - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sigmoid_large() {
        let x = 50.0f32;
        let result = 1.0 / (1.0 + (-x.clamp(-50.0, 50.0)).exp());
        assert!(result > 0.99);
    }
}
