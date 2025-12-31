//! Image Preprocessing Module
//! ===========================
//! High-performance image preprocessing for MouaadNet-Ultra.
//!
//! Operations:
//! - Letterbox resize (aspect ratio preserving)
//! - ImageNet normalization (mean/std)
//! - NCHW tensor conversion

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::ScaleInfo;

/// ImageNet normalization constants
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Preprocess an RGB image for model input.
///
/// Performs:
/// 1. Letterbox resize to input_size (preserving aspect ratio)
/// 2. Padding with gray (114, 114, 114)
/// 3. Normalization to [0, 1]
/// 4. ImageNet mean/std normalization
/// 5. HWC to CHW transpose
///
/// # Arguments
/// * `image` - Input RGB image as numpy array (H, W, 3) uint8
/// * `input_size` - Target size (e.g., 256)
///
/// # Returns
/// * Tuple of (preprocessed tensor as flat Vec<f32> in CHW, ScaleInfo)
#[pyfunction]
pub fn preprocess_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'py, u8>,
    input_size: u32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, ScaleInfo)> {
    let img = image.as_array();
    let (height, width, _channels) = (img.shape()[0], img.shape()[1], img.shape()[2]);

    let h = height as u32;
    let w = width as u32;

    // Calculate scale to fit in input_size (letterbox)
    let scale = (input_size as f32 / h as f32).min(input_size as f32 / w as f32);
    let new_h = (h as f32 * scale) as u32;
    let new_w = (w as f32 * scale) as u32;

    // Calculate padding
    let pad_h = input_size - new_h;
    let pad_w = input_size - new_w;
    let pad_top = pad_h / 2;
    let pad_left = pad_w / 2;

    // Create output tensor (CHW format)
    let tensor_size = (3 * input_size * input_size) as usize;
    let mut output = vec![0.0f32; tensor_size];

    // Fill with normalized gray (114/255 normalized with ImageNet stats)
    let gray_normalized: [f32; 3] = [
        (114.0 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
        (114.0 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
        (114.0 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
    ];

    // Initialize with gray
    for c in 0..3 {
        let channel_offset = c * (input_size * input_size) as usize;
        for i in 0..(input_size * input_size) as usize {
            output[channel_offset + i] = gray_normalized[c];
        }
    }

    // Bilinear interpolation resize and place in padded region
    let input_size_usize = input_size as usize;
    let new_h_usize = new_h as usize;
    let new_w_usize = new_w as usize;

    // Process each output pixel in the resized region
    for out_y in 0..new_h_usize {
        for out_x in 0..new_w_usize {
            // Map to input coordinates
            let src_x = (out_x as f32 + 0.5) * (width as f32 / new_w as f32) - 0.5;
            let src_y = (out_y as f32 + 0.5) * (height as f32 / new_h as f32) - 0.5;

            // Bilinear interpolation coordinates
            let x0 = src_x.floor().max(0.0) as usize;
            let y0 = src_y.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);

            let x_frac = src_x - x0 as f32;
            let y_frac = src_y - y0 as f32;

            // Bilinear weights
            let w00 = (1.0 - x_frac) * (1.0 - y_frac);
            let w01 = x_frac * (1.0 - y_frac);
            let w10 = (1.0 - x_frac) * y_frac;
            let w11 = x_frac * y_frac;

            // Output position in padded image
            let dst_y = out_y + pad_top as usize;
            let dst_x = out_x + pad_left as usize;
            let dst_idx = dst_y * input_size_usize + dst_x;

            // Interpolate each channel
            for c in 0..3 {
                let v00 = img[[y0, x0, c]] as f32;
                let v01 = img[[y0, x1, c]] as f32;
                let v10 = img[[y1, x0, c]] as f32;
                let v11 = img[[y1, x1, c]] as f32;

                let value = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;

                // Normalize: [0, 255] -> [0, 1] -> ImageNet normalized
                let normalized = (value / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];

                // Write to CHW tensor
                let channel_offset = c * (input_size * input_size) as usize;
                output[channel_offset + dst_idx] = normalized;
            }
        }
    }

    let scale_info = ScaleInfo::new(scale, pad_top, pad_left, h, w, new_h, new_w);
    let output_array = PyArray1::from_vec(py, output);

    Ok((output_array, scale_info))
}

/// Preprocess a BGR image (OpenCV format) for model input.
///
/// Same as preprocess_image but converts BGR to RGB first.
#[pyfunction]
pub fn preprocess_bgr<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'py, u8>,
    input_size: u32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, ScaleInfo)> {
    let img = image.as_array();
    let (height, width, _channels) = (img.shape()[0], img.shape()[1], img.shape()[2]);

    let h = height as u32;
    let w = width as u32;

    // Calculate scale
    let scale = (input_size as f32 / h as f32).min(input_size as f32 / w as f32);
    let new_h = (h as f32 * scale) as u32;
    let new_w = (w as f32 * scale) as u32;

    let pad_h = input_size - new_h;
    let pad_w = input_size - new_w;
    let pad_top = pad_h / 2;
    let pad_left = pad_w / 2;

    let tensor_size = (3 * input_size * input_size) as usize;
    let mut output = vec![0.0f32; tensor_size];

    // Fill with normalized gray
    let gray_normalized: [f32; 3] = [
        (114.0 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
        (114.0 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
        (114.0 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
    ];

    for c in 0..3 {
        let channel_offset = c * (input_size * input_size) as usize;
        for i in 0..(input_size * input_size) as usize {
            output[channel_offset + i] = gray_normalized[c];
        }
    }

    let input_size_usize = input_size as usize;
    let new_h_usize = new_h as usize;
    let new_w_usize = new_w as usize;

    // BGR channel mapping to RGB
    let bgr_to_rgb = [2, 1, 0]; // BGR[0]=B -> RGB[2], BGR[1]=G -> RGB[1], BGR[2]=R -> RGB[0]

    for out_y in 0..new_h_usize {
        for out_x in 0..new_w_usize {
            let src_x = (out_x as f32 + 0.5) * (width as f32 / new_w as f32) - 0.5;
            let src_y = (out_y as f32 + 0.5) * (height as f32 / new_h as f32) - 0.5;

            let x0 = src_x.floor().max(0.0) as usize;
            let y0 = src_y.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);

            let x_frac = src_x - x0 as f32;
            let y_frac = src_y - y0 as f32;

            let w00 = (1.0 - x_frac) * (1.0 - y_frac);
            let w01 = x_frac * (1.0 - y_frac);
            let w10 = (1.0 - x_frac) * y_frac;
            let w11 = x_frac * y_frac;

            let dst_y = out_y + pad_top as usize;
            let dst_x = out_x + pad_left as usize;
            let dst_idx = dst_y * input_size_usize + dst_x;

            // Process BGR -> RGB
            for rgb_c in 0..3 {
                let bgr_c = bgr_to_rgb[rgb_c];
                
                let v00 = img[[y0, x0, bgr_c]] as f32;
                let v01 = img[[y0, x1, bgr_c]] as f32;
                let v10 = img[[y1, x0, bgr_c]] as f32;
                let v11 = img[[y1, x1, bgr_c]] as f32;

                let value = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                let normalized = (value / 255.0 - IMAGENET_MEAN[rgb_c]) / IMAGENET_STD[rgb_c];

                let channel_offset = rgb_c * (input_size * input_size) as usize;
                output[channel_offset + dst_idx] = normalized;
            }
        }
    }

    let scale_info = ScaleInfo::new(scale, pad_top, pad_left, h, w, new_h, new_w);
    let output_array = PyArray1::from_vec(py, output);

    Ok((output_array, scale_info))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_calculation() {
        // 640x480 image -> 256x256 input
        let h = 480u32;
        let w = 640u32;
        let input_size = 256u32;

        let scale = (input_size as f32 / h as f32).min(input_size as f32 / w as f32);
        assert!((scale - 0.4).abs() < 0.01); // 256/640 = 0.4
    }
}
