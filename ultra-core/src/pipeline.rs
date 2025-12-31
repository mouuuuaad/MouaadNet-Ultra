//! Real-Time Video Pipeline Module
//! =================================
//! High-performance video processing orchestration.
//!
//! Features:
//! - Frame preprocessing
//! - Detection decoding
//! - Full pipeline with timing
//! - Configurable parameters

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use std::time::Instant;

use crate::{Detection, ScaleInfo};

/// Processed frame ready for inference
#[pyclass]
#[derive(Clone)]
pub struct ProcessedFrame {
    #[pyo3(get)]
    pub tensor: Vec<f32>,
    #[pyo3(get)]
    pub scale_info: ScaleInfo,
}

#[pymethods]
impl ProcessedFrame {
    /// Get tensor as numpy array
    fn get_tensor<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.tensor.clone())
    }
}

/// Pipeline result with detections and timing
#[pyclass]
#[derive(Clone)]
pub struct PipelineResult {
    #[pyo3(get)]
    pub detections: Vec<Detection>,
    #[pyo3(get)]
    pub preprocess_ms: f64,
    #[pyo3(get)]
    pub postprocess_ms: f64,
    #[pyo3(get)]
    pub total_ms: f64,
}

/// Real-time video processing pipeline.
///
/// Manages preprocessing, postprocessing, and coordinate transforms
/// for efficient video inference.
#[pyclass]
#[derive(Clone)]
pub struct VideoPipeline {
    input_size: u32,
    threshold: f32,
    top_k: usize,
}

#[pymethods]
impl VideoPipeline {
    /// Create a new video pipeline.
    ///
    /// # Arguments
    /// * `input_size` - Model input size (e.g., 256)
    /// * `threshold` - Detection confidence threshold
    /// * `top_k` - Maximum detections per frame
    #[new]
    pub fn new(input_size: u32, threshold: f32, top_k: usize) -> Self {
        VideoPipeline {
            input_size,
            threshold,
            top_k,
        }
    }

    /// Get current input size
    #[getter]
    fn input_size(&self) -> u32 {
        self.input_size
    }

    /// Get current threshold
    #[getter]
    fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Set detection threshold
    #[setter]
    fn set_threshold(&mut self, value: f32) {
        self.threshold = value.clamp(0.01, 0.99);
    }

    /// Get top_k setting
    #[getter]
    fn top_k(&self) -> usize {
        self.top_k
    }

    /// Preprocess a BGR frame (OpenCV format) for inference.
    ///
    /// # Arguments
    /// * `frame` - BGR image from OpenCV (H, W, 3)
    ///
    /// # Returns
    /// * ProcessedFrame with tensor and scale info
    pub fn preprocess_frame<'py>(
        &self,
        py: Python<'py>,
        frame: PyReadonlyArray3<'py, u8>,
    ) -> PyResult<ProcessedFrame> {
        let (tensor, scale_info) =
            crate::preprocessing::preprocess_bgr(py, frame, self.input_size)?;

        Ok(ProcessedFrame {
            tensor: tensor.to_vec()?,
            scale_info,
        })
    }

    /// Decode model outputs to detections.
    ///
    /// # Arguments
    /// * `heatmap` - Detection heatmap (H, W) after sigmoid
    /// * `wh` - Width/height predictions (2, H, W)
    /// * `offset` - Offset predictions (2, H, W)
    ///
    /// # Returns
    /// * Vector of Detection objects in input coords
    pub fn decode(
        &self,
        heatmap: PyReadonlyArray2<f32>,
        wh: PyReadonlyArray3<f32>,
        offset: PyReadonlyArray3<f32>,
    ) -> Vec<Detection> {
        crate::postprocessing::decode_centernet(
            heatmap,
            wh,
            offset,
            self.threshold,
            self.top_k,
            self.input_size,
        )
    }

    /// Transform detections to original frame coordinates.
    ///
    /// # Arguments
    /// * `detections` - Detections in input image coords
    /// * `scale_info` - Scale info from preprocessing
    ///
    /// # Returns
    /// * Detections in original frame coordinates
    pub fn transform(&self, detections: Vec<Detection>, scale_info: &ScaleInfo) -> Vec<Detection> {
        crate::transforms::transform_detections(detections, scale_info)
    }

    /// Full postprocessing pipeline: decode + transform.
    ///
    /// # Arguments
    /// * `heatmap` - Detection heatmap (H, W) after sigmoid
    /// * `wh` - Width/height predictions (2, H, W)
    /// * `offset` - Offset predictions (2, H, W)
    /// * `scale_info` - Scale info from preprocessing
    ///
    /// # Returns
    /// * Detections in original frame coordinates
    pub fn postprocess(
        &self,
        heatmap: PyReadonlyArray2<f32>,
        wh: PyReadonlyArray3<f32>,
        offset: PyReadonlyArray3<f32>,
        scale_info: &ScaleInfo,
    ) -> Vec<Detection> {
        let detections = self.decode(heatmap, wh, offset);
        self.transform(detections, scale_info)
    }

    /// Full postprocessing with timing information.
    ///
    /// # Returns
    /// * PipelineResult with detections and timing
    pub fn postprocess_with_timing(
        &self,
        heatmap: PyReadonlyArray2<f32>,
        wh: PyReadonlyArray3<f32>,
        offset: PyReadonlyArray3<f32>,
        scale_info: &ScaleInfo,
    ) -> PipelineResult {
        let start = Instant::now();

        let decode_start = Instant::now();
        let detections = self.decode(heatmap, wh, offset);
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        let transform_start = Instant::now();
        let transformed = self.transform(detections, scale_info);
        let transform_ms = transform_start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        PipelineResult {
            detections: transformed,
            preprocess_ms: 0.0, // Not measured here
            postprocess_ms: decode_ms + transform_ms,
            total_ms,
        }
    }

    /// Pipeline info string
    fn __repr__(&self) -> String {
        format!(
            "VideoPipeline(input_size={}, threshold={:.2}, top_k={})",
            self.input_size, self.threshold, self.top_k
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = VideoPipeline::new(256, 0.3, 10);
        assert_eq!(pipeline.input_size, 256);
        assert!((pipeline.threshold - 0.3).abs() < 0.001);
        assert_eq!(pipeline.top_k, 10);
    }
}
