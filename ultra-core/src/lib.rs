//! ULTRA-CORE: High-Performance Extensions
//! ========================================
//! Rust acceleration for MouaadNet-Ultra neural network.
//!
//! Modules:
//! - preprocessing: Image resize, normalize, pad
//! - postprocessing: CenterNet decoding, NMS
//! - transforms: Coordinate transformations
//! - pipeline: Real-time video processing

use pyo3::prelude::*;
use pyo3::types::PyModule;

mod preprocessing;
mod postprocessing;
mod transforms;
mod pipeline;

/// Detection result structure
#[pyclass]
#[derive(Clone, Debug)]
pub struct Detection {
    #[pyo3(get, set)]
    pub x1: f32,
    #[pyo3(get, set)]
    pub y1: f32,
    #[pyo3(get, set)]
    pub x2: f32,
    #[pyo3(get, set)]
    pub y2: f32,
    #[pyo3(get, set)]
    pub confidence: f32,
}

#[pymethods]
impl Detection {
    #[new]
    fn new(x1: f32, y1: f32, x2: f32, y2: f32, confidence: f32) -> Self {
        Detection { x1, y1, x2, y2, confidence }
    }

    fn __repr__(&self) -> String {
        format!(
            "Detection(x1={:.1}, y1={:.1}, x2={:.1}, y2={:.1}, conf={:.3})",
            self.x1, self.y1, self.x2, self.y2, self.confidence
        )
    }

    fn to_tuple(&self) -> (f32, f32, f32, f32, f32) {
        (self.x1, self.y1, self.x2, self.y2, self.confidence)
    }

    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }
}

/// Scale info for coordinate transformation
#[pyclass]
#[derive(Clone, Debug)]
pub struct ScaleInfo {
    #[pyo3(get)]
    pub scale: f32,
    #[pyo3(get)]
    pub pad_top: u32,
    #[pyo3(get)]
    pub pad_left: u32,
    #[pyo3(get)]
    pub orig_height: u32,
    #[pyo3(get)]
    pub orig_width: u32,
    #[pyo3(get)]
    pub new_height: u32,
    #[pyo3(get)]
    pub new_width: u32,
}

#[pymethods]
impl ScaleInfo {
    #[new]
    fn new(
        scale: f32,
        pad_top: u32,
        pad_left: u32,
        orig_height: u32,
        orig_width: u32,
        new_height: u32,
        new_width: u32,
    ) -> Self {
        ScaleInfo {
            scale,
            pad_top,
            pad_left,
            orig_height,
            orig_width,
            new_height,
            new_width,
        }
    }
}

/// ULTRA-CORE Python Module
#[pymodule]
fn ultra_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<Detection>()?;
    m.add_class::<ScaleInfo>()?;
    m.add_class::<pipeline::VideoPipeline>()?;

    // Preprocessing functions
    m.add_function(wrap_pyfunction!(preprocessing::preprocess_image, m)?)?;
    m.add_function(wrap_pyfunction!(preprocessing::preprocess_bgr, m)?)?;

    // Postprocessing functions
    m.add_function(wrap_pyfunction!(postprocessing::decode_centernet, m)?)?;
    m.add_function(wrap_pyfunction!(postprocessing::sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(postprocessing::nms_max_pool, m)?)?;

    // Transform functions
    m.add_function(wrap_pyfunction!(transforms::transform_detections, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::clip_to_bounds, m)?)?;

    // Version info
    m.add("__version__", "1.0.0")?;
    m.add("__author__", "MOUAAD IDOUFKIR")?;

    Ok(())
}
