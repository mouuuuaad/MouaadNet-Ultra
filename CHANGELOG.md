# Changelog

All notable changes to MOUAADNET-ULTRA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2024-12-29

### Added
- Initial release of MOUAADNET-ULTRA
- **Backbone**: 5-stage Nano-Backbone with PConv, Ghost Module, and IRB
- **Neck**: Slim-PAN with SPP-Lite and CSP connections
- **Heads**: Decoupled detection and gender classification heads
- **Losses**: CIoU, Focal, and Multi-Task loss functions
- **Optimization**: RepVGG blocks, One-Cycle LR scheduler
- **Export**: ONNX export with layer fusion
- **Quantization**: INT8 PTQ and QAT support
- Model variants: Ultra, Lite, and Pro
- Comprehensive test suite
- Documentation and examples

### Model Statistics
- Parameters: 868,860
- FP32 Size: 3.31 MB
- INT8 Size: 0.83 MB

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2024-12-29 | Initial release |
