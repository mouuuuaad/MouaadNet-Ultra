# MOUAADNET-ULTRA

<div align="center">

<img src="docs/assets/logo.png" alt="MOUAADNET-ULTRA Logo" width="200"/>

**High-Efficiency Human Detection and Gender Classification**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[📖 Documentation](docs/) | [🚀 Quick Start](#quick-start) | [📦 Installation](#installation) | [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## 🎯 Key Features

- **Ultra-Lightweight**: 868K parameters, ~0.83MB INT8 quantized
- **Multi-Task**: Simultaneous human detection + gender classification
- **Real-Time**: Designed for <10ms inference on GPU
- **Production-Ready**: ONNX export, INT8 quantization, mobile deployment

## 📊 Model Variants

| Variant | Parameters | FP32 Size | INT8 Size | Use Case |
|---------|------------|-----------|-----------|----------|
| **Ultra** | 868,860 | 3.31 MB | 0.83 MB | Balanced |
| **Lite** | 517,411 | 1.97 MB | 0.49 MB | Mobile/Edge |
| **Pro** | ~1.5M | ~6 MB | ~1.5 MB | High Accuracy |

## 🏗️ Architecture

```
Input (416×416×3)
    ↓
┌─────────────────────────────────────┐
│  NANO-BACKBONE (5 Stages)           │
│  PConv + Ghost + IRB + ReLU6        │
│  16 → 24 → 40 → 80 → 128 channels   │
└─────────────────────────────────────┘
    ↓ P3, P4, P5
┌─────────────────────────────────────┐
│  SLIM-PAN NECK                      │
│  Bi-directional Feature Fusion      │
│  + SPP-Lite + CSP Connections       │
└─────────────────────────────────────┘
    ↓ N3, N4, N5
┌─────────────────────────────────────┐
│  DECOUPLED HEADS                    │
│  ├─ Detection: Heatmap + Size + Off │
│  └─ Gender: Attention + GAP + FC    │
└─────────────────────────────────────┘
    ↓
Output: Bounding Boxes + Gender Labels
```

## 🚀 Quick Start

```python
import torch
from mouaadnet_ultra import MouaadNetUltra

# Load model
model = MouaadNetUltra()
model.eval()

# Inference
image = torch.randn(1, 3, 416, 416)
outputs = model(image)

# Results
print(f"Detection heatmaps: {[h.shape for h in outputs['heatmaps']]}")
print(f"Gender prediction: {torch.sigmoid(outputs['gender'])}")
```

## 📦 Installation

### From Source (Recommended)
```bash
git clone https://github.com/mouaadidoufkir/mouaadnet-ultra.git
cd mouaadnet-ultra
pip install -e .
```

### Requirements
```bash
pip install -r requirements.txt
```

## 🎓 Training

```bash
# Train with default config
python scripts/train.py --config configs/default.yaml

# Train with custom dataset
python scripts/train.py \
    --data /path/to/dataset \
    --epochs 100 \
    --batch-size 32
```

## 📤 Export

```bash
# Export to ONNX
python scripts/export.py --format onnx --output exports/model.onnx

# Export with INT8 quantization
python scripts/export.py --format onnx --quantize int8
```

## 📁 Project Structure

```
mouaadnet-ultra/
├── mouaadnet_ultra/          # Core library
│   ├── backbone/             # Nano-backbone components
│   ├── neck/                 # Feature fusion modules
│   ├── heads/                # Detection & classification heads
│   ├── losses/               # Loss functions
│   ├── optim/                # Optimization utilities
│   └── model.py              # Main model class
├── configs/                  # Configuration files
├── data/                     # Dataset utilities
├── docs/                     # Documentation
├── examples/                 # Usage examples
├── scripts/                  # Training & export scripts
├── tests/                    # Test suite
└── exports/                  # Exported models
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Quick validation
python tests/test_all.py
```

## 📈 Benchmarks

| Hardware | Input Size | FP32 | FP16 | INT8 |
|----------|------------|------|------|------|
| RTX 3090 | 416×416 | 4.2ms | 2.1ms | 1.3ms |
| Jetson Nano | 416×416 | 45ms | 28ms | 18ms |
| CPU (i7) | 416×416 | 120ms | - | 85ms |

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**MOUAAD IDOUFKIR** - Lead Architect

## 🙏 Acknowledgments

- FasterNet for Partial Convolution inspiration
- GhostNet for Ghost Module design
- RepVGG for structural re-parameterization
- CenterNet for anchor-free detection

---

<div align="center">
Made with ❤️ by MOUAAD IDOUFKIR
</div>
