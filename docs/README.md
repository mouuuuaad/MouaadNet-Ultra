# MOUAADNET-ULTRA Documentation

Welcome to the MOUAADNET-ULTRA documentation!

## 📚 Table of Contents

- [Getting Started](getting_started.md)
- [Architecture](architecture.md)
- [Training Guide](training.md)
- [Export & Deployment](deployment.md)
- [API Reference](api/README.md)

## 🎯 Quick Links

| Topic | Description |
|-------|-------------|
| [Installation](getting_started.md#installation) | How to install MOUAADNET-ULTRA |
| [Quick Start](getting_started.md#quick-start) | Basic usage examples |
| [Training](training.md) | Train on your own dataset |
| [Export](deployment.md#onnx-export) | Export to ONNX/TensorRT |

## 🏗️ Architecture Overview

MOUAADNET-ULTRA is a lightweight multi-task neural network for:
- **Human Detection**: Anchor-free CenterNet-style detection
- **Gender Classification**: Attention-based binary classification

### Key Components

```
┌─────────────────────────────────────────────────────────┐
│                    MOUAADNET-ULTRA                      │
├─────────────────────────────────────────────────────────┤
│  BACKBONE: 5-stage Nano-Backbone                        │
│  ├── Partial Convolution (PConv)                        │
│  ├── Ghost Module                                       │
│  └── Inverted Residual Block (IRB)                      │
├─────────────────────────────────────────────────────────┤
│  NECK: Slim-PAN                                         │
│  ├── Bi-directional Feature Fusion                      │
│  ├── SPP-Lite                                           │
│  └── CSP Connections                                    │
├─────────────────────────────────────────────────────────┤
│  HEADS: Decoupled Architecture                          │
│  ├── Detection: Heatmap + Size + Offset                 │
│  └── Gender: CBAM Attention + GAP                       │
└─────────────────────────────────────────────────────────┘
```

## 📊 Model Variants

| Variant | Parameters | INT8 Size | Target |
|---------|------------|-----------|--------|
| Ultra | 868,860 | 0.83 MB | Balanced |
| Lite | 517,411 | 0.49 MB | Edge/Mobile |
| Pro | ~1.5M | ~1.5 MB | High Accuracy |

## 🔗 Resources

- [GitHub Repository](https://github.com/mouaadidoufkir/mouaadnet-ultra)
- [Issue Tracker](https://github.com/mouaadidoufkir/mouaadnet-ultra/issues)
- [Changelog](../CHANGELOG.md)
