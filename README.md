# MOUAADNET-ULTRA

<div align="center">

**High-Efficiency Human Detection Network**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🎯 Features

- **Ultra-Lightweight**: ~0.61M parameters for V3, ~868K for V2
- **CenterNet Architecture**: Anchor-free detection with heatmap + size regression
- **Real-Time**: ONNX export for fast inference
- **Production-Ready**: Webcam demo included

## 🏗️ Architecture (V3)

```
Input (256×256×3)
       ↓
┌─────────────────────────────────┐
│  NANO-BACKBONE                  │
│  DepthwiseSeparable Convs       │
│  32 → 64 → 128 → 256 channels   │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│  ASPP MODULE (V3)               │
│  Dilated Convs: 1, 6, 12, 18    │
│  ~300px receptive field         │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│  DECOUPLED HEAD (V3)            │
│  ├─ Heatmap: 2 conv layers      │
│  ├─ WH: 2 conv + GlobalContext  │
│  └─ Offset: 1 conv layer        │
└─────────────────────────────────┘
       ↓
Output: Heatmap (64×64) + WH + Offset
```

## 🚀 Quick Start

### Webcam Demo (ONNX)
```bash
python examples/webcam_onnx_demo.py --model detection.onnx --threshold 0.1
```

### Python API
```python
import torch
from training.train_detection_v3 import MouaadNetUltraV3

model = MouaadNetUltraV3()
model.eval()

# Inference
image = torch.randn(1, 3, 256, 256)
outputs = model(image)

print(f"Heatmap: {outputs['heatmap'].shape}")  # [1, 1, 64, 64]
print(f"WH: {outputs['wh'].shape}")            # [1, 2, 64, 64]
print(f"Offset: {outputs['offset'].shape}")    # [1, 2, 64, 64]
```

## 📦 Installation

```bash
git clone https://github.com/mouuuuaad/MouaadNet-Ultra.git
cd MouaadNet-Ultra
pip install -r requirements.txt
```

## 🎓 Training

### V3 (Recommended - Full Body Detection)
```bash
python training/train_detection_v3.py \
    --data /path/to/coco \
    --epochs 50 \
    --export
```

**V3 Improvements:**
| Feature | V2 | V3 |
|---------|----|----|
| Receptive Field | ~96px | ~300px (ASPP) |
| WH Loss Weight | 0.1 | 1.0 |
| Min Gaussian Radius | 1 | 3 |
| WH Branch | 1 conv | 2 conv + GlobalContext |

## 📁 Project Structure

```
MouaadNet-Ultra/
├── mouaadnet_ultra/          # Core library (V1/V2)
│   ├── backbone/             # Nano-backbone
│   ├── neck/                 # Slim-PAN
│   └── heads/                # Detection heads
├── training/
│   ├── train_detection_v2.py # V2 training
│   └── train_detection_v3.py # V3 training (recommended)
├── examples/
│   ├── webcam_onnx_demo.py   # ONNX webcam demo
│   └── webcam_demo.py        # PyTorch webcam demo
├── configs/                  # Configuration files
├── detection.onnx            # Pre-trained V2 model
└── requirements.txt
```

## 📤 Export to ONNX

After training, models are exported automatically with `--export`:
```bash
# Output: checkpoints_v3/detection_v3.onnx
python training/train_detection_v3.py --data /path/to/coco --epochs 50 --export
```

## 📊 Model Variants

| Version | Parameters | Receptive Field | Use Case |
|---------|------------|-----------------|----------|
| V2 | 868K | ~96px | Fast/Mobile |
| **V3** | 610K | ~300px | Full-body detection |

## 👤 Author

**MOUAAD IDOUFKIR** - Lead Architect

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">
Made with ❤️ by MOUAAD IDOUFKIR
</div>
