# Data

This directory contains dataset utilities and data files.

## Structure

```
data/
├── raw/           # Raw downloaded data (gitignored)
├── processed/     # Preprocessed data (gitignored)
└── README.md      # This file
```

## Supported Formats

MOUAADNET-ULTRA supports the following dataset formats:

### COCO Format
```json
{
  "images": [...],
  "annotations": [...],
  "categories": [{"id": 1, "name": "person"}]
}
```

### YOLO Format
```
image.jpg
image.txt  # class x_center y_center width height (normalized)
```

## Custom Dataset

To use your own dataset:

1. Place images in `data/raw/images/`
2. Place annotations in `data/raw/labels/`
3. Update `configs/default.yaml` with paths

```yaml
data:
  train_path: "data/raw/train"
  val_path: "data/raw/val"
```
