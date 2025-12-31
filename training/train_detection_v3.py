#!/usr/bin/env python3
"""
MOUAADNET-ULTRA v3: Full-Body Detection Pipeline
=================================================
Lead Architect: MOUAAD IDOUFKIR

FIXES FOR FULL-BODY DETECTION:
1. ASPP Module: Dilated convolutions for 3x larger receptive field
2. Deeper WH Branch: 2 conv layers + global context for better size regression
3. Higher WH Loss Weight: 1.0 instead of 0.1 for balanced learning
4. Larger Gaussian Radius: min=3 for single-peak heatmaps per person

Usage:
    python training/train_detection_v3.py --data /path/to/coco --epochs 50 --export
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION (V3 - FULL BODY OPTIMIZED)
# =============================================================================
class Config:
    """Training configuration optimized for full-body detection."""
    img_size: int = 256
    batch_size: int = 32
    num_workers: int = 4
    stride: int = 4  # Output is img_size/4 = 64
    
    epochs: int = 50
    lr: float = 1e-3
    max_lr: float = 1e-2
    weight_decay: float = 1e-4
    
    # Focal Loss params
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    
    # Loss weights (V3: INCREASED WH WEIGHT)
    hm_weight: float = 1.0
    wh_weight: float = 1.0  # V3: Changed from 0.1 to 1.0
    off_weight: float = 1.0
    
    # Gaussian params (V3: LARGER RADIUS)
    min_gaussian_radius: int = 3  # V3: Changed from 1 to 3
    gaussian_overlap: float = 0.5  # V3: Changed from 0.7 to 0.5 for larger blobs
    
    # Augmentation (V3: MULTI-SCALE)
    scale_range: Tuple[float, float] = (0.5, 1.5)  # V3: Wider range
    flip_prob: float = 0.5
    color_jitter: float = 0.3


# =============================================================================
# GAUSSIAN TARGET GENERATION (V3: LARGER RADIUS)
# =============================================================================
def gaussian_radius(height: float, width: float, min_overlap: float = 0.5) -> int:
    """Calculate Gaussian radius - V3 uses lower overlap for larger radius."""
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = max(0, b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + math.sqrt(sq1)) / 2
    
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = max(0, b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + math.sqrt(sq2)) / 2
    
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = max(0, b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + math.sqrt(sq3)) / 2
    
    return max(0, int(min(r1, r2, r3)))


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
    """Draw Gaussian blob on heatmap."""
    diameter = 2 * radius + 1
    gaussian = np.zeros((diameter, diameter), dtype=np.float32)
    
    sigma = diameter / 6
    for i in range(diameter):
        for j in range(diameter):
            x, y = i - radius, j - radius
            gaussian[j, i] = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
    
    h, w = heatmap.shape
    left = min(cx, radius)
    right = min(w - cx, radius + 1)
    top = min(cy, radius)
    bottom = min(h - cy, radius + 1)
    
    if left + right > 0 and top + bottom > 0:
        hm_slice = heatmap[cy - top:cy + bottom, cx - left:cx + right]
        g_slice = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        np.maximum(hm_slice, g_slice, out=hm_slice)
    
    return heatmap


# =============================================================================
# DATASET
# =============================================================================
class COCOPersonDataset(Dataset):
    """COCO Person Detection Dataset with V3 Gaussian targets."""
    
    def __init__(self, data_dir: str, split: str = 'train', config: Config = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config or Config()
        self.img_size = self.config.img_size
        self.stride = self.config.stride
        self.out_size = self.img_size // self.stride
        self.augment = (split == 'train')
        
        # Load COCO
        from pycocotools.coco import COCO
        anno_path = self.data_dir / 'annotations' / f'instances_{split}2017.json'
        self.coco = COCO(str(anno_path))
        
        # Get images with persons and verify they exist
        all_ids = self.coco.getImgIds(catIds=[1])
        self.img_ids = []
        img_dir = self.data_dir / f'{split}2017'
        for img_id in all_ids:
            info = self.coco.loadImgs(img_id)[0]
            if (img_dir / info['file_name']).exists():
                self.img_ids.append(img_id)
        
        logger.info(f"{split}: {len(self.img_ids)}/{len(all_ids)} images verified")
        
        # Normalization (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def _load_image(self, idx: int) -> Tuple[np.ndarray, List]:
        """Load image and annotations."""
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = self.data_dir / f'{self.split}2017' / info['file_name']
        
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get person bboxes
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = [ann['bbox'] for ann in anns if ann['bbox'][2] > 1 and ann['bbox'][3] > 1]
        
        return img, bboxes
    
    def _augment(self, img: np.ndarray, bboxes: List) -> Tuple[np.ndarray, List]:
        """Apply augmentations with V3 multi-scale training."""
        h, w = img.shape[:2]
        
        # Random horizontal flip
        if np.random.random() < self.config.flip_prob:
            img = img[:, ::-1].copy()
            bboxes = [(w - x - bw, y, bw, bh) for x, y, bw, bh in bboxes]
        
        # V3: Wider scale range for multi-scale training
        scale = np.random.uniform(*self.config.scale_range)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        bboxes = [(x * scale, y * scale, bw * scale, bh * scale) for x, y, bw, bh in bboxes]
        
        # Color jitter
        if np.random.random() < 0.5:
            jitter = self.config.color_jitter
            img = img.astype(np.float32)
            img *= np.random.uniform(1 - jitter, 1 + jitter, (1, 1, 3))
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img, bboxes
    
    def _preprocess(self, img: np.ndarray, bboxes: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resize, pad, and generate V3 targets with larger Gaussian radius."""
        h, w = img.shape[:2]
        
        # Resize maintaining aspect ratio
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Pad to square
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        top = pad_h // 2
        left = pad_w // 2
        
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = img
        
        # Normalize
        img_norm = canvas.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std
        
        # Create targets
        heatmap = np.zeros((self.out_size, self.out_size), dtype=np.float32)
        wh_target = np.zeros((2, self.out_size, self.out_size), dtype=np.float32)
        offset_target = np.zeros((2, self.out_size, self.out_size), dtype=np.float32)
        mask = np.zeros((self.out_size, self.out_size), dtype=np.float32)
        
        for x, y, bw, bh in bboxes:
            # Transform bbox to input space
            x = x * scale + left
            y = y * scale + top
            bw = bw * scale
            bh = bh * scale
            
            # Center in output space
            cx = (x + bw / 2) / self.stride
            cy = (y + bh / 2) / self.stride
            
            if 0 <= cx < self.out_size and 0 <= cy < self.out_size:
                cx_int, cy_int = int(cx), int(cy)
                
                # V3: LARGER Gaussian radius
                radius = max(
                    self.config.min_gaussian_radius,  # Minimum radius = 3
                    gaussian_radius(bh / self.stride, bw / self.stride, 
                                   self.config.gaussian_overlap)
                )
                draw_gaussian(heatmap, cx_int, cy_int, radius)
                
                # WH target (normalized)
                wh_target[0, cy_int, cx_int] = bw / self.img_size
                wh_target[1, cy_int, cx_int] = bh / self.img_size
                
                # Offset (sub-pixel adjustment)
                offset_target[0, cy_int, cx_int] = cx - cx_int
                offset_target[1, cy_int, cx_int] = cy - cy_int
                
                mask[cy_int, cx_int] = 1
        
        return img_norm, heatmap, wh_target, offset_target, mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            img, bboxes = self._load_image(idx)
            
            if self.augment and bboxes:
                img, bboxes = self._augment(img, bboxes)
            
            img, heatmap, wh, offset, mask = self._preprocess(img, bboxes)
            
            return {
                'image': torch.from_numpy(img.transpose(2, 0, 1)),
                'heatmap': torch.from_numpy(heatmap[None]),
                'wh': torch.from_numpy(wh),
                'offset': torch.from_numpy(offset),
                'mask': torch.from_numpy(mask),
            }
        except Exception as e:
            logger.warning(f"Error loading {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))


# =============================================================================
# MODEL V3: WITH ASPP AND GLOBAL CONTEXT
# =============================================================================
class ConvBNSiLU(nn.Module):
    """Conv + BatchNorm + SiLU activation."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, d: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    """Depthwise Separable Convolution for efficiency."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class ASPPModule(nn.Module):
    """
    V3: Atrous Spatial Pyramid Pooling for large receptive field.
    
    Increases effective receptive field from ~96px to ~300px,
    allowing the model to "see" entire persons.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # 1x1 conv (rate=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        
        # 3x3 conv with dilation=6
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        
        # 3x3 conv with dilation=12
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        
        # 3x3 conv with dilation=18
        self.conv18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        
        # Global Average Pooling branch (no BatchNorm - 1x1 output incompatible)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),  # Use bias since no BN
            nn.SiLU(inplace=True),
        )
        
        # Fuse all branches
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Apply all branches
        out1 = self.conv1(x)
        out6 = self.conv6(x)
        out12 = self.conv12(x)
        out18 = self.conv18(x)
        out_gap = F.interpolate(self.gap(x), size=size, mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        out = torch.cat([out1, out6, out12, out18, out_gap], dim=1)
        out = self.fuse(out)
        
        return out


class GlobalContextBlock(nn.Module):
    """
    V3: Global context injection for WH prediction.
    
    Uses global average pooling + FC to inject scene-level context,
    helping the model predict full-body sizes.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        # Global context
        context = self.gap(x).view(b, c)
        scale = self.fc(context).view(b, c, 1, 1)
        # Apply scale
        return x * scale


class NanoBackboneV3(nn.Module):
    """V3: Ultra-lightweight backbone with ASPP for large receptive field."""
    
    def __init__(self, width_mult: float = 1.0):
        super().__init__()
        
        def ch(c): return max(8, int(c * width_mult))
        
        # Stem: 256 -> 64
        self.stem = nn.Sequential(
            ConvBNSiLU(3, ch(32), 3, 2, 1),      # /2 -> 128
            DepthwiseSeparable(ch(32), ch(64), 2),  # /2 -> 64
        )
        
        # Stage 2: 64 -> 32
        self.stage2 = nn.Sequential(
            DepthwiseSeparable(ch(64), ch(128), 2),  # /2 -> 32
            DepthwiseSeparable(ch(128), ch(128)),
        )
        
        # Stage 3: 32 -> 16
        self.stage3 = nn.Sequential(
            DepthwiseSeparable(ch(128), ch(256), 2),  # /2 -> 16
            DepthwiseSeparable(ch(256), ch(256)),
        )
        
        # V3: ASPP Module for large receptive field
        self.aspp = ASPPModule(ch(256), ch(256))
        
        self.out_channels = ch(256)
    
    def forward(self, x):
        x = self.stem(x)    # 64x64
        x = self.stage2(x)  # 32x32
        x = self.stage3(x)  # 16x16
        x = self.aspp(x)    # 16x16 with large receptive field
        return x


class DecoupledHeadV3(nn.Module):
    """
    V3: Decoupled Detection Head with improved WH branch.
    
    Changes from V2:
    - WH branch has 2 conv layers (was 1)
    - WH branch has global context injection
    """
    
    def __init__(self, in_ch: int, hidden_ch: int = 64):
        super().__init__()
        
        # Upsample 16x16 -> 64x64 (stride 4)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_ch, hidden_ch, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(hidden_ch, hidden_ch, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True),
        )
        
        # HEATMAP branch
        self.hm_conv = nn.Sequential(
            ConvBNSiLU(hidden_ch, hidden_ch, 3, 1, 1),
            ConvBNSiLU(hidden_ch, hidden_ch, 3, 1, 1),
        )
        self.hm_out = nn.Conv2d(hidden_ch, 1, 1)
        
        # V3: DEEPER WH (width/height) branch with global context
        self.wh_conv = nn.Sequential(
            ConvBNSiLU(hidden_ch, hidden_ch, 3, 1, 1),
            ConvBNSiLU(hidden_ch, hidden_ch, 3, 1, 1),  # V3: Added extra layer
        )
        self.wh_context = GlobalContextBlock(hidden_ch)  # V3: Global context
        self.wh_out = nn.Conv2d(hidden_ch, 2, 1)
        
        # OFFSET branch (sub-pixel)
        self.off_conv = nn.Sequential(
            ConvBNSiLU(hidden_ch, hidden_ch, 3, 1, 1),
        )
        self.off_out = nn.Conv2d(hidden_ch, 2, 1)
        
        # CRITICAL: Initialize heatmap bias to prevent zero output
        self.hm_out.bias.data.fill_(-2.19)  # sigmoid(-2.19) ≈ 0.1
    
    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.upsample(x)
        
        hm = self.hm_conv(x)
        hm = self.hm_out(hm)  # Raw logits
        
        # V3: WH with deeper conv and global context
        wh = self.wh_conv(x)
        wh = self.wh_context(wh)
        wh = self.wh_out(wh)
        
        off = self.off_conv(x)
        off = self.off_out(off)
        
        return {
            'heatmap': hm,
            'wh': wh,
            'offset': off,
        }


class MouaadNetUltraV3(nn.Module):
    """
    MOUAADNET-ULTRA v3: Full-Body Detection
    
    Improvements over V2:
    - ASPP module in backbone for 3x larger receptive field
    - Deeper WH branch (2 conv layers)
    - Global context injection for WH prediction
    
    Architecture:
    - Backbone: NanoBackboneV3 with ASPP (~550k params)
    - Head: DecoupledHeadV3 (Heatmap + WH with context + Offset)
    
    Total: ~1M parameters
    """
    
    def __init__(self, width_mult: float = 1.0):
        super().__init__()
        self.backbone = NanoBackboneV3(width_mult)
        self.head = DecoupledHeadV3(self.backbone.out_channels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs


# =============================================================================
# LOSS V3: BALANCED WEIGHTS
# =============================================================================
class DetectionLossV3(nn.Module):
    """
    V3: Detection Loss with balanced WH weight.
    
    Changes from V2:
    - WH weight = 1.0 (was 0.1)
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, 
                 hm_weight: float = 1.0, wh_weight: float = 1.0, off_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Numerically stable Focal loss."""
        eps = 1e-6
        pred_sig = torch.sigmoid(pred)
        pred_sig = torch.clamp(pred_sig, eps, 1 - eps)
        
        pos_mask = target.ge(0.99).float()
        neg_mask = target.lt(0.99).float()
        
        num_pos = pos_mask.sum()
        
        if num_pos == 0:
            neg_weight = torch.pow(1 - target + eps, self.beta)
            neg_loss = -torch.log(1 - pred_sig + eps) * torch.pow(pred_sig, self.alpha) * neg_weight * neg_mask
            return neg_loss.sum() / max(neg_mask.sum().item(), 1)
        
        pos_loss = -torch.log(pred_sig + eps) * torch.pow(1 - pred_sig + eps, self.alpha) * pos_mask
        neg_weight = torch.pow(1 - target + eps, self.beta)
        neg_loss = -torch.log(1 - pred_sig + eps) * torch.pow(pred_sig + eps, self.alpha) * neg_weight * neg_mask
        
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss
    
    def l1_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """L1 loss for WH and Offset."""
        mask = mask.unsqueeze(1).expand_as(pred)
        num = mask.sum()
        
        if num == 0:
            return torch.tensor(0.0, device=pred.device)
        
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        return loss / num
    
    def forward(self, pred: Dict, target: Dict) -> Dict[str, torch.Tensor]:
        hm_loss = self.focal_loss(pred['heatmap'], target['heatmap'])
        wh_loss = self.l1_loss(pred['wh'], target['wh'], target['mask'])
        off_loss = self.l1_loss(pred['offset'], target['offset'], target['mask'])
        
        # V3: Balanced weights
        total = (self.hm_weight * hm_loss + 
                 self.wh_weight * wh_loss + 
                 self.off_weight * off_loss)
        
        if not torch.isfinite(total):
            total = torch.tensor(0.0, device=pred['heatmap'].device, requires_grad=True)
        
        return {
            'total': total,
            'hm': hm_loss.detach() if torch.isfinite(hm_loss) else torch.tensor(0.0),
            'wh': wh_loss.detach() if torch.isfinite(wh_loss) else torch.tensor(0.0),
            'off': off_loss.detach() if torch.isfinite(off_loss) else torch.tensor(0.0),
        }


# =============================================================================
# TRAINER
# =============================================================================
class Trainer:
    """Training loop with 1CycleLR."""
    
    def __init__(self, model, train_loader, val_loader, config, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        
        # 1CycleLR: 30% warmup, then anneal
        steps = len(train_loader) * config.epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.max_lr,
            total_steps=steps,
            pct_start=0.3,
            anneal_strategy='cos',
        )
        
        # V3: Use balanced loss weights
        self.criterion = DetectionLossV3(
            config.focal_alpha, config.focal_beta,
            config.hm_weight, config.wh_weight, config.off_weight
        )
        self.scaler = torch.amp.GradScaler('cuda')
        
        self.epoch = 0
        self.best_loss = float('inf')
        self.history = {'train': [], 'val': [], 'wh': []}
    
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_wh_loss = 0.0
        n_batches = 0
        
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch in pbar:
            imgs = batch['image'].to(self.device)
            targets = {k: batch[k].to(self.device) for k in ['heatmap', 'wh', 'offset', 'mask']}
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                preds = self.model(imgs)
                losses = self.criterion(preds, targets)
                loss = losses['total']
            
            if torch.isfinite(loss):
                loss_val = loss.item()
                total_loss += loss_val
                total_wh_loss += losses['wh'].item()
                n_batches += 1
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if self.scaler.get_scale() >= old_scale:
                    self.scheduler.step()
                
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{loss_val:.2f}",
                    hm=f"{losses['hm'].item():.2f}",
                    wh=f"{losses['wh'].item():.2f}",  # V3: Track WH loss
                    lr=f"{lr:.1e}"
                )
            else:
                logger.warning(f"NaN/Inf loss at batch {n_batches}")
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_wh = total_wh_loss / max(n_batches, 1)
        return avg_loss, avg_wh
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total = 0
        
        for batch in self.val_loader:
            imgs = batch['image'].to(self.device)
            targets = {k: batch[k].to(self.device) for k in ['heatmap', 'wh', 'offset', 'mask']}
            
            preds = self.model(imgs)
            losses = self.criterion(preds, targets)
            total += losses['total'].item()
        
        return total / len(self.val_loader)
    
    def train(self):
        logger.info(f"Training MOUAADNET-ULTRA V3 on {self.device}")
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"V3 Config: wh_weight={self.config.wh_weight}, min_radius={self.config.min_gaussian_radius}")
        
        for self.epoch in range(self.config.epochs):
            train_loss, wh_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.history['train'].append(train_loss)
            self.history['val'].append(val_loss)
            self.history['wh'].append(wh_loss)
            
            logger.info(f"Epoch {self.epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}, wh={wh_loss:.4f}")
            
            # Save
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'history': self.history,
                'config': {
                    'wh_weight': self.config.wh_weight,
                    'min_gaussian_radius': self.config.min_gaussian_radius,
                },
            }, self.save_dir / 'latest.pt')
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                }, self.save_dir / 'best.pt')
                logger.info(f"*** New best: {val_loss:.4f}")
        
        logger.info(f"Training complete. Best: {self.best_loss:.4f}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="MOUAADNET-ULTRA V3: Full-Body Detection Training")
    parser.add_argument('--data', type=str, required=True, help="Path to COCO dataset")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--save-dir', type=str, default='checkpoints_v3')
    parser.add_argument('--export', action='store_true', help="Export ONNX after training")
    args = parser.parse_args()
    
    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    
    logger.info("=" * 60)
    logger.info("MOUAADNET-ULTRA V3: Full-Body Detection Training")
    logger.info("=" * 60)
    logger.info(f"V3 Improvements:")
    logger.info(f"  - ASPP module for 3x larger receptive field")
    logger.info(f"  - WH loss weight: {config.wh_weight} (was 0.1)")
    logger.info(f"  - Min gaussian radius: {config.min_gaussian_radius} (was 1)")
    logger.info(f"  - Deeper WH branch with global context")
    logger.info("=" * 60)
    
    # Data
    train_ds = COCOPersonDataset(args.data, 'train', config)
    val_ds = COCOPersonDataset(args.data, 'val', config)
    
    train_loader = DataLoader(
        train_ds, config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, config.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Model
    model = MouaadNetUltraV3()
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config, args.save_dir)
    trainer.train()
    
    # Export
    if args.export:
        logger.info("Exporting ONNX...")
        model.eval().cpu()
        
        # Load best weights
        checkpoint = torch.load(f'{args.save_dir}/best.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        dummy_input = torch.randn(1, 3, 256, 256)
        onnx_path = f'{args.save_dir}/detection_v3.onnx'
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['images'],
            output_names=['heatmap', 'wh', 'offset'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'heatmap': {0: 'batch_size'},
                'wh': {0: 'batch_size'},
                'offset': {0: 'batch_size'},
            },
            opset_version=11,
        )
        logger.info(f"Exported: {onnx_path}")


if __name__ == "__main__":
    main()
