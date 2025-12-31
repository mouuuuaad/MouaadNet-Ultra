#!/usr/bin/env python3
"""
MOUAADNET-ULTRA v2: Complete Detection Pipeline
================================================
Lead Architect: MOUAAD IDOUFKIR

FIXES ZERO-CONFIDENCE ISSUE:
1. Decoupled Head: Separate Heatmap/WH/Offset branches
2. Focal Loss: Forces focus on person pixels, ignores background
3. Gaussian Targets: Soft blobs instead of single pixels
4. Proper initialization: Prevents dead neurons

Usage:
    python training/train_detection_v2.py --data /path/to/coco --epochs 50
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
# CONFIGURATION
# =============================================================================
class Config:
    """Training configuration."""
    img_size: int = 256
    batch_size: int = 32
    num_workers: int = 4
    stride: int = 4  # Output is img_size/4 = 64
    
    epochs: int = 50
    lr: float = 1e-3
    max_lr: float = 1e-2
    weight_decay: float = 1e-4
    
    # Focal Loss params (CRITICAL for sparse heatmaps)
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    
    # Loss weights
    hm_weight: float = 1.0
    wh_weight: float = 0.1
    off_weight: float = 1.0
    
    # Augmentation
    scale_range: Tuple[float, float] = (0.6, 1.4)
    flip_prob: float = 0.5
    color_jitter: float = 0.3


# =============================================================================
# GAUSSIAN TARGET GENERATION
# =============================================================================
def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> int:
    """Calculate Gaussian radius based on object size."""
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
    """COCO Person Detection Dataset with Gaussian targets."""
    
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
        """Apply augmentations."""
        h, w = img.shape[:2]
        
        # Random horizontal flip
        if np.random.random() < self.config.flip_prob:
            img = img[:, ::-1].copy()
            bboxes = [(w - x - bw, y, bw, bh) for x, y, bw, bh in bboxes]
        
        # Random scale
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
        """Resize, pad, and generate targets."""
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
                
                # Gaussian radius based on object size
                radius = max(1, gaussian_radius(bh / self.stride, bw / self.stride))
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
# MODEL: DECOUPLED HEAD ARCHITECTURE
# =============================================================================
class ConvBNSiLU(nn.Module):
    """Conv + BatchNorm + SiLU activation."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
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


class NanoBackbone(nn.Module):
    """Ultra-lightweight backbone (~400k params)."""
    
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
        
        self.out_channels = ch(256)
    
    def forward(self, x):
        x = self.stem(x)    # 64x64
        x = self.stage2(x)  # 32x32
        x = self.stage3(x)  # 16x16
        return x


class DecoupledHead(nn.Module):
    """
    Decoupled Detection Head.
    
    Separate branches for:
    - Heatmap: Person center locations (sigmoid output)
    - WH: Width/Height predictions
    - Offset: Sub-pixel adjustments
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
        
        # WH (width/height) branch
        self.wh_conv = nn.Sequential(
            ConvBNSiLU(hidden_ch, hidden_ch, 3, 1, 1),
        )
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
        hm = self.hm_out(hm)  # Raw logits (apply sigmoid in loss/inference)
        
        wh = self.wh_conv(x)
        wh = self.wh_out(wh)
        
        off = self.off_conv(x)
        off = self.off_out(off)
        
        return {
            'heatmap': hm,
            'wh': wh,
            'offset': off,
        }


class MouaadNetUltraV2(nn.Module):
    """
    MOUAADNET-ULTRA v2
    
    Architecture:
    - Backbone: Ultra-lightweight NanoBackbone (~400k params)
    - Head: Decoupled (Heatmap + WH + Offset)
    
    Total: ~868k parameters
    """
    
    def __init__(self, width_mult: float = 1.0):
        super().__init__()
        self.backbone = NanoBackbone(width_mult)
        self.head = DecoupledHead(self.backbone.out_channels)
        
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
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs


# =============================================================================
# LOSS: FOCAL + L1
# =============================================================================
class DetectionLoss(nn.Module):
    """
    Combined Detection Loss.
    
    - Focal Loss for heatmap (handles sparse positives)
    - L1 Loss for WH and Offset
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss for heatmap with Gaussian targets."""
        pred = torch.clamp(torch.sigmoid(pred), 1e-4, 1 - 1e-4)
        
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        # Positive: focus on hard positive pixels
        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        
        # Negative: weight by distance from center (Gaussian)
        neg_weight = torch.pow(1 - target, self.beta)
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weight * neg_mask
        
        num_pos = pos_mask.sum().clamp(min=1)
        return (pos_loss.sum() + neg_loss.sum()) / num_pos
    
    def l1_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """L1 loss for WH and Offset."""
        mask = mask.unsqueeze(1).expand_as(pred)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        num = mask.sum().clamp(min=1)
        return loss / num
    
    def forward(self, pred: Dict, target: Dict) -> Dict[str, torch.Tensor]:
        hm_loss = self.focal_loss(pred['heatmap'], target['heatmap'])
        wh_loss = self.l1_loss(pred['wh'], target['wh'], target['mask'])
        off_loss = self.l1_loss(pred['offset'], target['offset'], target['mask'])
        
        total = hm_loss + 0.1 * wh_loss + 1.0 * off_loss
        
        return {
            'total': total,
            'hm': hm_loss.detach(),
            'wh': wh_loss.detach(),
            'off': off_loss.detach(),
        }


# =============================================================================
# INFERENCE
# =============================================================================
class Detector:
    """Robust inference pipeline."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', img_size: int = 256):
        self.model = model.to(device).eval()
        self.device = device
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, img: np.ndarray) -> Tuple[torch.Tensor, float, int, int]:
        """Preprocess image for inference."""
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(img, (new_w, new_h))
        
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        top = pad_h // 2
        left = pad_w // 2
        
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized
        
        # Normalize
        img_norm = canvas.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std
        
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(self.device), scale, top, left
    
    @torch.no_grad()
    def detect(self, img: np.ndarray, threshold: float = 0.3, top_k: int = 100) -> List[Dict]:
        """
        Run detection on image.
        
        Returns list of {'box': [x1,y1,x2,y2], 'score': float}
        """
        h, w = img.shape[:2]
        tensor, scale, pad_top, pad_left = self.preprocess(img)
        
        outputs = self.model(tensor)
        heatmap = torch.sigmoid(outputs['heatmap'][0, 0])
        wh = outputs['wh'][0]
        offset = outputs['offset'][0]
        
        # Find peaks
        hm_np = heatmap.cpu().numpy()
        coords = np.where(hm_np >= threshold)
        
        detections = []
        for cy, cx in zip(*coords):
            score = hm_np[cy, cx]
            
            # Get WH and offset
            bw = wh[0, cy, cx].item() * self.img_size
            bh = wh[1, cy, cx].item() * self.img_size
            ox = offset[0, cy, cx].item()
            oy = offset[1, cy, cx].item()
            
            # Convert to image coordinates
            center_x = (cx + ox) * 4 - pad_left
            center_y = (cy + oy) * 4 - pad_top
            
            x1 = (center_x - bw / 2) / scale
            y1 = (center_y - bh / 2) / scale
            x2 = (center_x + bw / 2) / scale
            y2 = (center_y + bh / 2) / scale
            
            # Clip
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            if x2 > x1 + 10 and y2 > y1 + 10:
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'score': float(score),
                })
        
        # Sort by score and take top_k
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)[:top_k]
        
        return detections


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
        
        self.criterion = DetectionLoss(config.focal_alpha, config.focal_beta)
        self.scaler = torch.amp.GradScaler('cuda')
        
        self.epoch = 0
        self.best_loss = float('inf')
        self.history = {'train': [], 'val': []}
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
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
            
            # Always accumulate loss for reporting
            if torch.isfinite(loss):
                loss_val = loss.item()
                total_loss += loss_val
                n_batches += 1
                
                # Backward and step
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Step scheduler only if optimizer stepped
                if self.scaler.get_scale() >= old_scale:
                    self.scheduler.step()
                
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{loss_val:.2f}",
                    hm=f"{losses['hm'].item():.2f}",
                    lr=f"{lr:.1e}"
                )
            else:
                logger.warning(f"NaN/Inf loss at batch {n_batches}")
        
        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss
    
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
        logger.info(f"Training on {self.device}")
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for self.epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.history['train'].append(train_loss)
            self.history['val'].append(val_loss)
            
            logger.info(f"Epoch {self.epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}")
            
            # Save
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'history': self.history,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    
    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    
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
    model = MouaadNetUltraV2()
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config, args.save_dir)
    trainer.train()
    
    # Export
    if args.export:
        model.eval().cpu()
        torch.onnx.export(
            model,
            torch.randn(1, 3, 256, 256),
            f'{args.save_dir}/detection.onnx',
            input_names=['image'],
            output_names=['heatmap', 'wh', 'offset'],
            opset_version=11,
        )
        logger.info(f"Exported ONNX: {args.save_dir}/detection.onnx")


if __name__ == '__main__':
    main()
