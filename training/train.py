"""
MOUAADNET-ULTRA Training Script
================================
Complete training loop with mixed precision, gradient accumulation,
and multi-task learning.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mouaadnet_ultra.model import MouaadNetUltra
from mouaadnet_ultra.losses import MultiTaskLoss
from mouaadnet_ultra.optim import OneCycleLR


class Trainer:
    """
    Trainer for MOUAADNET-ULTRA.
    
    Features:
    - Mixed precision training (FP16)
    - Gradient accumulation
    - One-cycle learning rate
    - Multi-task loss balancing
    - Checkpoint saving
    - Logging and metrics
    
    Args:
        model: MOUAADNET-ULTRA model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
    ):
        self.config = config or self._default_config()
        
        # Device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Training on: {self.device}")
        
        # Model
        self.model = model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.config['epochs']
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['max_lr'],
            total_steps=total_steps,
            pct_start=0.3,
        )
        
        # Loss
        self.criterion = MultiTaskLoss(
            det_weight=self.config['det_loss_weight'],
            gender_weight=self.config['gender_loss_weight'],
            gender_pos_weight=self.config['gender_pos_weight'],
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.config['use_amp'] else None
        
        # State
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
    def _default_config(self) -> Dict:
        return {
            'epochs': 100,
            'learning_rate': 1e-4,
            'max_lr': 1e-3,
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'accumulation_steps': 1,
            'use_amp': True,
            'det_loss_weight': 1.0,
            'gender_loss_weight': 1.0,
            'gender_pos_weight': 5.0,
            'save_dir': './checkpoints',
            'log_interval': 50,
            'val_interval': 1,
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {}
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch['images'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config['use_amp']):
                outputs = self.model(images)
                losses = self.criterion(outputs, targets)
                loss = losses['total'] / self.config['accumulation_steps']
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                if self.scaler:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
            
            # Log
            if batch_idx % self.config['log_interval'] == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {self.epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {losses['total'].item():.4f} "
                    f"Det: {losses['loss_detection'].item():.4f} "
                    f"Gender: {losses['loss_gender'].item():.4f} "
                    f"LR: {lr:.6f}"
                )
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        val_losses = {}
        num_batches = len(self.val_loader)
        
        for batch in self.val_loader:
            images = batch['images'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            with autocast(enabled=self.config['use_amp']):
                outputs = self.model(images)
                losses = self.criterion(outputs, targets)
            
            for k, v in losses.items():
                if k not in val_losses:
                    val_losses[k] = 0.0
                val_losses[k] += v.item()
        
        for k in val_losses:
            val_losses[k] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint."""
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.model.config,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, save_dir / filename)
        
        if is_best:
            torch.save(checkpoint, save_dir / 'best.pt')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(self.epoch, self.config['epochs']):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            print(f"Epoch {epoch} Train - " + 
                  " ".join(f"{k}: {v:.4f}" for k, v in train_losses.items()))
            
            # Validate
            if (epoch + 1) % self.config['val_interval'] == 0:
                val_losses = self.validate()
                if val_losses:
                    print(f"Epoch {epoch} Val - " +
                          " ".join(f"{k}: {v:.4f}" for k, v in val_losses.items()))
                    
                    # Save best
                    if val_losses['total'] < self.best_loss:
                        self.best_loss = val_losses['total']
                        self.save_checkpoint('last.pt', is_best=True)
                        print(f"New best model saved! Loss: {self.best_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint('last.pt')
        
        print("Training complete!")


def create_dummy_dataloader(batch_size: int = 8, img_size: int = 416):
    """Create dummy dataloader for testing."""
    from torch.utils.data import Dataset
    
    class DummyDataset(Dataset):
        def __init__(self, size: int = 100, img_size: int = 416):
            self.size = size
            self.img_size = img_size
            self.h, self.w = img_size // 8, img_size // 8  # Heatmap size
            
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            images = torch.randn(3, self.img_size, self.img_size)
            
            targets = {
                'heatmaps': [
                    torch.zeros(1, self.h, self.w),
                    torch.zeros(1, self.h // 2, self.w // 2),
                    torch.zeros(1, self.h // 4, self.w // 4),
                ],
                'sizes': [
                    torch.zeros(2, self.h, self.w),
                    torch.zeros(2, self.h // 2, self.w // 2),
                    torch.zeros(2, self.h // 4, self.w // 4),
                ],
                'offsets': [
                    torch.zeros(2, self.h, self.w),
                    torch.zeros(2, self.h // 2, self.w // 2),
                    torch.zeros(2, self.h // 4, self.w // 4),
                ],
                'gender_labels': torch.randint(0, 2, (1,)).float(),
            }
            
            # Add random positive sample
            y, x = torch.randint(0, self.h, (1,)).item(), torch.randint(0, self.w, (1,)).item()
            targets['heatmaps'][0][0, y, x] = 1.0
            
            return {'images': images, 'targets': targets}
    
    def collate_fn(batch):
        images = torch.stack([b['images'] for b in batch])
        targets = {
            'heatmaps': [torch.stack([b['targets']['heatmaps'][i] for b in batch]) for i in range(3)],
            'sizes': [torch.stack([b['targets']['sizes'][i] for b in batch]) for i in range(3)],
            'offsets': [torch.stack([b['targets']['offsets'][i] for b in batch]) for i in range(3)],
            'gender_labels': torch.stack([b['targets']['gender_labels'] for b in batch]),
        }
        return {'images': images, 'targets': targets}
    
    dataset = DummyDataset(size=100, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


if __name__ == "__main__":
    print("MOUAADNET-ULTRA Training")
    print("=" * 50)
    
    # Create model
    model = MouaadNetUltra()
    
    # Create dummy dataloaders for testing
    train_loader = create_dummy_dataloader(batch_size=4)
    val_loader = create_dummy_dataloader(batch_size=4)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'epochs': 2,  # Quick test
            'learning_rate': 1e-4,
            'max_lr': 1e-3,
            'use_amp': torch.cuda.is_available(),
            'log_interval': 10,
            'save_dir': './checkpoints',
        }
    )
    
    # Train
    trainer.train()
    
    print("\n✓ Training test complete!")
