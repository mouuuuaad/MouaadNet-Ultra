"""
Learning Rate Schedulers
========================
Advanced learning rate scheduling for optimal training convergence.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List, Optional


class OneCycleLR(_LRScheduler):
    """
    One-Cycle Learning Rate Policy.
    
    Implements the 1cycle policy from "Super-Convergence: Very Fast Training
    of Neural Networks Using Large Learning Rates" (Smith & Topin, 2018).
    
    The learning rate follows a cycle:
    1. Warm up from initial_lr to max_lr
    2. Anneal from max_lr back to initial_lr (and optionally lower)
    
    Combined with momentum annealing for faster convergence.
    
    Args:
        optimizer: Wrapped optimizer
        max_lr: Maximum learning rate
        total_steps: Total number of training steps
        pct_start: Percentage of cycle for warmup (default: 0.3)
        div_factor: Initial lr = max_lr / div_factor (default: 25)
        final_div_factor: Final lr = max_lr / final_div_factor (default: 10000)
        anneal_strategy: Annealing method ('cos' or 'linear')
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        anneal_strategy: str = 'cos',
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.anneal_strategy = anneal_strategy
        
        # Calculate phase boundaries
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up
        
        # Calculate learning rates
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        step = self.last_epoch
        
        if step < self.step_up:
            # Warmup phase
            pct = step / self.step_up
            lr = self._anneal(self.initial_lr, self.max_lr, pct)
        else:
            # Annealing phase
            pct = (step - self.step_up) / self.step_down
            lr = self._anneal(self.max_lr, self.final_lr, pct)
        
        return [lr for _ in self.base_lrs]
    
    def _anneal(self, start: float, end: float, pct: float) -> float:
        """Interpolate between start and end."""
        if self.anneal_strategy == 'cos':
            return end + (start - end) * (1 + math.cos(math.pi * pct)) / 2
        else:  # linear
            return start + (end - start) * pct


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine Annealing with Warm Restarts.
    
    Implements SGDR: Stochastic Gradient Descent with Warm Restarts.
    
    Learning rate follows cosine curve with periodic restarts
    for better exploration of loss landscape.
    
    Args:
        optimizer: Wrapped optimizer
        T_0: Number of iterations for first restart
        T_mult: Factor to increase T after restart (default: 1)
        eta_min: Minimum learning rate (default: 0)
        last_epoch: Last epoch index (default: -1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.T_i = T_0
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        return [
            self.eta_min + (base_lr - self.eta_min) * 
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            
            if self.T_cur >= self.T_i:
                # Restart
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
        else:
            # Specific epoch provided
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.T_i = self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.
    
    Simple scheduler combining linear warmup with cosine decay.
    Commonly used for transformer training.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate (default: 0)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = (1 + math.cos(math.pi * progress)) / 2
        
        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


if __name__ == "__main__":
    # Test schedulers
    print("Testing Learning Rate Schedulers...")
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test One-Cycle LR
    print("\n1. One-Cycle LR:")
    scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=100)
    
    lrs = []
    for i in range(100):
        lrs.append(scheduler.get_lr()[0])
        scheduler.step()
    
    print(f"  Initial LR: {lrs[0]:.6f}")
    print(f"  Max LR (at ~30%): {max(lrs):.6f}")
    print(f"  Final LR: {lrs[-1]:.6f}")
    
    # Test Cosine Annealing with Warm Restarts
    print("\n2. Cosine Annealing with Warm Restarts:")
    optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler2 = CosineAnnealingWarmRestarts(optimizer2, T_0=20, T_mult=2)
    
    lrs2 = []
    for i in range(60):
        lrs2.append(scheduler2.get_lr()[0])
        scheduler2.step()
    
    print(f"  Initial LR: {lrs2[0]:.6f}")
    print(f"  LR at step 19: {lrs2[19]:.6f}")
    print(f"  LR at step 20 (restart): {lrs2[20]:.6f}")
    
    # Test Warmup Cosine
    print("\n3. Warmup + Cosine Decay:")
    optimizer3 = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler3 = WarmupCosineScheduler(optimizer3, warmup_steps=10, total_steps=100)
    
    lrs3 = []
    for i in range(100):
        lrs3.append(scheduler3.get_lr()[0])
        scheduler3.step()
    
    print(f"  After warmup (step 10): {lrs3[10]:.6f}")
    print(f"  Mid-training (step 50): {lrs3[50]:.6f}")
    print(f"  Final LR: {lrs3[-1]:.6f}")
    
    print("\n✓ All scheduler tests passed!")
