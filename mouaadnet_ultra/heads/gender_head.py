"""
Gender Classification Head
==========================
Lightweight classification head with spatial attention for
focusing on gender-relevant features (face, body proportions).

Uses Global Average Pooling for efficient feature compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Learns to focus on relevant spatial regions (e.g., face, upper body)
    while suppressing irrelevant background areas.
    
    Uses both max and average pooling across channels to generate
    spatial attention weights.
    
    Args:
        kernel_size: Convolution kernel size for attention (default: 7)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.
        
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Attention-weighted features (B, C, H, W)
        """
        # Pool across channels
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate and generate attention
        pool_cat = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.conv(pool_cat)
        
        # Apply attention
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation style).
    
    Re-calibrates channel-wise feature responses by modeling
    inter-channel dependencies.
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        reduced = max(channels // reduction, 8)
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        attention = self.attention(x).view(b, c, 1, 1)
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Combines channel and spatial attention sequentially.
    
    Args:
        channels: Number of input channels
        reduction: Channel attention reduction ratio
        spatial_kernel: Spatial attention kernel size
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class GenderHead(nn.Module):
    """
    Gender Classification Head.
    
    Architecture:
    1. Conv layers for feature refinement
    2. CBAM attention for focusing on relevant regions
    3. Global Average Pooling for feature compression
    4. FC layers for binary classification
    
    Args:
        in_channels: Number of input feature channels
        hidden_dim: Hidden dimension for classifier (default: 128)
        dropout: Dropout rate (default: 0.2)
        use_attention: Whether to use CBAM attention (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        
        # Attention module
        if use_attention:
            self.attention = CBAM(hidden_dim, reduction=8)
        else:
            self.attention = nn.Identity()
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),  # Binary: Male (0) / Female (1)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for gender classification.
        
        Args:
            x: Input features (B, C, H, W) - typically from detected person region
            return_features: Whether to also return intermediate features
            
        Returns:
            Gender logits (B, 1) - apply sigmoid for probability
        """
        # Feature refinement
        feat = self.refine(x)
        
        # Attention
        feat = self.attention(feat)
        
        # Global pooling
        feat_pooled = self.gap(feat)
        
        # Classification
        logits = self.classifier(feat_pooled)
        
        if return_features:
            return logits, feat_pooled.flatten(1)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability of female class."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get binary prediction (0=Male, 1=Female)."""
        proba = self.predict_proba(x)
        return (proba > 0.5).long()


class MultiScaleGenderHead(nn.Module):
    """
    Multi-scale Gender Classification Head.
    
    Aggregates features from multiple scales before classification
    for more robust predictions.
    
    Args:
        in_channels: Number of input channels per scale
        num_scales: Number of input scales
        hidden_dim: Hidden dimension for classifier
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        num_scales: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        # Per-scale feature extractors
        self.scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU6(inplace=True),
            )
            for _ in range(num_scales)
        ])
        
        # Attention weights for scale fusion
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, num_scales),
            nn.Softmax(dim=1),
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Multi-scale gender classification.
        
        Args:
            features: Tuple of feature maps from different scales
            
        Returns:
            Gender logits (B, 1)
        """
        # Extract per-scale features
        scale_feats = []
        for i, feat in enumerate(features):
            scale_feats.append(self.scale_extractors[i](feat))
        
        # Stack: (B, num_scales, hidden_dim)
        stacked = torch.stack(scale_feats, dim=1)
        
        # Compute attention weights
        concat_feats = torch.cat(scale_feats, dim=1)
        attn_weights = self.scale_attention(concat_feats)
        
        # Weighted sum: (B, hidden_dim)
        fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        # Classify
        return self.classifier(fused)


if __name__ == "__main__":
    # Test Gender Head
    print("Testing Gender Head...")
    
    # Single scale input (cropped person region)
    x = torch.randn(4, 64, 16, 16)
    
    # Test basic GenderHead
    head = GenderHead(64, hidden_dim=128)
    logits = head(x)
    
    print(f"Input: {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Probabilities: {head.predict_proba(x).squeeze().tolist()[:2]}")
    print(f"Predictions: {head.predict(x).squeeze().tolist()}")
    
    # Test with feature return
    logits, features = head(x, return_features=True)
    print(f"Features: {features.shape}")
    
    # Test MultiScaleGenderHead
    multi_head = MultiScaleGenderHead(64, num_scales=3)
    multi_features = (
        torch.randn(4, 64, 52, 52),
        torch.randn(4, 64, 26, 26),
        torch.randn(4, 64, 13, 13),
    )
    multi_logits = multi_head(multi_features)
    print(f"\nMulti-scale logits: {multi_logits.shape}")
    
    # Test attention modules
    print("\nTesting attention modules...")
    spatial = SpatialAttention(7)
    channel = ChannelAttention(64, 16)
    cbam = CBAM(64)
    
    print(f"Spatial attention: {spatial(x).shape}")
    print(f"Channel attention: {channel(x).shape}")
    print(f"CBAM: {cbam(x).shape}")
    
    # Count parameters
    params = sum(p.numel() for p in head.parameters())
    print(f"\nGenderHead parameters: {params:,}")
    
    print("✓ All Gender Head tests passed!")
