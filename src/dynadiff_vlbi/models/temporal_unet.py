"""Compact 3D U-Net for dynamic sequence reconstruction."""

from __future__ import annotations

import torch
from torch import nn

from dynadiff_vlbi.models.base import BaseReconstructionModel
from dynadiff_vlbi.utils.config import ModelConfig


class ConvBlock3D(nn.Module):
    """Small convolutional block with GroupNorm and optional dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = max(1, min(8, out_channels))
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalUNet3D(BaseReconstructionModel):
    """Compact temporal U-Net operating on `[B, 1, T, H, W]` inputs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        base = config.base_channels
        self.enc1 = ConvBlock3D(config.in_channels, base, dropout=0.0)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.enc2 = ConvBlock3D(base, base * 2, dropout=0.0)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.bottleneck = ConvBlock3D(base * 2, base * 4, dropout=config.dropout)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = ConvBlock3D(base * 4, base * 2, dropout=config.dropout)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlock3D(base * 2, base, dropout=config.dropout)
        self.out_proj = nn.Conv3d(base, config.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        residual = x
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        up2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        correction = self.out_proj(dec1)
        return torch.clamp(residual + correction, 0.0, 1.0)
