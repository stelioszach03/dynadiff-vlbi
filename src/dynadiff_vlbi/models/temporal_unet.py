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
    """Compact temporal U-Net operating on `[B, 1, T, H, W]` inputs.

    Supports variable depth via ``config.num_levels`` (default 2 for backward
    compatibility with 32x32).  For 128x128 images use ``num_levels=3``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        base = config.base_channels
        num_levels = getattr(config, "num_levels", 2)

        # Build encoder / decoder lists
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        in_ch = config.in_channels
        ch = base
        for level in range(num_levels):
            drop = 0.0 if level < num_levels - 1 else config.dropout
            self.encoders.append(ConvBlock3D(in_ch, ch, dropout=0.0))
            self.pools.append(nn.MaxPool3d(kernel_size=(1, 2, 2)))
            in_ch = ch
            ch = ch * 2

        # Bottleneck
        self.bottleneck = ConvBlock3D(in_ch, ch, dropout=config.dropout)

        # Decoder path (reverse order)
        for level in reversed(range(num_levels)):
            dec_in = ch
            dec_out = ch // 2
            self.ups.append(
                nn.ConvTranspose3d(dec_in, dec_out, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )
            # After concat with skip: dec_out + encoder_channels
            enc_ch = base * (2 ** level) if level > 0 else base
            drop = config.dropout if level > 0 else config.dropout
            self.decoders.append(ConvBlock3D(dec_out + enc_ch, dec_out, dropout=drop))
            ch = dec_out

        self.out_proj = nn.Conv3d(base, config.out_channels, kernel_size=1)
        self.num_levels = num_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        residual = x

        # Encoder
        skips = []
        h = x
        for encoder, pool in zip(self.encoders, self.pools):
            h = encoder(h)
            skips.append(h)
            h = pool(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h)
            h = decoder(torch.cat([h, skip], dim=1))

        correction = self.out_proj(h)
        return torch.clamp(residual + correction, 0.0, 1.0)
