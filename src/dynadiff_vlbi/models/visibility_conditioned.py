"""Compact visibility-conditioned spatiotemporal reconstruction model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dynadiff_vlbi.models.base import BaseReconstructionModel
from dynadiff_vlbi.models.temporal_unet import ConvBlock3D
from dynadiff_vlbi.utils.config import ModelConfig


@dataclass
class VisibilityConditionedOutput:
    """Structured output for visibility-conditioned reconstructions."""

    mean: torch.Tensor
    log_variance: torch.Tensor | None


def visibility_input_channels(config: ModelConfig) -> int:
    """Infer the expected number of visibility input channels from the model config."""

    channels = 2
    if config.include_mask_channel:
        channels += 1
    if config.include_uv_coords:
        channels += 2
    if config.include_observation_metadata:
        channels += 4
    return channels


class VisibilityConditionedReconstructor(BaseReconstructionModel):
    """Dual-branch visibility-conditioned model with optional heteroscedastic head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base_channels
        self.include_dirty_input = config.include_dirty_input
        self.uncertainty_head = config.uncertainty_head

        self.visibility_stem = ConvBlock3D(visibility_input_channels(config), base, dropout=0.0)
        if self.include_dirty_input:
            self.dirty_stem = ConvBlock3D(1, base, dropout=0.0)
            fused_channels = base * 2
        else:
            self.dirty_stem = None
            fused_channels = base

        self.fusion = ConvBlock3D(fused_channels, base * 2, dropout=0.0)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.enc2 = ConvBlock3D(base * 2, base * 4, dropout=0.0)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.bottleneck = ConvBlock3D(base * 4, base * 8, dropout=config.dropout)
        self.up2 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = ConvBlock3D(base * 8, base * 4, dropout=config.dropout)
        self.up1 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlock3D(base * 4, base * 2, dropout=config.dropout)
        self.mean_head = nn.Conv3d(base * 2, config.out_channels, kernel_size=1)
        self.log_variance_head = (
            nn.Conv3d(base * 2, config.out_channels, kernel_size=1) if self.uncertainty_head else None
        )

    def forward(
        self,
        visibility_input: torch.Tensor,
        dirty_input: torch.Tensor | None = None,
        measurements: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        baseline_pairs: torch.Tensor | None = None,
        frame_uv_indices: torch.Tensor | None = None,
    ) -> VisibilityConditionedOutput:
        visibility_features = self.visibility_stem(visibility_input)
        if self.include_dirty_input:
            if dirty_input is None:
                raise ValueError("dirty_input is required when include_dirty_input=True.")
            dirty_features = self.dirty_stem(dirty_input)
            fused = self.fusion(torch.cat([visibility_features, dirty_features], dim=1))
        else:
            fused = self.fusion(visibility_features)

        enc2 = self.enc2(self.pool1(fused))
        bottleneck = self.bottleneck(self.pool2(enc2))
        up2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, fused], dim=1))

        mean_update = self.mean_head(dec1)
        if dirty_input is not None:
            mean = torch.clamp(dirty_input + mean_update, 0.0, 1.0)
        else:
            mean = torch.sigmoid(mean_update)

        log_variance = None
        if self.log_variance_head is not None:
            log_variance = torch.clamp(self.log_variance_head(dec1), min=-6.0, max=2.0)
        return VisibilityConditionedOutput(mean=mean, log_variance=log_variance)
