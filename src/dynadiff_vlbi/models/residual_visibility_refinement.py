"""Residual visibility-guided refinement model built on top of the baseline 3D U-Net."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dynadiff_vlbi.models.base import BaseReconstructionModel
from dynadiff_vlbi.models.temporal_unet import ConvBlock3D, TemporalUNet3D
from dynadiff_vlbi.models.visibility_conditioned import visibility_input_channels
from dynadiff_vlbi.utils.config import ModelConfig


@dataclass
class ResidualVisibilityRefinementOutput:
    """Structured outputs for the residual visibility refinement path."""

    mean: torch.Tensor
    log_variance: torch.Tensor | None
    baseline_prediction: torch.Tensor
    residual_correction: torch.Tensor


class ResidualVisibilityRefinementModel(BaseReconstructionModel):
    """Use a baseline backbone prediction plus a compact visibility-guided residual branch."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.freeze_backbone = config.freeze_backbone
        self.include_dirty_context = config.include_dirty_input
        self.uncertainty_head = config.uncertainty_head
        refine = config.refinement_channels

        self.backbone = TemporalUNet3D(config)

        self.visibility_stem = ConvBlock3D(visibility_input_channels(config), refine, dropout=0.0)
        self.baseline_stem = ConvBlock3D(1, refine, dropout=0.0)
        context_channels = refine * 2
        if self.include_dirty_context:
            self.dirty_stem = ConvBlock3D(1, refine, dropout=0.0)
            context_channels += refine
        else:
            self.dirty_stem = None

        self.fusion = ConvBlock3D(context_channels, refine * 2, dropout=0.0)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.enc2 = ConvBlock3D(refine * 2, refine * 4, dropout=0.0)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.bottleneck = ConvBlock3D(refine * 4, refine * 8, dropout=config.dropout)
        self.up2 = nn.ConvTranspose3d(refine * 8, refine * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = ConvBlock3D(refine * 8, refine * 4, dropout=config.dropout)
        self.up1 = nn.ConvTranspose3d(refine * 4, refine * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlock3D(refine * 4, refine * 2, dropout=config.dropout)
        self.mean_head = nn.Conv3d(refine * 2, config.out_channels, kernel_size=1)
        self.log_variance_head = (
            nn.Conv3d(refine * 2, config.out_channels, kernel_size=1) if self.uncertainty_head else None
        )

        if self.freeze_backbone:
            self._freeze_backbone_parameters()

    def _freeze_backbone_parameters(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True) -> ResidualVisibilityRefinementModel:
        """Keep the backbone in eval mode when it is frozen."""

        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        visibility_input: torch.Tensor,
        dirty_input: torch.Tensor | None = None,
        measurements: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        baseline_pairs: torch.Tensor | None = None,
        frame_uv_indices: torch.Tensor | None = None,
    ) -> ResidualVisibilityRefinementOutput:
        if dirty_input is None:
            raise ValueError("dirty_input is required for residual visibility refinement.")

        if self.freeze_backbone:
            with torch.no_grad():
                baseline_prediction = self.backbone(dirty_input)
        else:
            baseline_prediction = self.backbone(dirty_input)

        features = [
            self.visibility_stem(visibility_input),
            self.baseline_stem(baseline_prediction),
        ]
        if self.include_dirty_context:
            features.append(self.dirty_stem(dirty_input))

        fused = self.fusion(torch.cat(features, dim=1))
        enc2 = self.enc2(self.pool1(fused))
        bottleneck = self.bottleneck(self.pool2(enc2))
        up2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, fused], dim=1))

        residual_correction = torch.tanh(self.mean_head(dec1)) * self.config.residual_scale
        mean = torch.clamp(baseline_prediction + residual_correction, 0.0, 1.0)

        log_variance = None
        if self.log_variance_head is not None:
            log_variance = torch.clamp(self.log_variance_head(dec1), min=-6.0, max=2.0)

        return ResidualVisibilityRefinementOutput(
            mean=mean,
            log_variance=log_variance,
            baseline_prediction=baseline_prediction,
            residual_correction=residual_correction,
        )
