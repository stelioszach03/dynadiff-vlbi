"""Closure-consistent residual refinement model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dynadiff_vlbi.models.base import BaseReconstructionModel
from dynadiff_vlbi.models.temporal_unet import ConvBlock3D, TemporalUNet3D
from dynadiff_vlbi.models.visibility_conditioned import visibility_input_channels
from dynadiff_vlbi.physics.torch_fourier import fft2c_torch, ifft2c_torch
from dynadiff_vlbi.utils.config import ModelConfig


@dataclass
class CCRROutput:
    """Structured outputs for closure-consistent residual refinement."""

    mean: torch.Tensor
    log_variance: torch.Tensor | None
    baseline_prediction: torch.Tensor
    residual_correction: torch.Tensor
    pre_dc_prediction: torch.Tensor
    dc_weight: torch.Tensor


class DifferentiableDataConsistencyLayer(nn.Module):
    """Blend predicted and observed visibilities at sampled coefficients."""

    def __init__(self, enabled: bool, dc_weight: float, learnable: bool) -> None:
        super().__init__()
        self.enabled = enabled
        self.learnable = learnable
        dc_weight = float(max(0.0, min(1.0, dc_weight)))
        if learnable:
            init = torch.logit(torch.tensor(dc_weight).clamp(1e-4, 1.0 - 1e-4))
            self.dc_logit = nn.Parameter(init.clone().detach())
        else:
            self.register_buffer("dc_value", torch.tensor(dc_weight, dtype=torch.float32))

    def weight(self) -> torch.Tensor:
        if self.learnable:
            return torch.sigmoid(self.dc_logit)
        return self.dc_value

    def forward(
        self,
        prediction: torch.Tensor,
        measurements: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight()
        if (not self.enabled) or measurements is None or mask is None:
            return prediction, weight

        prediction_vis = fft2c_torch(prediction.squeeze(1))
        corrected_vis = prediction_vis + weight * mask * (measurements - prediction_vis)
        corrected = ifft2c_torch(corrected_vis).real.unsqueeze(1)
        return corrected, weight


class ClosureConsistentResidualRefinementModel(BaseReconstructionModel):
    """Residual refinement with an in-model differentiable data-consistency layer."""

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
        self.data_consistency = DifferentiableDataConsistencyLayer(
            enabled=config.dc_enabled,
            dc_weight=config.dc_weight,
            learnable=config.dc_learnable,
        )

        if self.freeze_backbone:
            self._freeze_backbone_parameters()

    def _freeze_backbone_parameters(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True) -> ClosureConsistentResidualRefinementModel:
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
    ) -> CCRROutput:
        if dirty_input is None:
            raise ValueError("dirty_input is required for CCRR.")

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
        pre_dc_prediction = torch.clamp(baseline_prediction + residual_correction, 0.0, 1.0)
        mean, dc_weight = self.data_consistency(
            prediction=pre_dc_prediction,
            measurements=measurements,
            mask=mask,
        )

        log_variance = None
        if self.log_variance_head is not None:
            log_variance = torch.clamp(self.log_variance_head(dec1), min=-6.0, max=2.0)

        return CCRROutput(
            mean=mean,
            log_variance=log_variance,
            baseline_prediction=baseline_prediction,
            residual_correction=residual_correction,
            pre_dc_prediction=pre_dc_prediction,
            dc_weight=dc_weight.detach().reshape(1),
        )
