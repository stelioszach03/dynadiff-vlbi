from __future__ import annotations

import torch

from dynadiff_vlbi.models.ccrr import ClosureConsistentResidualRefinementModel, DifferentiableDataConsistencyLayer
from dynadiff_vlbi.physics.torch_fourier import fft2c_torch
from dynadiff_vlbi.physics.sampling import conjugate_index
from dynadiff_vlbi.utils.config import ModelConfig


def _build_ccrr_config() -> ModelConfig:
    return ModelConfig(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        dropout=0.1,
        model_type="ccrr",
        include_dirty_input=True,
        visibility_representation="real_imag",
        include_uv_coords=True,
        include_mask_channel=True,
        include_observation_metadata=True,
        uncertainty_head=True,
        refinement_channels=8,
        residual_scale=0.2,
        freeze_backbone=True,
        dc_enabled=True,
        dc_weight=1.0,
        dc_learnable=False,
    )


def test_differentiable_data_consistency_layer_enforces_observed_coefficients() -> None:
    layer = DifferentiableDataConsistencyLayer(enabled=True, dc_weight=1.0, learnable=False)
    prediction = torch.rand(1, 1, 3, 16, 16)
    mask = torch.zeros(1, 3, 16, 16, dtype=torch.float32)
    mask[:, :, 4, 5] = 1.0
    mask[:, :, conjugate_index(4, 16), conjugate_index(5, 16)] = 1.0
    measurements = fft2c_torch(prediction.squeeze(1))
    perturbation = torch.tensor(0.3 + 0.2j, dtype=torch.complex64)
    measurements[:, :, 4, 5] = measurements[:, :, 4, 5] + perturbation
    measurements[:, :, conjugate_index(4, 16), conjugate_index(5, 16)] = torch.conj(measurements[:, :, 4, 5])

    corrected, weight = layer(prediction=prediction, measurements=measurements, mask=mask)
    corrected_vis = fft2c_torch(corrected.squeeze(1))

    assert torch.allclose(weight, torch.tensor(1.0))
    assert torch.allclose(corrected_vis[:, :, 4, 5], measurements[:, :, 4, 5], atol=1e-4)


def test_ccrr_model_forward_shapes_and_outputs() -> None:
    model = ClosureConsistentResidualRefinementModel(_build_ccrr_config())
    visibility_input = torch.randn(2, 9, 4, 32, 32)
    dirty_input = torch.randn(2, 1, 4, 32, 32)
    measurements = torch.randn(2, 4, 32, 32, dtype=torch.complex64)
    mask = (torch.rand(2, 4, 32, 32) > 0.9).to(torch.float32)

    outputs = model(
        visibility_input=visibility_input,
        dirty_input=dirty_input,
        measurements=measurements,
        mask=mask,
    )

    assert outputs.mean.shape == (2, 1, 4, 32, 32)
    assert outputs.baseline_prediction.shape == (2, 1, 4, 32, 32)
    assert outputs.residual_correction.shape == (2, 1, 4, 32, 32)
    assert outputs.pre_dc_prediction.shape == (2, 1, 4, 32, 32)
    assert outputs.log_variance is not None
    assert outputs.log_variance.shape == (2, 1, 4, 32, 32)
    assert outputs.dc_weight.shape == (1,)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
