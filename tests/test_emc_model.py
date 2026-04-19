from __future__ import annotations

import torch

from dynadiff_vlbi.models.emc_ccrr import EarnedMeasurementConsistencyModel
from dynadiff_vlbi.utils.config import ModelConfig


def _build_emc_config() -> ModelConfig:
    return ModelConfig(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        dropout=0.1,
        model_type="emc_ccrr",
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


def test_emc_model_forward_shapes_match_ccrr_contract() -> None:
    model = EarnedMeasurementConsistencyModel(_build_emc_config())
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
    assert outputs.pre_dc_prediction.shape == (2, 1, 4, 32, 32)
    assert outputs.log_variance is not None
    assert outputs.log_variance.shape == (2, 1, 4, 32, 32)
