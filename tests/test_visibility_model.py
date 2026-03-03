from __future__ import annotations

import torch

from dynadiff_vlbi.models.visibility_conditioned import VisibilityConditionedReconstructor
from dynadiff_vlbi.utils.config import ModelConfig


def test_visibility_conditioned_model_forward_shapes() -> None:
    model = VisibilityConditionedReconstructor(
        ModelConfig(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            dropout=0.1,
            model_type="visibility_conditioned",
            include_dirty_input=True,
            visibility_representation="real_imag",
            include_uv_coords=True,
            include_mask_channel=True,
            uncertainty_head=True,
        )
    )
    visibility_input = torch.randn(2, 5, 4, 32, 32)
    dirty_input = torch.randn(2, 1, 4, 32, 32)

    outputs = model(visibility_input=visibility_input, dirty_input=dirty_input)

    assert outputs.mean.shape == (2, 1, 4, 32, 32)
    assert outputs.log_variance is not None
    assert outputs.log_variance.shape == (2, 1, 4, 32, 32)
