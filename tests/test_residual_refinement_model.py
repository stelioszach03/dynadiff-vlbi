from __future__ import annotations

import torch

from dynadiff_vlbi.models.residual_visibility_refinement import ResidualVisibilityRefinementModel
from dynadiff_vlbi.utils.config import ModelConfig


def _build_residual_config(*, uncertainty_head: bool = True) -> ModelConfig:
    return ModelConfig(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        dropout=0.1,
        model_type="residual_visibility_refinement",
        include_dirty_input=True,
        visibility_representation="real_imag",
        include_uv_coords=True,
        include_mask_channel=True,
        uncertainty_head=uncertainty_head,
        refinement_channels=8,
        residual_scale=0.2,
        freeze_backbone=True,
    )


def test_residual_refinement_model_forward_shapes() -> None:
    model = ResidualVisibilityRefinementModel(_build_residual_config())
    visibility_input = torch.randn(2, 5, 4, 32, 32)
    dirty_input = torch.randn(2, 1, 4, 32, 32)

    outputs = model(visibility_input=visibility_input, dirty_input=dirty_input)

    assert outputs.mean.shape == (2, 1, 4, 32, 32)
    assert outputs.baseline_prediction.shape == (2, 1, 4, 32, 32)
    assert outputs.residual_correction.shape == (2, 1, 4, 32, 32)
    assert outputs.log_variance is not None
    assert outputs.log_variance.shape == (2, 1, 4, 32, 32)
    expected = torch.clamp(outputs.baseline_prediction + outputs.residual_correction, 0.0, 1.0)
    assert torch.allclose(outputs.mean, expected, atol=1e-6)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())


def test_residual_refinement_zero_branch_reduces_to_backbone_prediction() -> None:
    model = ResidualVisibilityRefinementModel(_build_residual_config(uncertainty_head=False))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith("backbone."):
                parameter.zero_()

    visibility_input = torch.randn(1, 5, 4, 32, 32)
    dirty_input = torch.randn(1, 1, 4, 32, 32)
    outputs = model(visibility_input=visibility_input, dirty_input=dirty_input)

    assert outputs.log_variance is None
    assert torch.allclose(outputs.residual_correction, torch.zeros_like(outputs.residual_correction), atol=1e-7)
    assert torch.allclose(outputs.mean, outputs.baseline_prediction, atol=1e-6)
