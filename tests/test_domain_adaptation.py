from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dynadiff_vlbi.data.feature_formatting import build_temporal_uv_grid
from dynadiff_vlbi.emc.domain_bridge import EHTDomainBridge
from dynadiff_vlbi.emc.inference import predict_real_phase2
from dynadiff_vlbi.physics.fourier_operator import fft2c
from dynadiff_vlbi.utils.config import ModelConfig


def test_domain_bridge_normalizes_support_to_unit_median() -> None:
    bridge = EHTDomainBridge()
    support_vis = np.asarray(
        [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [3.0 + 4.0j, 1.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    support_mask = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    normalized, scale = bridge.normalize_support(support_vis, support_mask)

    amplitudes = np.abs(normalized[support_mask > 0.0])
    assert np.isclose(np.median(amplitudes), 1.0)
    denormalized = bridge.denormalize_prediction(np.ones((2, 2), dtype=np.float32), scale)
    assert np.allclose(denormalized, np.full((2, 2), scale, dtype=np.float32))


@dataclass
class DummyPhase2Output:
    mean: torch.Tensor
    pre_dc_prediction: torch.Tensor


class DummyPhase2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(
        self,
        visibility_input: torch.Tensor,
        dirty_input: torch.Tensor | None = None,
        measurements: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        baseline_pairs: torch.Tensor | None = None,
        frame_uv_indices: torch.Tensor | None = None,
    ) -> DummyPhase2Output:
        assert dirty_input is not None
        prediction = dirty_input * self.scale
        return DummyPhase2Output(mean=prediction, pre_dc_prediction=prediction)


def test_predict_real_phase2_domain_adaptation_reduces_support_loss() -> None:
    sequence_length = 2
    image_size = 8
    target_image = np.zeros((sequence_length, image_size, image_size), dtype=np.float32)
    target_image[:, 2:6, 2:6] = 1.0
    support_measurements = fft2c(target_image).astype(np.complex64)
    support_mask = np.ones_like(target_image, dtype=np.float32)
    dirty = target_image.copy()

    model = DummyPhase2Model()
    model_config = ModelConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        dropout=0.0,
        model_type="emc_ccrr",
        include_dirty_input=True,
        visibility_representation="real_imag",
        include_uv_coords=True,
        include_mask_channel=True,
        include_observation_metadata=False,
        uncertainty_head=False,
        refinement_channels=4,
        residual_scale=0.2,
        freeze_backbone=False,
        dc_enabled=False,
        dc_weight=1.0,
        dc_learnable=False,
    )

    prediction = predict_real_phase2(
        model=model,
        model_config=model_config,
        support_vis_real=support_measurements.real.astype(np.float32),
        support_vis_imag=support_measurements.imag.astype(np.float32),
        support_mask=support_mask,
        support_dirty=dirty,
        uv_coords=build_temporal_uv_grid(image_size=image_size, sequence_length=sequence_length).astype(np.float32),
        frame_uv_coords=np.zeros((sequence_length, 1, 2), dtype=np.float32),
        frame_uv_indices=np.zeros((sequence_length, 1, 2), dtype=np.int64),
        measurements=support_measurements,
        device=torch.device("cpu"),
        use_domain_adaptation=True,
        tto_steps=25,
        tto_lr=5.0e-2,
    )

    assert prediction.mean.shape == target_image.shape
    assert prediction.support_loss_before is not None
    assert prediction.support_loss_after is not None
    assert prediction.support_loss_after < prediction.support_loss_before
