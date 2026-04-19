from __future__ import annotations

import numpy as np
import torch

from dynadiff_vlbi.emc.baselines.dps_baseline import DPSBaseline, DPSScoreUNet
from dynadiff_vlbi.evaluation.metrics import observed_visibility_rmse
from dynadiff_vlbi.physics.fourier_operator import fft2c


def test_dps_training_loss_is_finite() -> None:
    model = DPSScoreUNet(image_channels=1, base_channels=16)
    baseline = DPSBaseline(model, total_timesteps=32, ddim_steps=4)
    frames = torch.rand(4, 1, 16, 16)
    loss = baseline.training_loss(frames)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_dps_sampling_projects_back_to_support_measurements() -> None:
    rng = np.random.default_rng(7)
    model = DPSScoreUNet(image_channels=1, base_channels=16)
    baseline = DPSBaseline(model, total_timesteps=16, ddim_steps=4)
    target = rng.random((2, 16, 16), dtype=np.float32)
    support_mask = np.zeros_like(target, dtype=np.float32)
    support_mask[:, ::4, ::4] = 1.0
    support_vis = fft2c(target) * support_mask
    dirty = np.fft.ifft2(np.fft.ifftshift(support_vis, axes=(-2, -1)), norm="ortho")
    prediction = baseline.sample(support_vis=support_vis, support_mask=support_mask, dirty_recon=dirty.real.astype(np.float32))
    assert prediction.shape == target.shape
    assert np.isfinite(prediction).all()
    support_rmse = observed_visibility_rmse(prediction=prediction, measurements=support_vis, mask=support_mask)
    assert support_rmse < 1.0e-4
