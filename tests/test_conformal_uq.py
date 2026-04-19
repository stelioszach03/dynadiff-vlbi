from __future__ import annotations

import numpy as np
import torch

from dynadiff_vlbi.emc.uq import VLBIConformalUQ


def test_conformal_uq_calibrates_and_reports_finite_metrics() -> None:
    uq = VLBIConformalUQ(alpha=0.1)
    prediction = torch.ones(2, 8, 8, dtype=torch.float32)
    target_vis = torch.zeros(2, 8, 8, dtype=torch.complex64)
    target_mask = torch.ones(2, 8, 8, dtype=torch.float32)
    uq.calibrate(
        predictions=prediction,
        support_vis=torch.zeros_like(target_vis),
        support_mask=torch.zeros_like(target_mask),
        target_vis=target_vis,
        target_mask=target_mask,
    )
    q_hat = uq.q_hat()
    assert np.isfinite(q_hat)
    report = uq.coverage_width_report(
        predictions=np.ones((2, 8, 8), dtype=np.float32),
        ground_truth=np.ones((2, 8, 8), dtype=np.float32),
    )
    assert report["coverage"] == 1.0
    assert report["mean_interval_width"] >= 0.0
