"""Split conformal uncertainty helpers for VLBI held-out evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from dynadiff_vlbi.physics.torch_fourier import fft2c_torch


class VLBIConformalUQ:
    """Scalar split-conformal intervals calibrated from held-out target residuals."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.calibration_scores: list[float] = []

    @staticmethod
    def _target_residual_score(
        prediction: torch.Tensor,
        target_vis: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_vis = fft2c_torch(prediction)
        residual = torch.abs((pred_vis - target_vis) * target_mask)
        denom = target_mask.sum(dim=(-2, -1)).clamp_min(1.0)
        return residual.sum(dim=(-2, -1)) / denom

    def calibrate(
        self,
        predictions: torch.Tensor,
        support_vis: torch.Tensor,
        support_mask: torch.Tensor,
        target_vis: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> None:
        del support_vis, support_mask
        if predictions.ndim != 3:
            raise ValueError("Predictions must have shape [T, H, W].")
        scores = self._target_residual_score(
            prediction=predictions,
            target_vis=target_vis,
            target_mask=target_mask,
        )
        self.calibration_scores.extend(float(value) for value in scores.detach().cpu().reshape(-1).tolist())

    def q_hat(self) -> float:
        if not self.calibration_scores:
            raise ValueError("Conformal calibration scores are empty.")
        scores = np.asarray(self.calibration_scores, dtype=np.float64)
        n = int(scores.size)
        quantile_level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / max(n, 1))
        return float(np.quantile(scores, quantile_level, method="higher"))

    def predict_interval(self, prediction: np.ndarray, support_vis: np.ndarray, support_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del support_vis, support_mask
        q_hat = np.float32(self.q_hat())
        prediction_array = np.asarray(prediction, dtype=np.float32)
        return prediction_array - q_hat, prediction_array + q_hat

    def interval_width(self) -> float:
        return float(2.0 * self.q_hat())

    def coverage_width_report(self, predictions: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
        lower, upper = self.predict_interval(
            prediction=predictions,
            support_vis=np.zeros_like(predictions, dtype=np.complex64),
            support_mask=np.zeros_like(predictions, dtype=np.float32),
        )
        ground_truth_array = np.asarray(ground_truth, dtype=np.float32)
        coverage = np.mean((ground_truth_array >= lower) & (ground_truth_array <= upper))
        return {
            "coverage": float(coverage),
            "mean_interval_width": float(np.mean(upper - lower)),
            "q_hat": float(self.q_hat()),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "q_hat": self.q_hat(),
            "calibration_count": len(self.calibration_scores),
        }
