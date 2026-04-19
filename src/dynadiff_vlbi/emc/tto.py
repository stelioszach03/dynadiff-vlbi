"""Test-time optimization utilities for real-data EMC adaptation."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from dynadiff_vlbi.physics.torch_fourier import fft2c_torch


class TestTimeOptimizer:
    """
    Adapt a visibility-conditioned model on support-only real-data measurements.

    The optimizer clones the original model so that the benchmark checkpoint remains unchanged.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_steps: int = 50,
        lr: float = 1.0e-4,
        lambda_support: float = 1.0,
    ) -> None:
        self.model = copy.deepcopy(model)
        self.n_steps = int(n_steps)
        self.lr = float(lr)
        self.lambda_support = float(lambda_support)
        self.device = next(self.model.parameters()).device

    @staticmethod
    def support_visibility_loss(
        prediction: torch.Tensor,
        support_vis: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_vis = fft2c_torch(prediction.squeeze(1))
        diff = (pred_vis - support_vis) * support_mask
        denom = support_mask.sum().clamp_min(1.0)
        return (diff.real.pow(2).sum() + diff.imag.pow(2).sum()) / denom

    def adapt(
        self,
        *,
        visibility_input: torch.Tensor,
        dirty_input: torch.Tensor,
        support_vis: torch.Tensor,
        support_mask: torch.Tensor,
        baseline_pairs: torch.Tensor | None = None,
        frame_uv_indices: torch.Tensor | None = None,
    ):
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable or self.n_steps <= 0:
            self.model.eval()
            with torch.no_grad():
                return self.model(
                    visibility_input=visibility_input,
                    dirty_input=dirty_input,
                    measurements=support_vis,
                    mask=support_mask,
                    baseline_pairs=baseline_pairs,
                    frame_uv_indices=frame_uv_indices,
                )

        optimizer = torch.optim.Adam(trainable, lr=self.lr)
        self.model.eval()
        for _ in range(self.n_steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(
                visibility_input=visibility_input,
                dirty_input=dirty_input,
                measurements=support_vis,
                mask=support_mask,
                baseline_pairs=baseline_pairs,
                frame_uv_indices=frame_uv_indices,
            )
            support_prediction = getattr(outputs, "pre_dc_prediction", outputs.mean)
            loss = self.lambda_support * self.support_visibility_loss(
                prediction=support_prediction,
                support_vis=support_vis,
                support_mask=support_mask,
            )
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            return self.model(
                visibility_input=visibility_input,
                dirty_input=dirty_input,
                measurements=support_vis,
                mask=support_mask,
                baseline_pairs=baseline_pairs,
                frame_uv_indices=frame_uv_indices,
            )
