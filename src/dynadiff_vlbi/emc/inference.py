"""Real-data EMC inference wrappers with optional domain adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dynadiff_vlbi.data.feature_formatting import format_dirty_input, format_visibility_tensor
from dynadiff_vlbi.emc.domain_bridge import EHTDomainBridge
from dynadiff_vlbi.emc.tto import TestTimeOptimizer
from dynadiff_vlbi.utils.config import ModelConfig


@dataclass(frozen=True)
class RealPhase2Prediction:
    mean: np.ndarray
    pre_dc_prediction: np.ndarray | None
    adaptation_scale: float
    support_loss_before: float | None
    support_loss_after: float | None


def _build_phase2_tensors(
    *,
    support_vis_real: np.ndarray,
    support_vis_imag: np.ndarray,
    support_mask: np.ndarray,
    support_dirty: np.ndarray,
    uv_coords: np.ndarray,
    frame_uv_coords: np.ndarray | None,
    frame_uv_indices: np.ndarray | None,
    measurements: np.ndarray,
    device: torch.device,
    model_config: ModelConfig,
) -> dict[str, torch.Tensor | None]:
    visibility_input_array = format_visibility_tensor(
        vis_real=support_vis_real,
        vis_imag=support_vis_imag,
        mask=support_mask,
        representation=model_config.visibility_representation,
        include_mask_channel=model_config.include_mask_channel,
        include_uv_coords=model_config.include_uv_coords,
        uv_coords=uv_coords,
        include_observation_metadata=model_config.include_observation_metadata,
        frame_uv_coords=frame_uv_coords,
        frame_uv_indices=frame_uv_indices,
    )
    visibility_input = torch.from_numpy(visibility_input_array).unsqueeze(0).to(device)
    dirty_input = torch.from_numpy(format_dirty_input(support_dirty)).unsqueeze(0).to(device)
    model_measurements = torch.from_numpy(measurements.astype(np.complex64)).unsqueeze(0).to(device)
    model_mask = torch.from_numpy(support_mask.astype(np.float32)).unsqueeze(0).to(device)
    baseline_pairs_tensor = None
    frame_uv_indices_tensor = None
    if frame_uv_indices is not None:
        frame_uv_indices_tensor = torch.from_numpy(frame_uv_indices.astype(np.int64)).unsqueeze(0).to(device)
    return {
        "visibility_input": visibility_input,
        "dirty_input": dirty_input,
        "measurements": model_measurements,
        "mask": model_mask,
        "baseline_pairs": baseline_pairs_tensor,
        "frame_uv_indices": frame_uv_indices_tensor,
    }


def _support_loss_from_outputs(outputs, support_vis: torch.Tensor, support_mask: torch.Tensor) -> float:
    prediction = getattr(outputs, "pre_dc_prediction", outputs.mean)
    loss = TestTimeOptimizer.support_visibility_loss(
        prediction=prediction,
        support_vis=support_vis,
        support_mask=support_mask,
    )
    return float(loss.detach().item())


def predict_real_phase2(
    *,
    model: torch.nn.Module,
    model_config: ModelConfig,
    support_vis_real: np.ndarray,
    support_vis_imag: np.ndarray,
    support_mask: np.ndarray,
    support_dirty: np.ndarray,
    uv_coords: np.ndarray,
    frame_uv_coords: np.ndarray | None,
    frame_uv_indices: np.ndarray | None,
    measurements: np.ndarray,
    device: torch.device,
    use_domain_adaptation: bool = False,
    domain_bridge: EHTDomainBridge | None = None,
    tto_steps: int = 50,
    tto_lr: float = 1.0e-4,
    tto_lambda_support: float = 1.0,
) -> RealPhase2Prediction:
    support_complex = (support_vis_real + 1j * support_vis_imag).astype(np.complex64)

    if not use_domain_adaptation:
        tensors = _build_phase2_tensors(
            support_vis_real=support_vis_real,
            support_vis_imag=support_vis_imag,
            support_mask=support_mask,
            support_dirty=support_dirty,
            uv_coords=uv_coords,
            frame_uv_coords=frame_uv_coords,
            frame_uv_indices=frame_uv_indices,
            measurements=support_complex,
            device=device,
            model_config=model_config,
        )
        with torch.no_grad():
            outputs = model(
                visibility_input=tensors["visibility_input"],  # type: ignore[arg-type]
                dirty_input=tensors["dirty_input"],  # type: ignore[arg-type]
                measurements=tensors["measurements"],  # type: ignore[arg-type]
                mask=tensors["mask"],  # type: ignore[arg-type]
                baseline_pairs=tensors["baseline_pairs"],  # type: ignore[arg-type]
                frame_uv_indices=tensors["frame_uv_indices"],  # type: ignore[arg-type]
            )
        pre_dc = getattr(outputs, "pre_dc_prediction", None)
        return RealPhase2Prediction(
            mean=outputs.mean.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32),
            pre_dc_prediction=(
                pre_dc.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32) if pre_dc is not None else None
            ),
            adaptation_scale=1.0,
            support_loss_before=None,
            support_loss_after=None,
        )

    bridge = domain_bridge or EHTDomainBridge()
    normalized_support, scale = bridge.normalize_support(support_complex, support_mask)
    normalized_dirty = bridge.normalize_dirty(support_dirty, scale)
    tensors = _build_phase2_tensors(
        support_vis_real=normalized_support.real.astype(np.float32),
        support_vis_imag=normalized_support.imag.astype(np.float32),
        support_mask=support_mask,
        support_dirty=normalized_dirty,
        uv_coords=uv_coords,
        frame_uv_coords=frame_uv_coords,
        frame_uv_indices=frame_uv_indices,
        measurements=normalized_support,
        device=device,
        model_config=model_config,
    )

    with torch.no_grad():
        outputs_before = model(
            visibility_input=tensors["visibility_input"],  # type: ignore[arg-type]
            dirty_input=tensors["dirty_input"],  # type: ignore[arg-type]
            measurements=tensors["measurements"],  # type: ignore[arg-type]
            mask=tensors["mask"],  # type: ignore[arg-type]
            baseline_pairs=tensors["baseline_pairs"],  # type: ignore[arg-type]
            frame_uv_indices=tensors["frame_uv_indices"],  # type: ignore[arg-type]
        )
    support_loss_before = _support_loss_from_outputs(
        outputs_before,
        tensors["measurements"],  # type: ignore[arg-type]
        tensors["mask"],  # type: ignore[arg-type]
    )

    optimizer = TestTimeOptimizer(
        model=model,
        n_steps=tto_steps,
        lr=tto_lr,
        lambda_support=tto_lambda_support,
    )
    outputs_after = optimizer.adapt(
        visibility_input=tensors["visibility_input"],  # type: ignore[arg-type]
        dirty_input=tensors["dirty_input"],  # type: ignore[arg-type]
        support_vis=tensors["measurements"],  # type: ignore[arg-type]
        support_mask=tensors["mask"],  # type: ignore[arg-type]
        baseline_pairs=tensors["baseline_pairs"],  # type: ignore[arg-type]
        frame_uv_indices=tensors["frame_uv_indices"],  # type: ignore[arg-type]
    )
    support_loss_after = _support_loss_from_outputs(
        outputs_after,
        tensors["measurements"],  # type: ignore[arg-type]
        tensors["mask"],  # type: ignore[arg-type]
    )

    pre_dc = getattr(outputs_after, "pre_dc_prediction", None)
    denormalized_mean = bridge.denormalize_prediction(
        outputs_after.mean.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32),
        scale,
    )
    denormalized_pre_dc = None
    if pre_dc is not None:
        denormalized_pre_dc = bridge.denormalize_prediction(
            pre_dc.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32),
            scale,
        )

    return RealPhase2Prediction(
        mean=denormalized_mean,
        pre_dc_prediction=denormalized_pre_dc,
        adaptation_scale=float(scale),
        support_loss_before=support_loss_before,
        support_loss_after=support_loss_after,
    )
