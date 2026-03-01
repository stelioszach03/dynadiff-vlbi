"""Metric implementations for sequence reconstruction and uncertainty."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def mean_squared_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def peak_signal_to_noise_ratio(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = mean_squared_error(prediction, target)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def structural_similarity_sequence(prediction: np.ndarray, target: np.ndarray) -> float:
    values = []
    for pred_frame, target_frame in zip(prediction, target):
        values.append(
            structural_similarity(
                pred_frame,
                target_frame,
                data_range=1.0,
            )
        )
    return float(np.mean(values))


def temporal_consistency_error(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.shape[0] <= 1:
        return 0.0
    pred_diff = np.diff(prediction, axis=0)
    target_diff = np.diff(target, axis=0)
    return mean_squared_error(pred_diff, target_diff)


def radial_profile(image: np.ndarray) -> np.ndarray:
    center_y = (image.shape[0] - 1) / 2.0
    center_x = (image.shape[1] - 1) / 2.0
    yy, xx = np.indices(image.shape)
    radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    radius_int = np.rint(radius).astype(np.int32)
    radial_sum = np.bincount(radius_int.ravel(), weights=image.ravel())
    radial_count = np.bincount(radius_int.ravel())
    return radial_sum / np.maximum(radial_count, 1)


def estimate_ring_radius(sequence: np.ndarray) -> float:
    mean_frame = np.mean(sequence, axis=0)
    profile = radial_profile(mean_frame)
    if profile.shape[0] <= 1:
        return 0.0
    return float(np.argmax(profile[1:]) + 1)


def ring_radius_error(prediction: np.ndarray, target_radius_px: float) -> float:
    return float(abs(estimate_ring_radius(prediction) - target_radius_px))


def brightest_centroid(frame: np.ndarray) -> np.ndarray:
    threshold = np.quantile(frame, 0.995)
    mask = frame >= threshold
    if not np.any(mask):
        y_idx, x_idx = np.unravel_index(np.argmax(frame), frame.shape)
        return np.asarray([x_idx, y_idx], dtype=np.float32)
    yy, xx = np.indices(frame.shape)
    weights = frame * mask
    total = max(float(weights.sum()), 1e-8)
    x_coord = float((weights * xx).sum() / total)
    y_coord = float((weights * yy).sum() / total)
    return np.asarray([x_coord, y_coord], dtype=np.float32)


def hotspot_localization_error(prediction: np.ndarray, target_hotspot_coords_px: np.ndarray) -> float:
    errors = []
    for frame, target_coord in zip(prediction, target_hotspot_coords_px):
        if not np.all(np.isfinite(target_coord)):
            continue
        pred_coord = brightest_centroid(frame)
        errors.append(float(np.linalg.norm(pred_coord - target_coord)))
    return float(np.mean(errors)) if errors else 0.0


def empirical_coverage(target: np.ndarray, predictive_mean: np.ndarray, predictive_std: np.ndarray) -> float:
    lower = predictive_mean - 2.0 * predictive_std
    upper = predictive_mean + 2.0 * predictive_std
    covered = (target >= lower) & (target <= upper)
    return float(covered.mean())


def uncertainty_error_correlation(target: np.ndarray, predictive_mean: np.ndarray, predictive_std: np.ndarray) -> float:
    errors = np.abs(target - predictive_mean).reshape(-1)
    uncertainties = predictive_std.reshape(-1)
    if np.allclose(uncertainties.std(), 0.0):
        return 0.0
    correlation = np.corrcoef(errors, uncertainties)[0, 1]
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def compute_reconstruction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_ring_radius_px: float,
    target_hotspot_coords_px: np.ndarray,
) -> dict[str, float]:
    """Compute the core reconstruction metrics for a sequence."""

    return {
        "mse": mean_squared_error(prediction, target),
        "psnr": peak_signal_to_noise_ratio(prediction, target),
        "ssim": structural_similarity_sequence(prediction, target),
        "temporal_consistency": temporal_consistency_error(prediction, target),
        "ring_radius_error": ring_radius_error(prediction, target_ring_radius_px),
        "hotspot_localization_error": hotspot_localization_error(prediction, target_hotspot_coords_px),
    }
