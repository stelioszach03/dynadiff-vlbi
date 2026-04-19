"""Metric implementations for sequence reconstruction and uncertainty."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from skimage.metrics import structural_similarity

from dynadiff_vlbi.physics.fourier_operator import fft2c


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


def estimate_ring_thickness(sequence: np.ndarray) -> float:
    """Estimate ring thickness from the radial-profile spread around the peak."""

    mean_frame = np.mean(sequence, axis=0)
    profile = radial_profile(mean_frame)
    if profile.shape[0] <= 2:
        return 0.0
    radii = np.arange(profile.shape[0], dtype=np.float32)
    peak_index = int(np.argmax(profile[1:]) + 1)
    weights = np.maximum(profile, 0.0)
    local_window = np.abs(radii - float(peak_index)) <= 4.0
    if not np.any(local_window):
        return 0.0
    weights = weights * local_window
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-8:
        return 0.0
    mean_radius = float(np.sum(weights * radii) / weight_sum)
    variance = float(np.sum(weights * (radii - mean_radius) ** 2) / weight_sum)
    return float(2.355 * np.sqrt(max(variance, 0.0)))


def ring_radius_error(prediction: np.ndarray, target_radius_px: float) -> float:
    return float(abs(estimate_ring_radius(prediction) - target_radius_px))


def ring_thickness_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Difference in estimated ring thickness between prediction and target."""

    return float(abs(estimate_ring_thickness(prediction) - estimate_ring_thickness(target)))


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


def hotspot_track_velocity_error(prediction: np.ndarray, target_hotspot_coords_px: np.ndarray) -> float:
    predicted_track = np.stack([brightest_centroid(frame) for frame in prediction], axis=0)
    valid = np.all(np.isfinite(target_hotspot_coords_px), axis=1)
    if valid.sum() <= 1:
        return 0.0
    predicted_velocity = np.diff(predicted_track[valid], axis=0)
    target_velocity = np.diff(target_hotspot_coords_px[valid], axis=0)
    return float(np.mean(np.linalg.norm(predicted_velocity - target_velocity, axis=1)))


def _angular_profile(frame: np.ndarray, target_radius_px: float, width_px: float = 2.0, bins: int = 36) -> np.ndarray:
    center_y = (frame.shape[0] - 1) / 2.0
    center_x = (frame.shape[1] - 1) / 2.0
    yy, xx = np.indices(frame.shape)
    radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    angle = np.mod(np.arctan2(yy - center_y, xx - center_x), 2.0 * np.pi)
    annulus = np.abs(radius - target_radius_px) <= width_px
    if not np.any(annulus):
        return np.zeros(bins, dtype=np.float32)
    bin_indices = np.floor(angle[annulus] / (2.0 * np.pi) * bins).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    values = frame[annulus]
    weighted_sum = np.bincount(bin_indices, weights=values, minlength=bins)
    weighted_count = np.bincount(bin_indices, minlength=bins)
    return (weighted_sum / np.maximum(weighted_count, 1)).astype(np.float32)


def arc_profile_correlation(prediction: np.ndarray, target: np.ndarray, target_ring_radius_px: float) -> float:
    prediction_profile = _angular_profile(np.mean(prediction, axis=0), target_ring_radius_px)
    target_profile = _angular_profile(np.mean(target, axis=0), target_ring_radius_px)
    if np.allclose(prediction_profile.std(), 0.0) or np.allclose(target_profile.std(), 0.0):
        return 0.0
    correlation = np.corrcoef(prediction_profile, target_profile)[0, 1]
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def _dominant_sector_angle(frame: np.ndarray, target_ring_radius_px: float, bins: int = 36) -> float:
    profile = _angular_profile(frame, target_ring_radius_px, bins=bins)
    if np.allclose(profile.std(), 0.0):
        return 0.0
    dominant_index = int(np.argmax(profile))
    return float((dominant_index + 0.5) / bins * 2.0 * np.pi)


def bright_sector_angle_error(prediction: np.ndarray, target: np.ndarray, target_ring_radius_px: float) -> float:
    """Angular error of the brightest ring sector on the mean frame."""

    prediction_angle = _dominant_sector_angle(np.mean(prediction, axis=0), target_ring_radius_px)
    target_angle = _dominant_sector_angle(np.mean(target, axis=0), target_ring_radius_px)
    return abs(_wrapped_phase_difference(prediction_angle, target_angle))


def observed_visibility_rmse(
    prediction: np.ndarray,
    measurements: np.ndarray,
    mask: np.ndarray,
) -> float:
    predicted_visibilities = fft2c(prediction)
    residual = (predicted_visibilities - measurements) * mask
    if np.count_nonzero(mask) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(residual[mask.astype(bool)]) ** 2)))


def weighted_visibility_chi2(
    prediction: np.ndarray,
    measurements: np.ndarray,
    mask: np.ndarray,
    sigma: np.ndarray | None,
) -> float:
    """Sigma-weighted visibility residual sum on observed coefficients."""

    if sigma is None:
        return float("nan")
    predicted_visibilities = fft2c(prediction)
    valid = mask.astype(bool) & np.isfinite(sigma) & (sigma > 0.0)
    if not np.any(valid):
        return float("nan")
    residual = predicted_visibilities - measurements
    return float(np.sum((np.abs(residual[valid]) / sigma[valid]) ** 2))


def reduced_weighted_visibility_chi2(
    prediction: np.ndarray,
    measurements: np.ndarray,
    mask: np.ndarray,
    sigma: np.ndarray | None,
) -> float:
    """Sigma-weighted visibility residual average on observed coefficients."""

    chi2 = weighted_visibility_chi2(
        prediction=prediction,
        measurements=measurements,
        mask=mask,
        sigma=sigma,
    )
    if np.isnan(chi2):
        return float("nan")
    valid = mask.astype(bool) & np.isfinite(sigma) & (sigma > 0.0) if sigma is not None else np.zeros_like(mask, dtype=bool)
    coefficient_count = int(np.count_nonzero(valid))
    if coefficient_count <= 0:
        return float("nan")
    return float(chi2 / float(coefficient_count))


def _wrapped_phase_difference(first: float, second: float) -> float:
    return float(np.angle(np.exp(1j * (first - second))))


def closure_phase_mae(
    prediction: np.ndarray,
    measurements: np.ndarray,
    mask: np.ndarray,
    baseline_pairs: np.ndarray | None,
    frame_uv_indices: np.ndarray | None,
    max_triangles: int = 24,
) -> float:
    if baseline_pairs is None or frame_uv_indices is None:
        return float("nan")
    if baseline_pairs.size == 0 or frame_uv_indices.size == 0:
        return float("nan")

    pair_lookup = {tuple(pair.tolist()): pair_index for pair_index, pair in enumerate(baseline_pairs)}
    triangles = list(combinations(range(int(baseline_pairs.max()) + 1), 3))
    if not triangles:
        return float("nan")
    triangles = triangles[:max_triangles]

    predicted_visibilities = fft2c(prediction)
    errors: list[float] = []
    for frame_index in range(prediction.shape[0]):
        frame_mask = mask[frame_index].astype(bool)
        frame_indices = frame_uv_indices[frame_index]
        for first, second, third in triangles:
            pair_ab = pair_lookup.get((first, second))
            pair_bc = pair_lookup.get((second, third))
            pair_ac = pair_lookup.get((first, third))
            if pair_ab is None or pair_bc is None or pair_ac is None:
                continue

            row_ab, col_ab = frame_indices[pair_ab]
            row_bc, col_bc = frame_indices[pair_bc]
            row_ac, col_ac = frame_indices[pair_ac]
            if not (frame_mask[row_ab, col_ab] and frame_mask[row_bc, col_bc] and frame_mask[row_ac, col_ac]):
                continue

            pred_ab = predicted_visibilities[frame_index, row_ab, col_ab]
            pred_bc = predicted_visibilities[frame_index, row_bc, col_bc]
            pred_ac = predicted_visibilities[frame_index, row_ac, col_ac]

            meas_ab = measurements[frame_index, row_ab, col_ab]
            meas_bc = measurements[frame_index, row_bc, col_bc]
            meas_ac = measurements[frame_index, row_ac, col_ac]

            predicted_phase = np.angle(pred_ab * pred_bc * np.conj(pred_ac))
            measured_phase = np.angle(meas_ab * meas_bc * np.conj(meas_ac))
            errors.append(abs(_wrapped_phase_difference(float(predicted_phase), float(measured_phase))))

    if not errors:
        return float("nan")
    return float(np.mean(errors))


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


def risk_coverage_auc(
    target: np.ndarray,
    predictive_mean: np.ndarray,
    predictive_std: np.ndarray,
    min_coverage: float = 0.1,
    num_points: int = 20,
) -> float:
    errors = ((target - predictive_mean) ** 2).reshape(-1)
    uncertainties = predictive_std.reshape(-1)
    ordering = np.argsort(uncertainties)
    sorted_errors = errors[ordering]
    coverages = np.linspace(min_coverage, 1.0, num_points)
    risks = []
    for coverage in coverages:
        keep = max(1, int(round(coverage * sorted_errors.shape[0])))
        risks.append(float(np.mean(sorted_errors[:keep])))
    return float(np.trapezoid(risks, coverages) / max(1.0 - min_coverage, 1e-6))


def topk_error_recall(
    target: np.ndarray,
    predictive_mean: np.ndarray,
    predictive_std: np.ndarray,
    fraction: float = 0.10,
) -> float:
    errors = np.abs(target - predictive_mean).reshape(-1)
    uncertainties = predictive_std.reshape(-1)
    topk = max(1, int(round(fraction * errors.shape[0])))
    top_error_indices = np.argpartition(errors, -topk)[-topk:]
    top_uncertainty_indices = np.argpartition(uncertainties, -topk)[-topk:]
    overlap = np.intersect1d(top_error_indices, top_uncertainty_indices).size
    return float(overlap / topk)


def compute_reconstruction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_ring_radius_px: float,
    target_hotspot_coords_px: np.ndarray,
    measurements: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    baseline_pairs: np.ndarray | None = None,
    frame_uv_indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the core reconstruction metrics for a sequence."""

    metrics = {
        "mse": mean_squared_error(prediction, target),
        "psnr": peak_signal_to_noise_ratio(prediction, target),
        "ssim": structural_similarity_sequence(prediction, target),
        "temporal_consistency": temporal_consistency_error(prediction, target),
        "ring_radius_error": ring_radius_error(prediction, target_ring_radius_px),
        "ring_thickness_error": ring_thickness_error(prediction, target),
        "hotspot_localization_error": hotspot_localization_error(prediction, target_hotspot_coords_px),
        "arc_profile_correlation": arc_profile_correlation(prediction, target, target_ring_radius_px),
        "bright_sector_angle_error": bright_sector_angle_error(prediction, target, target_ring_radius_px),
        "hotspot_track_velocity_error": hotspot_track_velocity_error(prediction, target_hotspot_coords_px),
        "observed_visibility_rmse": float("nan"),
        "closure_phase_mae": float("nan"),
    }
    if measurements is not None and mask is not None:
        metrics["observed_visibility_rmse"] = observed_visibility_rmse(
            prediction=prediction,
            measurements=measurements,
            mask=mask,
        )
        metrics["closure_phase_mae"] = closure_phase_mae(
            prediction=prediction,
            measurements=measurements,
            mask=mask,
            baseline_pairs=baseline_pairs,
            frame_uv_indices=frame_uv_indices,
        )
    return metrics
