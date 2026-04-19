"""Image-domain morphology metrics for EHT-like validation.

These metrics allow approximate morphology validation on reconstructed
images by comparing extracted structural properties against published
EHT measurements (e.g., M87* ring diameter ~42 uas, brightness asymmetry).

The metrics do not require image-domain ground truth in the same way as
MSE/SSIM. Instead, they extract physical properties from reconstructions
and compare against externally published reference values.

References:
  - EHT Collaboration (2019), ApJL 875, L6 (M87* ring diameter 42 +/- 3 uas)
  - EHT Collaboration (2019), ApJL 875, L4 (imaging methods, brightness ratio)
  - EHT Collaboration (2022), ApJL 930, L12 (Sgr A* ring 51.8 +/- 2.3 uas)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynadiff_vlbi.evaluation.metrics import radial_profile


@dataclass
class RingMorphology:
    """Extracted ring morphology properties from a reconstructed image."""

    ring_diameter_px: float
    ring_width_px: float
    brightness_ratio: float  # max/min along the ring annulus
    position_angle_deg: float  # angle of peak brightness
    ring_circularity: float  # 1.0 = perfect circle
    shadow_depth: float  # contrast between center and ring
    fractional_central_brightness: float  # center / ring_peak


@dataclass
class MorphologyReference:
    """Published reference morphology for a known source."""

    source_name: str
    ring_diameter_uas: float
    ring_diameter_uas_error: float
    brightness_ratio_range: tuple[float, float]  # (min, max) from published
    position_angle_deg: float | None
    pixel_scale_uas_per_px: float | None  # set at evaluation time


# Published EHT reference values
M87_2017_REFERENCE = MorphologyReference(
    source_name="M87* 2017",
    ring_diameter_uas=42.0,
    ring_diameter_uas_error=3.0,
    brightness_ratio_range=(2.0, 10.0),
    position_angle_deg=None,  # varies with epoch
    pixel_scale_uas_per_px=None,
)

SGRA_2017_REFERENCE = MorphologyReference(
    source_name="Sgr A* 2017",
    ring_diameter_uas=51.8,
    ring_diameter_uas_error=2.3,
    brightness_ratio_range=(1.5, 5.0),
    position_angle_deg=None,
    pixel_scale_uas_per_px=None,
)


def extract_ring_morphology(
    image: np.ndarray,
    annulus_width_px: float = 3.0,
    angular_bins: int = 72,
) -> RingMorphology:
    """Extract ring morphology properties from a single 2D image.

    Args:
        image: 2D array (H, W) of the reconstructed image.
        annulus_width_px: Width of the annular region for azimuthal analysis.
        angular_bins: Number of bins for azimuthal brightness profile.

    Returns:
        RingMorphology with extracted structural properties.
    """
    H, W = image.shape
    center_y = (H - 1) / 2.0
    center_x = (W - 1) / 2.0

    # --- Ring diameter from radial profile peak ---
    profile = radial_profile(image)
    if profile.shape[0] <= 2:
        return RingMorphology(
            ring_diameter_px=0.0, ring_width_px=0.0, brightness_ratio=1.0,
            position_angle_deg=0.0, ring_circularity=1.0,
            shadow_depth=0.0, fractional_central_brightness=1.0,
        )

    # Find peak excluding the center pixel
    ring_radius_px = float(np.argmax(profile[1:]) + 1)
    ring_diameter_px = 2.0 * ring_radius_px

    # --- Ring width from FWHM of radial profile ---
    peak_val = profile[int(ring_radius_px)]
    half_max = 0.5 * peak_val
    above_half = profile >= half_max
    fwhm_indices = np.where(above_half)[0]
    if len(fwhm_indices) >= 2:
        ring_width_px = float(fwhm_indices[-1] - fwhm_indices[0])
    else:
        ring_width_px = 1.0

    # --- Azimuthal brightness profile for asymmetry ---
    yy, xx = np.indices(image.shape)
    radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    angle = np.mod(np.arctan2(yy - center_y, xx - center_x), 2.0 * np.pi)

    annulus_mask = np.abs(radius - ring_radius_px) <= annulus_width_px
    if not np.any(annulus_mask):
        return RingMorphology(
            ring_diameter_px=ring_diameter_px, ring_width_px=ring_width_px,
            brightness_ratio=1.0, position_angle_deg=0.0,
            ring_circularity=1.0, shadow_depth=0.0,
            fractional_central_brightness=1.0,
        )

    bin_edges = np.linspace(0, 2 * np.pi, angular_bins + 1)
    azimuthal_profile = np.zeros(angular_bins, dtype=np.float64)
    for i in range(angular_bins):
        in_bin = annulus_mask & (angle >= bin_edges[i]) & (angle < bin_edges[i + 1])
        if np.any(in_bin):
            azimuthal_profile[i] = float(np.mean(image[in_bin]))

    # Brightness ratio
    nonzero = azimuthal_profile[azimuthal_profile > 0]
    if len(nonzero) >= 2:
        brightness_ratio = float(nonzero.max() / max(nonzero.min(), 1e-10))
    else:
        brightness_ratio = 1.0

    # Position angle of peak brightness
    peak_bin = int(np.argmax(azimuthal_profile))
    position_angle_rad = (peak_bin + 0.5) / angular_bins * 2 * np.pi
    position_angle_deg = float(np.degrees(position_angle_rad))

    # --- Ring circularity ---
    # Measure radius at peak brightness in each angular bin
    radii_at_peak = []
    for i in range(angular_bins):
        in_sector = (angle >= bin_edges[i]) & (angle < bin_edges[i + 1])
        sector_img = image * in_sector
        if sector_img.max() > 0:
            max_idx = np.unravel_index(np.argmax(sector_img), sector_img.shape)
            r = np.sqrt((max_idx[1] - center_x) ** 2 + (max_idx[0] - center_y) ** 2)
            radii_at_peak.append(r)
    if len(radii_at_peak) >= 4:
        radii_arr = np.array(radii_at_peak)
        ring_circularity = float(1.0 - radii_arr.std() / max(radii_arr.mean(), 1e-8))
    else:
        ring_circularity = 1.0

    # --- Shadow depth ---
    center_region = radius <= max(ring_radius_px * 0.4, 2.0)
    if np.any(center_region):
        center_brightness = float(np.mean(image[center_region]))
    else:
        center_brightness = float(image[int(center_y), int(center_x)])
    shadow_depth = float(1.0 - center_brightness / max(peak_val, 1e-10))
    fractional_central_brightness = float(center_brightness / max(peak_val, 1e-10))

    return RingMorphology(
        ring_diameter_px=ring_diameter_px,
        ring_width_px=ring_width_px,
        brightness_ratio=brightness_ratio,
        position_angle_deg=position_angle_deg,
        ring_circularity=ring_circularity,
        shadow_depth=shadow_depth,
        fractional_central_brightness=fractional_central_brightness,
    )


def morphology_consistency_score(
    morphology: RingMorphology,
    reference: MorphologyReference,
    pixel_scale_uas_per_px: float,
) -> dict[str, float]:
    """Score extracted morphology against a published reference.

    Returns a dict of per-property consistency scores in [0, 1],
    where 1.0 means perfect agreement with the reference.
    """
    scores: dict[str, float] = {}

    # Ring diameter consistency (Gaussian penalty centered on reference)
    extracted_diameter_uas = morphology.ring_diameter_px * pixel_scale_uas_per_px
    diameter_error_uas = abs(extracted_diameter_uas - reference.ring_diameter_uas)
    scores["ring_diameter_score"] = float(
        np.exp(-0.5 * (diameter_error_uas / reference.ring_diameter_uas_error) ** 2)
    )
    scores["ring_diameter_uas"] = extracted_diameter_uas
    scores["ring_diameter_error_uas"] = diameter_error_uas

    # Brightness ratio consistency
    br = morphology.brightness_ratio
    br_min, br_max = reference.brightness_ratio_range
    if br_min <= br <= br_max:
        scores["brightness_ratio_score"] = 1.0
    elif br < br_min:
        scores["brightness_ratio_score"] = float(np.exp(-((br_min - br) / br_min) ** 2))
    else:
        scores["brightness_ratio_score"] = float(np.exp(-((br - br_max) / br_max) ** 2))
    scores["brightness_ratio"] = br

    # Shadow depth (should be significant for black hole images)
    scores["shadow_depth_score"] = float(np.clip(morphology.shadow_depth, 0.0, 1.0))
    scores["shadow_depth"] = morphology.shadow_depth

    # Ring circularity (should be close to 1.0 for M87)
    scores["circularity_score"] = float(np.clip(morphology.ring_circularity, 0.0, 1.0))
    scores["ring_circularity"] = morphology.ring_circularity

    # Composite score
    scores["composite_morphology_score"] = float(np.mean([
        scores["ring_diameter_score"],
        scores["brightness_ratio_score"],
        scores["shadow_depth_score"],
        scores["circularity_score"],
    ]))

    return scores


def compute_sequence_morphology(
    sequence: np.ndarray,
    reference: MorphologyReference | None = None,
    pixel_scale_uas_per_px: float | None = None,
) -> dict[str, float]:
    """Compute morphology metrics for a full temporal sequence.

    Returns per-frame averages and optionally reference consistency scores.
    """
    T = sequence.shape[0]
    morphologies = [extract_ring_morphology(sequence[t]) for t in range(T)]

    results: dict[str, float] = {
        "mean_ring_diameter_px": float(np.mean([m.ring_diameter_px for m in morphologies])),
        "std_ring_diameter_px": float(np.std([m.ring_diameter_px for m in morphologies])),
        "mean_ring_width_px": float(np.mean([m.ring_width_px for m in morphologies])),
        "mean_brightness_ratio": float(np.mean([m.brightness_ratio for m in morphologies])),
        "mean_shadow_depth": float(np.mean([m.shadow_depth for m in morphologies])),
        "mean_circularity": float(np.mean([m.ring_circularity for m in morphologies])),
        "temporal_diameter_stability": float(
            1.0 - np.std([m.ring_diameter_px for m in morphologies])
            / max(np.mean([m.ring_diameter_px for m in morphologies]), 1e-8)
        ),
    }

    if reference is not None and pixel_scale_uas_per_px is not None:
        # Score the time-averaged image
        mean_image = np.mean(sequence, axis=0)
        mean_morph = extract_ring_morphology(mean_image)
        consistency = morphology_consistency_score(
            mean_morph, reference, pixel_scale_uas_per_px
        )
        results.update({f"ref_{k}": v for k, v in consistency.items()})

    return results
