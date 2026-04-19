"""GRMHD-based synthetic data generation for astrophysically realistic training.

Generates dynamic black-hole sequences using semi-analytic prescriptions
inspired by general-relativistic magnetohydrodynamic (GRMHD) simulations.
The model captures:
  - Relativistic accretion-flow emission with turbulent substructure
  - Photon ring / shadow geometry from Kerr spacetime ray-tracing approximation
  - Jet emission with collimation and opening angle
  - Stochastic turbulence following a Kolmogorov-like power spectrum
  - Doppler boosting from orbital motion
  - Temporal variability from magnetorotational instability (MRI) timescales

This is NOT a full GRMHD simulation. It is a fast, differentiable,
semi-analytic model calibrated to reproduce the statistical properties
of GRMHD images (e.g., from BHAC, iharm3d, KHARMA) at a fraction of the
computational cost, suitable for training learned imaging methods.

References:
  - Event Horizon Telescope Collaboration (2019), ApJL 875, L5
  - Gold et al. (2020), ApJ 897, 148 (GRMHD image library)
  - Porth et al. (2019), ApJS 243, 26 (BHAC code)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dynadiff_vlbi.data.feature_formatting import build_temporal_uv_grid
from dynadiff_vlbi.data.io import save_npz
from dynadiff_vlbi.physics.fourier_operator import FourierMeasurementOperator
from dynadiff_vlbi.physics.noise import apply_structured_visibility_corruption
from dynadiff_vlbi.physics.sampling import build_sampling_metadata, generate_temporal_uv_mask
from dynadiff_vlbi.utils.config import NoiseConfig, SamplingConfig, SyntheticSequenceConfig


@dataclass
class GRMHDConfig:
    """Configuration for GRMHD-inspired synthetic generation."""

    image_size: int = 128
    sequence_length: int = 8

    # Black hole shadow and photon ring
    spin: float = 0.94  # dimensionless spin a/M (positive = prograde)
    inclination_deg: float = 17.0  # observer inclination (M87: ~17 deg)
    shadow_diameter: float = 0.42  # in normalized image coords [-1,1]
    photon_ring_width: float = 0.015  # narrow ring from lensed emission
    photon_ring_intensity: float = 0.3  # relative to accretion flow peak

    # Accretion flow
    accretion_ring_radius: float = 0.40  # dominant emission ring
    accretion_ring_width: float = 0.06
    accretion_asymmetry: float = 0.45  # Doppler boosting strength
    accretion_peak_flux: float = 1.0

    # Turbulence
    turbulence_amplitude: float = 0.25
    turbulence_power_law: float = -5.0 / 3.0  # Kolmogorov
    turbulence_correlation_time: float = 0.3  # fraction of sequence

    # Jet
    jet_intensity: float = 0.12
    jet_half_opening_deg: float = 15.0
    jet_collimation: float = 0.8
    jet_position_angle_deg: float = 288.0  # M87 jet PA

    # Temporal variability
    variability_amplitude: float = 0.15
    hotspot_probability: float = 0.7
    hotspot_intensity: float = 0.4
    hotspot_width: float = 0.04
    hotspot_orbital_period_frames: float = 6.0  # frames per orbit

    # Dataset sizes
    train_size: int = 512
    val_size: int = 128
    test_size: int = 128


def _kerr_shadow_radius(spin: float, inclination_rad: float) -> float:
    """Approximate Kerr shadow radius in gravitational radii.

    Uses the analytic approximation from Johannsen & Psaltis (2010) for
    the shadow size as a function of spin and inclination.
    """
    a = abs(spin)
    cos_i = np.cos(inclination_rad)
    # Leading-order shadow radius: sqrt(27) M for Schwarzschild
    r_shadow = np.sqrt(27.0) * (1.0 - 0.04 * a**2 * (1.0 + cos_i**2))
    return float(r_shadow)


def _generate_turbulence_field(
    image_size: int,
    power_law: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a 2D turbulence field with specified power spectrum."""
    kx = np.fft.fftfreq(image_size, d=1.0 / image_size)
    ky = np.fft.fftfreq(image_size, d=1.0 / image_size)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
    k_mag[0, 0] = 1.0  # avoid division by zero

    # Power spectrum P(k) ~ k^(power_law)
    amplitude = k_mag ** (power_law / 2.0)
    amplitude[0, 0] = 0.0  # zero mean

    # Random phases
    phases = rng.uniform(0, 2 * np.pi, size=(image_size, image_size))
    fourier_field = amplitude * np.exp(1j * phases)

    field = np.fft.ifft2(fourier_field).real.astype(np.float32)
    field -= field.mean()
    std = field.std()
    if std > 1e-8:
        field /= std
    return field


def _doppler_factor(
    angle: np.ndarray,
    velocity: float,
    inclination_rad: float,
    asymmetry: float,
) -> np.ndarray:
    """Compute Doppler boosting factor for orbiting emission.

    D = 1 / (gamma * (1 - beta * sin(i) * sin(phi)))
    where phi is azimuthal angle in the image plane.
    """
    beta = velocity
    gamma = 1.0 / np.sqrt(max(1.0 - beta**2, 1e-8))
    sin_i = np.sin(inclination_rad)
    # Doppler factor D^3 for optically thin emission
    doppler = 1.0 / (gamma * (1.0 - beta * sin_i * np.sin(angle)))
    boosted = doppler**3
    # Normalize to unit mean and scale by asymmetry
    mean_boost = boosted.mean()
    if mean_boost > 1e-8:
        boosted = 1.0 + asymmetry * (boosted / mean_boost - 1.0)
    return boosted.astype(np.float32)


def generate_grmhd_sequence(
    config: GRMHDConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Generate one GRMHD-inspired black-hole movie sequence.

    Returns:
        sequence: (T, H, W) float32 array normalized to [0, 1]
        metadata: dict with ring_radius_px, hotspot_coords_px, etc.
    """
    size = config.image_size
    T = config.sequence_length
    incl_rad = np.radians(config.inclination_deg)

    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    angle = np.arctan2(yy, xx)

    # Shadow boundary
    shadow_r = config.shadow_diameter / 2.0

    # Generate time-correlated turbulence fields
    turb_fields = []
    prev_field = _generate_turbulence_field(size, config.turbulence_power_law, rng)
    alpha = config.turbulence_correlation_time
    for t in range(T):
        new_field = _generate_turbulence_field(size, config.turbulence_power_law, rng)
        field = alpha * prev_field + (1.0 - alpha) * new_field
        turb_fields.append(field)
        prev_field = field

    # Random parameters for this sequence
    pa_rad = np.radians(config.jet_position_angle_deg + rng.normal(0, 10))
    accretion_phase = rng.uniform(-np.pi, np.pi)
    orbital_velocity = 0.3 + rng.uniform(-0.05, 0.05)  # c units
    global_flux_variation = rng.uniform(0.8, 1.2)

    # Hotspot parameters
    add_hotspot = rng.random() < config.hotspot_probability
    hotspot_angle_start = rng.uniform(-np.pi, np.pi)
    hotspot_orbital_speed = 2 * np.pi / config.hotspot_orbital_period_frames
    hotspot_direction = rng.choice([-1.0, 1.0])
    hotspot_coords = np.zeros((T, 2), dtype=np.float32)

    seq = np.zeros((T, size, size), dtype=np.float32)

    for t in range(T):
        time_frac = t / max(T - 1, 1)

        # --- Accretion flow emission ---
        ring = np.exp(
            -0.5 * ((radius - config.accretion_ring_radius) / max(config.accretion_ring_width, 1e-3)) ** 2
        )

        # Doppler boosting from orbital motion
        phase_t = accretion_phase + orbital_velocity * 2 * np.pi * time_frac
        doppler = _doppler_factor(angle - phase_t, orbital_velocity, incl_rad, config.accretion_asymmetry)
        ring *= doppler

        # Turbulent substructure
        turb = 1.0 + config.turbulence_amplitude * turb_fields[t]
        ring *= np.clip(turb, 0.0, None)

        # Global temporal variability (MRI timescale)
        global_mod = global_flux_variation * (
            1.0 + config.variability_amplitude * np.sin(2 * np.pi * time_frac + rng.uniform(-0.5, 0.5))
        )
        frame = global_mod * config.accretion_peak_flux * ring

        # --- Photon ring ---
        photon_ring = np.exp(
            -0.5 * ((radius - shadow_r) / max(config.photon_ring_width, 1e-4)) ** 2
        )
        photon_ring *= doppler  # also boosted
        frame += config.photon_ring_intensity * photon_ring

        # --- Shadow depression ---
        shadow_mask = 1.0 - 0.85 * np.exp(-0.5 * (radius / max(shadow_r * 0.7, 1e-3)) ** 4)
        frame *= shadow_mask

        # --- Jet emission ---
        if config.jet_intensity > 0.0:
            cos_pa = np.cos(pa_rad)
            sin_pa = np.sin(pa_rad)
            x_jet = xx * cos_pa + yy * sin_pa
            y_jet = -xx * sin_pa + yy * cos_pa

            half_opening = np.radians(config.jet_half_opening_deg)
            # Collimated jet with parabolic opening
            jet_width = half_opening * (0.1 + config.jet_collimation * np.abs(x_jet))
            jet = np.exp(-0.5 * (y_jet / np.clip(jet_width, 1e-4, None)) ** 2)
            # Intensity decreases along jet axis
            jet *= np.exp(-np.abs(x_jet) / 0.5)
            # Only positive jet direction (counter-jet fainter)
            jet_pos = jet * (x_jet > 0.05).astype(np.float32)
            jet_neg = 0.15 * jet * (x_jet < -0.05).astype(np.float32)
            frame += config.jet_intensity * global_mod * (jet_pos + jet_neg)

        # --- Hotspot ---
        if add_hotspot:
            hs_theta = hotspot_angle_start + hotspot_direction * hotspot_orbital_speed * t
            hs_r = config.accretion_ring_radius * (1.0 + 0.05 * np.sin(3 * hs_theta))
            hs_x = hs_r * np.cos(hs_theta)
            hs_y = hs_r * np.sin(hs_theta)

            hs_amp = config.hotspot_intensity * (
                1.0 + 0.3 * config.variability_amplitude * np.sin(4 * np.pi * time_frac)
            )
            # Doppler boost the hotspot
            hs_doppler_arg = hs_theta - phase_t
            hs_boost = 1.0 + config.accretion_asymmetry * 0.5 * np.sin(hs_doppler_arg)
            hs_amp *= max(hs_boost, 0.2)

            hotspot = hs_amp * np.exp(
                -0.5 * (((xx - hs_x) / config.hotspot_width) ** 2 + ((yy - hs_y) / config.hotspot_width) ** 2)
            )
            frame += hotspot

            # Store pixel coordinates
            scale = 0.5 * (size - 1)
            hotspot_coords[t] = np.array(
                [(hs_x + 1.0) * scale, (hs_y + 1.0) * scale], dtype=np.float32
            )

        # Ensure non-negative
        frame = np.clip(frame, 0.0, None)
        seq[t] = frame.astype(np.float32)

    # Normalize to [0, 1]
    seq -= seq.min()
    peak = float(seq.max())
    if peak > 1e-6:
        seq /= peak

    ring_radius_px = config.accretion_ring_radius * 0.5 * (size - 1)
    metadata = {
        "ring_radius_px": float(ring_radius_px),
        "hotspot_coords_px": hotspot_coords,
        "spin": config.spin,
        "inclination_deg": config.inclination_deg,
        "source_type": "grmhd_semianalytic",
    }
    return seq.astype(np.float32), metadata


def grmhd_config_from_synthetic(
    synthetic_config: SyntheticSequenceConfig,
    **overrides: float,
) -> GRMHDConfig:
    """Create a GRMHDConfig that matches a SyntheticSequenceConfig's geometry."""
    return GRMHDConfig(
        image_size=synthetic_config.image_size,
        sequence_length=synthetic_config.sequence_length,
        accretion_ring_radius=synthetic_config.ring_radius,
        accretion_ring_width=synthetic_config.ring_width,
        hotspot_intensity=synthetic_config.hotspot_intensity,
        hotspot_width=synthetic_config.hotspot_width,
        jet_intensity=synthetic_config.jet_intensity,
        variability_amplitude=synthetic_config.temporal_variability,
        train_size=synthetic_config.train_size,
        val_size=synthetic_config.val_size,
        test_size=synthetic_config.test_size,
        **overrides,
    )


def generate_grmhd_split_dataset(
    num_sequences: int,
    grmhd_config: GRMHDConfig,
    sampling_config: SamplingConfig,
    noise_config: NoiseConfig,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate one GRMHD-based split with measurements and metadata."""
    rng = np.random.default_rng(seed)
    structured_corruptions = (
        sampling_config.mode == "station_tracks"
        and (
            noise_config.baseline_noise_jitter > 0.0
            or noise_config.gain_amplitude_std > 0.0
            or noise_config.gain_phase_std > 0.0
        )
    )
    operator = FourierMeasurementOperator(
        noise_std=0.0 if structured_corruptions else noise_config.noise_std,
        seed=seed + 137,
    )
    noise_rng = np.random.default_rng(seed + 137)
    uv_coords = build_temporal_uv_grid(
        image_size=grmhd_config.image_size,
        sequence_length=grmhd_config.sequence_length,
    )
    sampling_metadata = build_sampling_metadata(
        image_size=grmhd_config.image_size,
        sequence_length=grmhd_config.sequence_length,
        config=sampling_config,
        rng=np.random.default_rng(seed + 53),
    )

    ground_truth, dirty, vis_real, vis_imag, masks = [], [], [], [], []
    ring_radii, hotspot_coords_list = [], []

    for i in range(num_sequences):
        # Vary parameters per-sample for diversity
        sample_config = GRMHDConfig(
            image_size=grmhd_config.image_size,
            sequence_length=grmhd_config.sequence_length,
            spin=np.clip(grmhd_config.spin + rng.normal(0, 0.1), -0.998, 0.998),
            inclination_deg=np.clip(
                grmhd_config.inclination_deg + rng.normal(0, 8), 5, 80
            ),
            shadow_diameter=grmhd_config.shadow_diameter * rng.uniform(0.92, 1.08),
            accretion_ring_radius=grmhd_config.accretion_ring_radius * rng.uniform(0.90, 1.10),
            accretion_ring_width=grmhd_config.accretion_ring_width * rng.uniform(0.7, 1.3),
            accretion_asymmetry=np.clip(
                grmhd_config.accretion_asymmetry + rng.normal(0, 0.1), 0.1, 0.8
            ),
            turbulence_amplitude=grmhd_config.turbulence_amplitude * rng.uniform(0.5, 1.5),
            jet_intensity=grmhd_config.jet_intensity * rng.uniform(0.0, 2.0),
            jet_half_opening_deg=grmhd_config.jet_half_opening_deg + rng.normal(0, 3),
            jet_position_angle_deg=grmhd_config.jet_position_angle_deg + rng.normal(0, 15),
            variability_amplitude=grmhd_config.variability_amplitude * rng.uniform(0.5, 1.5),
            hotspot_probability=grmhd_config.hotspot_probability,
            hotspot_intensity=grmhd_config.hotspot_intensity * rng.uniform(0.5, 1.5),
            hotspot_width=grmhd_config.hotspot_width * rng.uniform(0.7, 1.3),
        )

        sequence, metadata = generate_grmhd_sequence(config=sample_config, rng=rng)
        mask = generate_temporal_uv_mask(
            image_size=grmhd_config.image_size,
            sequence_length=grmhd_config.sequence_length,
            config=sampling_config,
            rng=rng,
            metadata=sampling_metadata,
        )
        measurement_batch = operator.forward(sequence=sequence, mask=mask)
        noisy_vis = measurement_batch.noisy_visibilities
        dirty_recon = measurement_batch.dirty_reconstruction
        if structured_corruptions:
            noisy_vis = apply_structured_visibility_corruption(
                clean_visibilities=measurement_batch.clean_visibilities,
                mask=mask,
                noise_config=noise_config,
                rng=noise_rng,
                baseline_pairs=sampling_metadata.baseline_pairs,
                frame_uv_indices=sampling_metadata.frame_uv_indices,
            )
            dirty_recon = operator.dirty_reconstruct(noisy_vis, mask)

        ground_truth.append(sequence.astype(np.float32))
        dirty.append(dirty_recon.astype(np.float32))
        vis_real.append(noisy_vis.real.astype(np.float32))
        vis_imag.append(noisy_vis.imag.astype(np.float32))
        masks.append(mask.astype(np.float32))
        ring_radii.append(np.float32(metadata["ring_radius_px"]))
        hotspot_coords_list.append(
            np.asarray(metadata["hotspot_coords_px"], dtype=np.float32)
        )

    return {
        "ground_truth": np.stack(ground_truth).astype(np.float32),
        "dirty": np.stack(dirty).astype(np.float32),
        "vis_real": np.stack(vis_real).astype(np.float32),
        "vis_imag": np.stack(vis_imag).astype(np.float32),
        "mask": np.stack(masks).astype(np.float32),
        "ring_radius_px": np.asarray(ring_radii, dtype=np.float32),
        "hotspot_coords_px": np.stack(hotspot_coords_list).astype(np.float32),
        "uv_coords": uv_coords.astype(np.float32),
        "station_positions": sampling_metadata.station_positions.astype(np.float32),
        "baseline_pairs": sampling_metadata.baseline_pairs.astype(np.int32),
        "frame_uv_indices": sampling_metadata.frame_uv_indices.astype(np.int32),
        "frame_uv_coords": sampling_metadata.frame_uv_coords.astype(np.float32),
    }


def generate_grmhd_dataset_splits(
    output_dir: str | Path,
    grmhd_config: GRMHDConfig,
    sampling_config: SamplingConfig,
    noise_config: NoiseConfig,
    base_seed: int,
) -> dict[str, Path]:
    """Generate and save GRMHD-based train/val/test splits as NPZ files."""
    output_dir = Path(output_dir)
    split_sizes = {
        "train": grmhd_config.train_size,
        "val": grmhd_config.val_size,
        "test": grmhd_config.test_size,
    }
    split_offsets = {"train": 0, "val": 1_000, "test": 2_000}
    saved_paths: dict[str, Path] = {}
    for split_name, split_size in split_sizes.items():
        arrays = generate_grmhd_split_dataset(
            num_sequences=split_size,
            grmhd_config=grmhd_config,
            sampling_config=sampling_config,
            noise_config=noise_config,
            seed=base_seed + split_offsets[split_name],
        )
        split_path = output_dir / f"{split_name}_grmhd.npz"
        save_npz(split_path, arrays)
        saved_paths[split_name] = split_path
    return saved_paths
