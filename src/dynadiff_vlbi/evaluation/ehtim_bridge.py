"""Frozen eht-imaging bridge baseline for deterministic support-target public-EHT evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io

import numpy as np

from dynadiff_vlbi.physics.classical_reconstruction import dirty_image_reconstruction


@dataclass(frozen=True)
class EHTIMBridgeConfig:
    """One frozen, benchmark-facing eht-imaging bridge configuration."""

    image_size: int = 32
    fov: float = 1.0
    bw_hz: float = 4.0e9
    data_term_vis_weight: float = 300.0
    reg_simple_weight: float = 1.0
    reg_tv2_weight: float = 1.0
    max_iterations: int = 40
    stop: float = 1.0e-5
    min_sigma: float = 1.0e-4
    min_flux: float = 1.0e-3


EHTIM_PUBLIC_BRIDGE_CONFIG = EHTIMBridgeConfig()


def _import_ehtim():
    import ehtim as eh  # type: ignore[import-not-found]
    import ehtim.const_def as ehc  # type: ignore[import-not-found]

    return eh, ehc


def centered_frequency_axis(image_size: int) -> np.ndarray:
    """Return centered discrete Fourier frequencies for the benchmark grid."""

    return np.fft.fftshift(np.fft.fftfreq(int(image_size), d=1.0 / float(image_size))).astype(np.float64)


def build_station_array(*, station_labels: np.ndarray, station_positions: np.ndarray) -> np.ndarray:
    """Build a small pseudo array table for the benchmark bridge."""

    _, ehc = _import_ehtim()
    rows = []
    for index, label in enumerate(np.asarray(station_labels).tolist()):
        x_coord, y_coord = np.asarray(station_positions, dtype=np.float64)[index]
        rows.append((str(label), float(x_coord), float(y_coord), 0.0, 1.0, 1.0, 0j, 0j, 0.0, 0.0, 0.0))
    return np.asarray(rows, dtype=ehc.DTARR)


def representative_support_pairs(
    *,
    support_mask: np.ndarray,
    frame_uv_indices: np.ndarray,
    baseline_pairs: np.ndarray,
) -> dict[tuple[int, int, int], tuple[int, int]]:
    """Assign one representative station pair to each support cell."""

    mapping: dict[tuple[int, int, int], tuple[int, int]] = {}
    if frame_uv_indices.size == 0 or baseline_pairs.size == 0:
        return mapping
    for frame_index in range(int(frame_uv_indices.shape[0])):
        for baseline_index, (first, second) in enumerate(np.asarray(baseline_pairs, dtype=np.int64).tolist()):
            row, col = frame_uv_indices[frame_index, baseline_index]
            if support_mask[frame_index, int(row), int(col)] <= 0.0:
                continue
            mapping.setdefault((frame_index, int(row), int(col)), (int(first), int(second)))
    return mapping


def _frame_datatable(
    *,
    frame_index: int,
    measurements: np.ndarray,
    support_mask: np.ndarray,
    sigma: np.ndarray | None,
    station_labels: np.ndarray,
    support_pairs: dict[tuple[int, int, int], tuple[int, int]],
    frequency_axis: np.ndarray,
    min_sigma: float,
) -> np.ndarray:
    _, ehc = _import_ehtim()
    rows = []
    observed_indices = np.argwhere(support_mask[frame_index] > 0.0)
    fallback_pair = (0, 1 if len(station_labels) > 1 else 0)
    for row, col in observed_indices.tolist():
        first, second = support_pairs.get((frame_index, int(row), int(col)), fallback_pair)
        sigma_value = min_sigma
        if sigma is not None:
            sigma_value = float(max(float(sigma[frame_index, row, col]), min_sigma))
        rows.append(
            (
                float(frame_index),
                1.0,
                str(station_labels[first]),
                str(station_labels[second]),
                0.0,
                0.0,
                float(frequency_axis[int(col)]),
                float(frequency_axis[int(row)]),
                complex(measurements[frame_index, row, col]),
                0j,
                0j,
                0j,
                sigma_value,
                1.0,
                1.0,
                1.0,
            )
        )
    return np.asarray(rows, dtype=ehc.DTPOL_STOKES)


def predict_ehtim_bridge_sequence(
    *,
    measurements: np.ndarray,
    support_mask: np.ndarray,
    sigma: np.ndarray | None,
    frame_uv_indices: np.ndarray,
    baseline_pairs: np.ndarray,
    station_labels: np.ndarray,
    station_positions: np.ndarray,
    rf_hz: float,
    source_name: str,
    mjd: float,
    config: EHTIMBridgeConfig = EHTIM_PUBLIC_BRIDGE_CONFIG,
) -> np.ndarray:
    """Run the frozen eht-imaging bridge on one support-only sequence."""

    eh, _ = _import_ehtim()
    image_size = int(measurements.shape[-1])
    frequency_axis = centered_frequency_axis(image_size)
    station_array = build_station_array(station_labels=station_labels, station_positions=station_positions)
    support_pairs = representative_support_pairs(
        support_mask=support_mask,
        frame_uv_indices=frame_uv_indices,
        baseline_pairs=baseline_pairs,
    )

    frames: list[np.ndarray] = []
    for frame_index in range(int(measurements.shape[0])):
        datatable = _frame_datatable(
            frame_index=frame_index,
            measurements=measurements,
            support_mask=support_mask,
            sigma=sigma,
            station_labels=station_labels,
            support_pairs=support_pairs,
            frequency_axis=frequency_axis,
            min_sigma=config.min_sigma,
        )
        if datatable.size == 0:
            frames.append(np.zeros((image_size, image_size), dtype=np.float32))
            continue
        if not np.any((np.abs(datatable["u"]) > 0.0) | (np.abs(datatable["v"]) > 0.0)):
            fallback = dirty_image_reconstruction(
                measurements=measurements[frame_index : frame_index + 1],
                mask=support_mask[frame_index : frame_index + 1],
            )[0]
            frames.append(fallback.astype(np.float32))
            continue

        obs = eh.obsdata.Obsdata(
            ra=0.0,
            dec=0.0,
            rf=float(rf_hz),
            bw=float(config.bw_hz),
            datatable=datatable,
            tarr=station_array,
            source=str(source_name),
            mjd=int(np.floor(float(mjd))),
            polrep="stokes",
            timetype="UTC",
        )
        init = eh.image.make_empty(
            image_size,
            float(config.fov),
            0.0,
            0.0,
            rf=float(rf_hz),
            source=str(source_name),
            mjd=int(np.floor(float(mjd))),
            time=float(frame_index),
        )
        amplitudes = np.abs(datatable["vis"])
        flux = float(max(float(np.max(amplitudes)), config.min_flux))
        prior = init.add_flat(flux)
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            imager = eh.imager.Imager(
                obs,
                prior,
                prior_im=prior,
                flux=flux,
                data_term={"vis": float(config.data_term_vis_weight)},
                reg_term={
                    "simple": float(config.reg_simple_weight),
                    "tv2": float(config.reg_tv2_weight),
                },
                maxit=int(config.max_iterations),
                norm_reg=True,
                ttype="direct",
                weighting="natural",
                stop=float(config.stop),
            )
            imager.make_image_I(show_updates=False, niter=1, blur_frac=0.0)
        frame_prediction = imager.out_last().imarr("I").astype(np.float32)
        frames.append(np.clip(frame_prediction, 0.0, None))

    return np.stack(frames).astype(np.float32)
