"""Helpers for official public EHT calibrated-data releases used in EMC validation."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import numpy as np

from dynadiff_vlbi.data.feature_formatting import build_temporal_uv_grid
from dynadiff_vlbi.data.io import save_npz
from dynadiff_vlbi.physics.classical_reconstruction import dirty_image_reconstruction
from dynadiff_vlbi.utils.logging_utils import save_json


@dataclass(frozen=True)
class PublicEHTReleaseSpec:
    """One official public EHT calibrated-data release supported by the loader."""

    release_code: str
    repo_url: str
    target_id: str
    campaign_year: int
    pipeline_name: str
    file_glob: str
    filename_regex: str
    dataset_description: str


PUBLIC_EHT_RELEASE_SPECS: dict[str, PublicEHTReleaseSpec] = {
    "2019-D01-01": PublicEHTReleaseSpec(
        release_code="2019-D01-01",
        repo_url="https://github.com/eventhorizontelescope/2019-D01-01.git",
        target_id="M87",
        campaign_year=2017,
        pipeline_name="HOPS netcal Stokes I",
        file_glob="SR1_M87_2017_*_hops_netcal_StokesI.csv",
        filename_regex=r"SR1_M87_2017_(?P<day>\d{3})_(?P<band>hi|lo)_hops_netcal_StokesI\.csv$",
        dataset_description=(
            "Official public calibrated 2017 EHT M87 Stokes I release "
            "(2019-D01-01, HOPS netcal, hi/lo bands)."
        ),
    ),
    "2024-D01-01": PublicEHTReleaseSpec(
        release_code="2024-D01-01",
        repo_url="https://github.com/eventhorizontelescope/2024-D01-01.git",
        target_id="M87",
        campaign_year=2018,
        pipeline_name="HOPS netcal 10 s Stokes I",
        file_glob="L2V1_M87_2018_*_hops_netcal_10s_StokesI.csv",
        filename_regex=r"L2V1_M87_2018_(?P<day>\d{3})_(?P<band>b[1-4])_hops_netcal_10s_StokesI\.csv$",
        dataset_description=(
            "Official public calibrated 2018 EHT M87 Stokes I release "
            "(2024-D01-01, HOPS netcal 10 s, bands b1-b4)."
        ),
    ),
    "2020-D01-01": PublicEHTReleaseSpec(
        release_code="2020-D01-01",
        repo_url="https://github.com/eventhorizontelescope/2020-D01-01.git",
        target_id="3C279",
        campaign_year=2017,
        pipeline_name="HOPS netcal Stokes I",
        file_glob="SR1_3C279_2017_*_hops_netcal_StokesI.csv",
        filename_regex=r"SR1_3C279_2017_(?P<day>\d{3})_(?P<band>hi|lo)_hops_netcal_StokesI\.csv$",
        dataset_description=(
            "Official public calibrated 2017 EHT 3C279 Stokes I release "
            "(2020-D01-01, HOPS netcal, hi/lo bands)."
        ),
    ),
    "2021-D03-01": PublicEHTReleaseSpec(
        release_code="2021-D03-01",
        repo_url="https://github.com/eventhorizontelescope/2021-D03-01.git",
        target_id="CenA",
        campaign_year=2017,
        pipeline_name="HOPS netcal Stokes I",
        file_glob="CenA_2017_*_hops_netcal_StokesI.csv",
        filename_regex=r"CenA_2017_(?P<day>\d{3})_(?P<band>hi|lo)_hops_netcal_StokesI\.csv$",
        dataset_description=(
            "Official public calibrated 2017 EHT Centaurus A Stokes I release "
            "(2021-D03-01, HOPS netcal, hi/lo bands)."
        ),
    ),
}

DEFAULT_PUBLIC_EHT_RELEASE_CODE = "2019-D01-01"
EHT_PUBLIC_M87_REPO = PUBLIC_EHT_RELEASE_SPECS[DEFAULT_PUBLIC_EHT_RELEASE_CODE].repo_url
DEFAULT_GLOB = PUBLIC_EHT_RELEASE_SPECS[DEFAULT_PUBLIC_EHT_RELEASE_CODE].file_glob

STATION_LAYOUT = {
    "AA": (-0.92, -0.35),
    "AP": (-0.84, -0.22),
    "AX": (-0.86, -0.28),
    "AZ": (-0.35, 0.58),
    "GL": (0.10, 0.96),
    "JC": (0.72, 0.58),
    "LM": (-0.08, 0.78),
    "MG": (0.34, 0.21),
    "MM": (0.40, 0.52),
    "PV": (0.88, -0.12),
    "SM": (0.62, 0.71),
    "SP": (-0.05, -0.98),
    "SW": (0.70, 0.67),
}


@dataclass(frozen=True)
class PublicEHTRecord:
    time_utc_hours: float
    station_a: str
    station_b: str
    u_lambda: float
    v_lambda: float
    visibility_jy: complex
    sigma_jy: float


@dataclass(frozen=True)
class PublicEHTSample:
    sample_id: str
    release_code: str
    source_name: str
    day_of_year: int
    band: str
    freq_ghz: float
    mjd: float
    source_file: str
    pipeline_name: str
    vis_real: np.ndarray
    vis_imag: np.ndarray
    vis_sigma: np.ndarray
    vis_weight: np.ndarray
    mask: np.ndarray
    dirty: np.ndarray
    uv_coords: np.ndarray
    baseline_pairs: np.ndarray
    frame_uv_indices: np.ndarray
    frame_uv_coords: np.ndarray
    station_labels: np.ndarray
    station_positions: np.ndarray
    visibility_scale_jy: float
    uv_scale_lambda: float
    frame_time_centers_utc: np.ndarray
    frame_counts: np.ndarray
    collision_count: int


def get_public_eht_release_spec(release_code: str) -> PublicEHTReleaseSpec:
    """Return the registered spec for one official public EHT release."""

    if release_code not in PUBLIC_EHT_RELEASE_SPECS:
        supported = ", ".join(sorted(PUBLIC_EHT_RELEASE_SPECS))
        raise KeyError(f"Unsupported public EHT release '{release_code}'. Supported releases: {supported}")
    return PUBLIC_EHT_RELEASE_SPECS[release_code]


def list_public_eht_release_specs() -> list[PublicEHTReleaseSpec]:
    """Return the supported public EHT release specs in a stable order."""

    return [PUBLIC_EHT_RELEASE_SPECS[key] for key in sorted(PUBLIC_EHT_RELEASE_SPECS)]


def ensure_public_eht_repo(
    *,
    destination: str | Path,
    source_dir: str | Path | None = None,
    release_code: str = DEFAULT_PUBLIC_EHT_RELEASE_CODE,
    repo_url: str | None = None,
) -> Path:
    """Return a local directory containing one supported official public EHT release."""

    destination = Path(destination)
    spec = get_public_eht_release_spec(release_code)
    resolved_repo_url = spec.repo_url if repo_url is None else repo_url
    if source_dir is not None:
        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Public EHT source directory not found: {source_dir}")
        if destination.resolve() != source_dir.resolve():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source_dir, destination)
        return destination

    if destination.exists() and (destination / "README.md").exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", resolved_repo_url, str(destination)],
        check=True,
    )
    return destination


def _parse_metadata_line(line: str) -> dict[str, str]:
    payload = line.lstrip("#").strip()
    parts = [item.strip() for item in payload.split(",") if item.strip()]
    metadata: dict[str, str] = {}
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _canonicalize_record(row: list[str]) -> PublicEHTRecord:
    time_utc_hours = float(row[0])
    station_a = row[1].strip()
    station_b = row[2].strip()
    u_lambda = float(row[3])
    v_lambda = float(row[4])
    amplitude = float(row[5])
    phase_deg = float(row[6])
    sigma_jy = float(row[7])
    visibility_jy = amplitude * np.exp(1j * np.deg2rad(phase_deg))

    canonical_pair = tuple(sorted((station_a, station_b)))
    if canonical_pair != (station_a, station_b):
        station_a, station_b = canonical_pair
        u_lambda = -u_lambda
        v_lambda = -v_lambda
        visibility_jy = np.conj(visibility_jy)

    return PublicEHTRecord(
        time_utc_hours=time_utc_hours,
        station_a=station_a,
        station_b=station_b,
        u_lambda=u_lambda,
        v_lambda=v_lambda,
        visibility_jy=complex(visibility_jy),
        sigma_jy=sigma_jy,
    )


def load_public_eht_csv(path: str | Path) -> tuple[dict[str, Any], list[PublicEHTRecord]]:
    """Load one official public EHT CSV measurement product."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if len(lines) < 3:
        raise ValueError(f"CSV file is too short to parse: {path}")
    metadata = _parse_metadata_line(lines[0])
    reader = csv.reader(lines[2:])
    records = [_canonicalize_record(row) for row in reader if row]
    return metadata, records


def _infer_release_spec_from_path(path: Path) -> PublicEHTReleaseSpec:
    for spec in list_public_eht_release_specs():
        if re.match(spec.filename_regex, path.name) is not None:
            return spec
    raise ValueError(f"Could not infer a supported public EHT release from file name: {path.name}")


def _dataset_id_from_path(path: Path, release_spec: PublicEHTReleaseSpec) -> tuple[int, str]:
    match = re.match(release_spec.filename_regex, path.name)
    if match is None:
        raise ValueError(
            f"Could not parse day/band from file name '{path.name}' for release '{release_spec.release_code}'."
        )
    return int(match.group("day")), match.group("band")


def _station_position(label: str) -> tuple[float, float]:
    if label in STATION_LAYOUT:
        return STATION_LAYOUT[label]
    seed = sum(ord(char) for char in label)
    angle = np.deg2rad(float(seed % 360))
    radius = 0.55 + 0.35 * float((seed % 17) / 16.0)
    return float(radius * np.cos(angle)), float(radius * np.sin(angle))


def _station_indices(records: list[PublicEHTRecord]) -> tuple[np.ndarray, dict[str, int]]:
    labels = np.asarray(
        sorted({record.station_a for record in records} | {record.station_b for record in records}),
        dtype="<U8",
    )
    return labels, {label: int(index) for index, label in enumerate(labels.tolist())}


def _station_positions(station_labels: np.ndarray) -> np.ndarray:
    return np.asarray([_station_position(str(label)) for label in station_labels.tolist()], dtype=np.float32)


def _baseline_pair_array(
    records: list[PublicEHTRecord],
    station_to_index: dict[str, int],
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    pairs = sorted(
        {
            tuple(sorted((station_to_index[record.station_a], station_to_index[record.station_b])))
            for record in records
        }
    )
    baseline_pairs = np.asarray(pairs, dtype=np.int32)
    pair_to_index = {pair: pair_index for pair_index, pair in enumerate(pairs)}
    return baseline_pairs, pair_to_index


def _bin_times(records: list[PublicEHTRecord], sequence_length: int) -> tuple[dict[float, int], np.ndarray]:
    unique_times = np.asarray(sorted({record.time_utc_hours for record in records}), dtype=np.float64)
    if unique_times.size == 0:
        raise ValueError("Cannot build time bins without any timestamps.")
    time_chunks = np.array_split(unique_times, sequence_length)
    time_to_frame: dict[float, int] = {}
    frame_centers = np.zeros(sequence_length, dtype=np.float32)
    last_time = float(unique_times[-1])
    for frame_index, chunk in enumerate(time_chunks):
        if chunk.size == 0:
            frame_centers[frame_index] = np.float32(last_time)
            continue
        frame_centers[frame_index] = np.float32(np.mean(chunk))
        for time_value in chunk.tolist():
            time_to_frame[float(time_value)] = frame_index
    for frame_index in range(sequence_length):
        if frame_centers[frame_index] == 0.0 and frame_index > 0:
            frame_centers[frame_index] = frame_centers[frame_index - 1]
    return time_to_frame, frame_centers


def _to_grid_index(coord: float, image_size: int) -> int:
    coord = float(np.clip(coord, -1.0, 1.0))
    scaled = 0.5 * (coord + 1.0) * float(image_size - 1)
    return int(np.rint(scaled))


def build_public_eht_sample(
    *,
    csv_path: str | Path,
    release_spec: PublicEHTReleaseSpec | None = None,
    image_size: int = 32,
    sequence_length: int = 8,
) -> PublicEHTSample:
    """Convert one supported official public EHT CSV file into one EMC sample."""

    csv_path = Path(csv_path)
    resolved_release_spec = release_spec or _infer_release_spec_from_path(csv_path)
    metadata, records = load_public_eht_csv(csv_path)
    if not records:
        raise ValueError(f"No records found in {csv_path}")

    day_of_year, band = _dataset_id_from_path(csv_path, resolved_release_spec)
    source_name = metadata.get("SRC", resolved_release_spec.target_id)
    mjd = float(metadata.get("DATE(MJD)", "0.0"))
    freq_text = metadata.get("FREQ", "0.0GHz")
    freq_match = re.match(r"([0-9.]+)", freq_text)
    freq_ghz = float(freq_match.group(1)) if freq_match else 0.0
    station_labels, station_to_index = _station_indices(records)
    station_positions = _station_positions(station_labels)
    baseline_pairs, pair_to_index = _baseline_pair_array(records, station_to_index)
    time_to_frame, frame_centers = _bin_times(records, sequence_length=sequence_length)

    amplitude_values = np.asarray([abs(record.visibility_jy) for record in records], dtype=np.float32)
    visibility_scale_jy = max(float(np.quantile(amplitude_values, 0.99)), 1e-6)
    uv_scale_lambda = max(
        float(max(np.sqrt(record.u_lambda**2 + record.v_lambda**2) for record in records)),
        1e-6,
    )

    baseline_count = int(baseline_pairs.shape[0])
    frame_vis = np.zeros((sequence_length, image_size, image_size), dtype=np.complex64)
    frame_weight_sum = np.zeros((sequence_length, image_size, image_size), dtype=np.float32)
    frame_sigma = np.zeros((sequence_length, image_size, image_size), dtype=np.float32)
    frame_norm_weight = np.zeros((sequence_length, image_size, image_size), dtype=np.float32)
    frame_mask = np.zeros((sequence_length, image_size, image_size), dtype=np.float32)
    frame_counts = np.zeros((sequence_length,), dtype=np.int32)
    frame_uv_indices = np.zeros((sequence_length, baseline_count, 2), dtype=np.int32)
    frame_uv_coords = np.zeros((sequence_length, baseline_count, 2), dtype=np.float32)
    collision_count = 0

    per_frame_pair_records: list[list[list[PublicEHTRecord]]] = [
        [[] for _ in range(baseline_count)] for _ in range(sequence_length)
    ]
    for record in records:
        frame_index = time_to_frame[float(record.time_utc_hours)]
        pair = tuple(sorted((station_to_index[record.station_a], station_to_index[record.station_b])))
        pair_index = pair_to_index[pair]
        per_frame_pair_records[frame_index][pair_index].append(record)

    for frame_index in range(sequence_length):
        occupied_cells: set[tuple[int, int]] = set()
        for pair_index in range(baseline_count):
            pair_records = per_frame_pair_records[frame_index][pair_index]
            if not pair_records:
                continue
            weights = np.asarray(
                [1.0 / max(record.sigma_jy**2, 1e-8) for record in pair_records],
                dtype=np.float64,
            )
            complex_values = np.asarray([record.visibility_jy for record in pair_records], dtype=np.complex128)
            u_values = np.asarray([record.u_lambda for record in pair_records], dtype=np.float64)
            v_values = np.asarray([record.v_lambda for record in pair_records], dtype=np.float64)
            weight_sum = max(float(weights.sum()), 1e-8)
            mean_visibility = complex(np.sum(weights * complex_values) / weight_sum)
            mean_u = float(np.sum(weights * u_values) / weight_sum)
            mean_v = float(np.sum(weights * v_values) / weight_sum)
            norm_u = mean_u / uv_scale_lambda
            norm_v = mean_v / uv_scale_lambda
            row = _to_grid_index(norm_v, image_size=image_size)
            col = _to_grid_index(norm_u, image_size=image_size)
            frame_uv_indices[frame_index, pair_index] = np.asarray([row, col], dtype=np.int32)
            frame_uv_coords[frame_index, pair_index] = np.asarray([norm_u, norm_v], dtype=np.float32)
            cell = (row, col)
            if cell in occupied_cells:
                collision_count += 1
            occupied_cells.add(cell)
            frame_vis[frame_index, row, col] += np.complex64(mean_visibility / visibility_scale_jy) * np.float32(weight_sum)
            frame_weight_sum[frame_index, row, col] += np.float32(weight_sum)
            frame_mask[frame_index, row, col] = 1.0
            frame_counts[frame_index] += len(pair_records)

        valid = frame_weight_sum[frame_index] > 0.0
        frame_vis[frame_index, valid] /= frame_weight_sum[frame_index, valid]
        frame_sigma[frame_index, valid] = (
            np.sqrt(1.0 / frame_weight_sum[frame_index, valid]) / np.float32(visibility_scale_jy)
        )
        frame_norm_weight[frame_index, valid] = 1.0 / np.maximum(frame_sigma[frame_index, valid] ** 2, 1e-8)

    dirty = dirty_image_reconstruction(measurements=frame_vis, mask=frame_mask)
    sample_id = f"{resolved_release_spec.target_id}_{resolved_release_spec.campaign_year}_{day_of_year:03d}_{band}"
    return PublicEHTSample(
        sample_id=sample_id,
        release_code=resolved_release_spec.release_code,
        source_name=source_name,
        day_of_year=day_of_year,
        band=band,
        freq_ghz=freq_ghz,
        mjd=mjd,
        source_file=csv_path.name,
        pipeline_name=resolved_release_spec.pipeline_name,
        vis_real=frame_vis.real.astype(np.float32),
        vis_imag=frame_vis.imag.astype(np.float32),
        vis_sigma=frame_sigma.astype(np.float32),
        vis_weight=frame_norm_weight.astype(np.float32),
        mask=frame_mask.astype(np.float32),
        dirty=dirty.astype(np.float32),
        uv_coords=build_temporal_uv_grid(image_size=image_size, sequence_length=sequence_length).astype(np.float32),
        baseline_pairs=baseline_pairs.astype(np.int32),
        frame_uv_indices=frame_uv_indices.astype(np.int32),
        frame_uv_coords=frame_uv_coords.astype(np.float32),
        station_labels=station_labels.astype("<U8"),
        station_positions=station_positions.astype(np.float32),
        visibility_scale_jy=visibility_scale_jy,
        uv_scale_lambda=uv_scale_lambda,
        frame_time_centers_utc=frame_centers.astype(np.float32),
        frame_counts=frame_counts.astype(np.int32),
        collision_count=int(collision_count),
    )


def _global_station_labels(samples: list[PublicEHTSample]) -> np.ndarray:
    labels = sorted({str(label) for sample in samples for label in sample.station_labels.tolist()})
    return np.asarray(labels, dtype="<U8")


def _global_baseline_pairs(
    *,
    samples: list[PublicEHTSample],
    station_index: dict[str, int],
) -> np.ndarray:
    pairs = sorted(
        {
            tuple(sorted((station_index[str(sample.station_labels[first])], station_index[str(sample.station_labels[second])])))
            for sample in samples
            for first, second in sample.baseline_pairs.tolist()
        }
    )
    return np.asarray(pairs, dtype=np.int32)


def _remap_sample_to_global_topology(
    *,
    sample: PublicEHTSample,
    global_station_index: dict[str, int],
    global_baseline_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_to_global = {
        int(local_index): int(global_station_index[str(label)])
        for local_index, label in enumerate(sample.station_labels.tolist())
    }
    global_pair_index = {
        tuple(pair.tolist()): int(pair_index)
        for pair_index, pair in enumerate(np.asarray(global_baseline_pairs, dtype=np.int64))
    }
    sequence_length = int(sample.frame_uv_indices.shape[0])
    baseline_count = int(global_baseline_pairs.shape[0])
    remapped_indices = np.zeros((sequence_length, baseline_count, 2), dtype=np.int32)
    remapped_coords = np.zeros((sequence_length, baseline_count, 2), dtype=np.float32)
    for local_pair_index, (first, second) in enumerate(sample.baseline_pairs.tolist()):
        global_pair = tuple(sorted((local_to_global[int(first)], local_to_global[int(second)])))
        pair_index = global_pair_index[global_pair]
        remapped_indices[:, pair_index] = sample.frame_uv_indices[:, local_pair_index]
        remapped_coords[:, pair_index] = sample.frame_uv_coords[:, local_pair_index]
    return remapped_indices, remapped_coords


def prepare_public_eht_validation_dataset(
    *,
    source_root: str | Path,
    output_dir: str | Path,
    release_code: str = DEFAULT_PUBLIC_EHT_RELEASE_CODE,
    image_size: int = 32,
    sequence_length: int = 8,
    file_glob: str | None = None,
) -> dict[str, Any]:
    """Convert one supported public EHT release into one evaluation-only EMC dataset."""

    release_spec = get_public_eht_release_spec(release_code)
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    csv_root = source_root / "csv"
    if not csv_root.exists():
        raise FileNotFoundError(f"Expected csv/ subdirectory under {source_root}")

    resolved_glob = release_spec.file_glob if file_glob is None else file_glob
    csv_paths = sorted(path for path in csv_root.glob(resolved_glob) if path.name.endswith(".csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matched {resolved_glob} under {csv_root}")

    samples = [
        build_public_eht_sample(
            csv_path=csv_path,
            release_spec=release_spec,
            image_size=image_size,
            sequence_length=sequence_length,
        )
        for csv_path in csv_paths
    ]

    station_labels = _global_station_labels(samples)
    station_index = {str(label): int(index) for index, label in enumerate(station_labels.tolist())}
    station_positions = _station_positions(station_labels)
    baseline_pairs = _global_baseline_pairs(samples=samples, station_index=station_index)
    uv_coords = samples[0].uv_coords
    remapped_topologies = [
        _remap_sample_to_global_topology(
            sample=sample,
            global_station_index=station_index,
            global_baseline_pairs=baseline_pairs,
        )
        for sample in samples
    ]

    arrays = {
        "sample_id": np.asarray([sample.sample_id for sample in samples], dtype="<U32"),
        "source_name": np.asarray([sample.source_name for sample in samples], dtype="<U16"),
        "release_code": np.asarray([release_spec.release_code for _ in samples], dtype="<U16"),
        "target_name": np.asarray([release_spec.target_id for _ in samples], dtype="<U16"),
        "campaign_year": np.asarray([release_spec.campaign_year for _ in samples], dtype=np.int32),
        "pipeline_name": np.asarray([sample.pipeline_name for sample in samples], dtype="<U24"),
        "source_file": np.asarray([sample.source_file for sample in samples], dtype="<U80"),
        "day_of_year": np.asarray([sample.day_of_year for sample in samples], dtype=np.int32),
        "band": np.asarray([sample.band for sample in samples], dtype="<U4"),
        "freq_ghz": np.asarray([sample.freq_ghz for sample in samples], dtype=np.float32),
        "mjd": np.asarray([sample.mjd for sample in samples], dtype=np.float32),
        "vis_real": np.stack([sample.vis_real for sample in samples]).astype(np.float32),
        "vis_imag": np.stack([sample.vis_imag for sample in samples]).astype(np.float32),
        "vis_sigma": np.stack([sample.vis_sigma for sample in samples]).astype(np.float32),
        "vis_weight": np.stack([sample.vis_weight for sample in samples]).astype(np.float32),
        "mask": np.stack([sample.mask for sample in samples]).astype(np.float32),
        "dirty": np.stack([sample.dirty for sample in samples]).astype(np.float32),
        "uv_coords": uv_coords.astype(np.float32),
        "baseline_pairs": baseline_pairs.astype(np.int32),
        "frame_uv_indices": np.stack([item[0] for item in remapped_topologies]).astype(np.int32),
        "frame_uv_coords": np.stack([item[1] for item in remapped_topologies]).astype(np.float32),
        "station_labels": station_labels.astype("<U8"),
        "station_positions": station_positions.astype(np.float32),
        "visibility_scale_jy": np.asarray([sample.visibility_scale_jy for sample in samples], dtype=np.float32),
        "uv_scale_lambda": np.asarray([sample.uv_scale_lambda for sample in samples], dtype=np.float32),
        "frame_time_centers_utc": np.stack([sample.frame_time_centers_utc for sample in samples]).astype(np.float32),
        "frame_counts": np.stack([sample.frame_counts for sample in samples]).astype(np.int32),
        "collision_count": np.asarray([sample.collision_count for sample in samples], dtype=np.int32),
    }

    save_npz(output_dir / "test.npz", arrays)
    manifest = {
        "source_root": str(source_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "release_code": release_spec.release_code,
        "target": release_spec.target_id,
        "campaign_year": release_spec.campaign_year,
        "pipeline": release_spec.pipeline_name,
        "source_repo": release_spec.repo_url,
        "file_glob": resolved_glob,
        "sample_count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "image_size": image_size,
        "sequence_length": sequence_length,
        "files": [path.name for path in csv_paths],
        "station_labels": station_labels.tolist(),
        "dataset_description": (
            f"{release_spec.dataset_description} Converted into time-binned real-measurement sequences "
            "for support-target held-out observation-domain validation."
        ),
        "sample_groups": [
            {
                "sample_id": sample.sample_id,
                "day_of_year": sample.day_of_year,
                "band": sample.band,
                "freq_ghz": sample.freq_ghz,
                "source_file": sample.source_file,
                "visibility_scale_jy": sample.visibility_scale_jy,
                "uv_scale_lambda": sample.uv_scale_lambda,
                "collision_count": sample.collision_count,
            }
            for sample in samples
        ],
    }
    save_json(output_dir / "real_data_manifest.json", manifest)
    return manifest


def prepare_public_m87_validation_dataset(
    *,
    source_root: str | Path,
    output_dir: str | Path,
    image_size: int = 32,
    sequence_length: int = 8,
    file_glob: str = DEFAULT_GLOB,
) -> dict[str, Any]:
    """Backward-compatible wrapper for the public 2019 M87 validation dataset."""

    return prepare_public_eht_validation_dataset(
        source_root=source_root,
        output_dir=output_dir,
        release_code=DEFAULT_PUBLIC_EHT_RELEASE_CODE,
        image_size=image_size,
        sequence_length=sequence_length,
        file_glob=file_glob,
    )
