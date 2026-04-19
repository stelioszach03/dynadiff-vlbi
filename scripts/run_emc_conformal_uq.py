#!/usr/bin/env python3
"""Calibrate and export conformal UQ summaries for EMC and EMC-TTO."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.data.measurement_holdout import build_structured_holdout_split  # noqa: E402
from dynadiff_vlbi.emc.uq import VLBIConformalUQ  # noqa: E402
from dynadiff_vlbi.evaluation.emc_protocol import _load_checkpoint_model, _predict_phase2  # noqa: E402
from dynadiff_vlbi.utils.config import _build_config  # noqa: E402
from dynadiff_vlbi.utils.device import get_device  # noqa: E402
from dynadiff_vlbi.utils.logging_utils import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-release-root", default="outputs/emc_benchmark_release")
    parser.add_argument("--public-root", default="outputs/public_eht_suite")
    parser.add_argument("--output-root", default="outputs/emc_conformal_uq")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_config_from_manifest(path: Path):
    payload = _load_json(path)
    return _build_config(payload, preset_name=str(payload.get("preset_name", "manifest")))


def _load_split_arrays(dataset_dir: Path, split_name: str) -> dict[str, np.ndarray]:
    with np.load(dataset_dir / f"{split_name}.npz") as payload:
        return {key: payload[key] for key in payload.files}


def _family_manifests(benchmark_release_root: Path) -> list[dict[str, Any]]:
    manifests = []
    for key in ("baseline_tracks", "scan_segments", "station_dropout"):
        manifests.append(_load_json(benchmark_release_root / "results_manifests" / f"{key}.json"))
    return manifests


def _public_manifests(public_root: Path) -> list[dict[str, Any]]:
    return [
        _load_json(path)
        for path in sorted((public_root / "results_manifests").glob("*.json"))
    ]


def _support_tags(manifest: dict[str, Any]) -> list[str]:
    return [f"{int(round(float(value) * 100.0)):02d}" for value in manifest["support_fractions"]]


def calibrate_quantiles(family_manifests: list[dict[str, Any]]) -> dict[str, VLBIConformalUQ]:
    device = get_device()
    calibrators: dict[str, VLBIConformalUQ] = {}
    for manifest in family_manifests:
        config = _load_config_from_manifest(ROOT / manifest["config_manifest_path"])
        dataset_dir = ROOT / manifest["dataset_dir"]
        dataset = _load_split_arrays(dataset_dir, "val")
        model, model_config = _load_checkpoint_model(ROOT / manifest["checkpoint_paths"]["emc"], device=device)
        uv_coords = dataset["uv_coords"]
        baseline_pairs = dataset.get("baseline_pairs")
        frame_uv_indices = dataset.get("frame_uv_indices")
        frame_uv_coords = dataset.get("frame_uv_coords")
        for fraction in config.holdout.eval_support_fractions:
            support_tag = f"{int(round(float(fraction) * 100.0)):02d}"
            calibrator = calibrators.setdefault(support_tag, VLBIConformalUQ(alpha=0.1))
            for sample_index in range(dataset["ground_truth"].shape[0]):
                measurements = (
                    dataset["vis_real"][sample_index] + 1j * dataset["vis_imag"][sample_index]
                ).astype(np.complex64)
                split = build_structured_holdout_split(
                    measurements=measurements,
                    observed_mask=dataset["mask"][sample_index].astype(np.float32),
                    frame_uv_indices=frame_uv_indices,
                    frame_uv_coords=frame_uv_coords,
                    baseline_pairs=baseline_pairs,
                    station_positions=dataset.get("station_positions"),
                    base_seed=config.project.seed,
                    sample_index=sample_index,
                    support_fraction=float(fraction),
                    strategy=config.holdout.strategy,
                )
                prediction = _predict_phase2(
                    model=model,
                    model_config=model_config,
                    support_vis_real=(measurements.real * split.support_mask).astype(np.float32),
                    support_vis_imag=(measurements.imag * split.support_mask).astype(np.float32),
                    support_mask=split.support_mask,
                    support_dirty=split.support_dirty.astype(np.float32),
                    uv_coords=uv_coords,
                    frame_uv_coords=frame_uv_coords,
                    frame_uv_indices=frame_uv_indices,
                    measurements=measurements,
                    device=device,
                )["mean"]
                calibrator.calibrate(
                    predictions=torch.from_numpy(prediction.astype(np.float32)),
                    support_vis=torch.from_numpy(split.support_measurements.astype(np.complex64)),
                    support_mask=torch.from_numpy(split.support_mask.astype(np.float32)),
                    target_vis=torch.from_numpy(split.target_measurements.astype(np.complex64)),
                    target_mask=torch.from_numpy(split.target_mask.astype(np.float32)),
                )
    return calibrators


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_synthetic_uq_rows(
    family_manifests: list[dict[str, Any]],
    calibrators: dict[str, VLBIConformalUQ],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for manifest in family_manifests:
        summary = _load_json(ROOT / manifest["summary_path"])
        protocol_dir = ROOT / manifest["protocol_output_dir"]
        for support_tag in _support_tags(manifest):
            calibrator = calibrators[support_tag]
            with np.load(protocol_dir / "predictions" / f"support_{support_tag}.npz") as bundle:
                ground_truth = bundle["ground_truth"].astype(np.float32)
                emc_predictions = bundle["emc"].astype(np.float32)
            coverage_values: list[float] = []
            interval_width_values: list[float] = []
            for sample_index in range(emc_predictions.shape[0]):
                report = calibrator.coverage_width_report(
                    predictions=emc_predictions[sample_index],
                    ground_truth=ground_truth[sample_index],
                )
                coverage_values.append(report["coverage"])
                interval_width_values.append(report["mean_interval_width"])
                per_sample_rows.append(
                    {
                        "dataset": manifest["condition_key"],
                        "family": manifest["condition_key"],
                        "track_label": summary.get("holdout", {}).get("label", manifest["condition_key"]),
                        "support_fraction_tag": support_tag,
                        "sample_index": sample_index,
                        "model": "emc",
                        "coverage": report["coverage"],
                        "mean_interval_width": report["mean_interval_width"],
                        "q_hat": report["q_hat"],
                    }
                )
            summary_rows.append(
                {
                    "family": manifest["condition_key"],
                    "support_fraction_tag": support_tag,
                    "emc_coverage_90": float(np.mean(coverage_values)),
                    "emc_miw": float(np.mean(interval_width_values)),
                    "q_hat": float(calibrator.q_hat()),
                }
            )
    return per_sample_rows, summary_rows


def build_public_uq_rows(
    public_manifests: list[dict[str, Any]],
    calibrators: dict[str, VLBIConformalUQ],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for manifest in public_manifests:
        summary = _load_json(ROOT / manifest["summary_path"])
        run_dir = ROOT / manifest["output_dir"]
        for support_tag in _support_tags(manifest):
            with np.load(run_dir / "predictions" / f"support_{support_tag}.npz") as bundle:
                for model_key in ("emc", "emc_tto"):
                    if model_key not in bundle.files:
                        continue
                    prediction_array = bundle[model_key].astype(np.float32)
                    interval_width = float(calibrators[support_tag].interval_width())
                    for sample_index in range(prediction_array.shape[0]):
                        per_sample_rows.append(
                            {
                                "family": manifest["family"],
                                "release_code": manifest["release_code"],
                                "track_label": f"{summary['target']} {summary['campaign_year']} ({summary['release_code']})",
                                "support_fraction_tag": support_tag,
                                "sample_index": sample_index,
                                "model": model_key,
                                "mean_interval_width": interval_width,
                                "q_hat": float(calibrators[support_tag].q_hat()),
                            }
                        )
                    summary_rows.append(
                        {
                            "family": manifest["family"],
                            "release_code": manifest["release_code"],
                            "track_label": f"{summary['target']} {summary['campaign_year']} ({summary['release_code']})",
                            "support_fraction_tag": support_tag,
                            "model": model_key,
                            "mean_interval_width": interval_width,
                            "q_hat": float(calibrators[support_tag].q_hat()),
                        }
                    )
    return per_sample_rows, summary_rows


def main() -> int:
    args = parse_args()
    benchmark_release_root = (ROOT / args.benchmark_release_root).resolve()
    public_root = (ROOT / args.public_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    tables_dir = output_root / "tables"
    per_sample_dir = output_root / "per_sample"
    tables_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    family_manifests = _family_manifests(benchmark_release_root)
    public_manifests = _public_manifests(public_root)
    calibrators = calibrate_quantiles(family_manifests)

    synthetic_per_sample, synthetic_summary = build_synthetic_uq_rows(family_manifests, calibrators)
    public_per_sample, public_summary = build_public_uq_rows(public_manifests, calibrators)

    _write_csv(per_sample_dir / "synthetic_emc_uq.csv", synthetic_per_sample)
    _write_csv(per_sample_dir / "public_emc_uq.csv", public_per_sample)
    _write_csv(tables_dir / "synthetic_emc_conformal_uq.csv", synthetic_summary)
    _write_csv(tables_dir / "public_emc_conformal_uq.csv", public_summary)
    save_json(
        output_root / "conformal_uq_manifest.json",
        {
            "calibration": {tag: calibrator.to_dict() for tag, calibrator in calibrators.items()},
            "synthetic_summary_csv": str((tables_dir / "synthetic_emc_conformal_uq.csv").relative_to(ROOT)),
            "public_summary_csv": str((tables_dir / "public_emc_conformal_uq.csv").relative_to(ROOT)),
            "synthetic_per_sample_csv": str((per_sample_dir / "synthetic_emc_uq.csv").relative_to(ROOT)),
            "public_per_sample_csv": str((per_sample_dir / "public_emc_uq.csv").relative_to(ROOT)),
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "calibration_tags": sorted(calibrators),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
