#!/usr/bin/env python3
"""Run the benchmark-first EMC protocol and export deterministic release manifests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.benchmark_release import (  # noqa: E402
    export_split_manifests,
    write_benchmark_output_manifest,
)
from dynadiff_vlbi.evaluation.emc_protocol import ComparatorSpec, evaluate_emc_condition, load_comparators  # noqa: E402
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, load_experiment_config  # noqa: E402
from dynadiff_vlbi.utils.device import get_device  # noqa: E402
from dynadiff_vlbi.utils.logging_utils import save_json  # noqa: E402
from dynadiff_vlbi.utils.seed import set_seed  # noqa: E402


@dataclass(frozen=True)
class BenchmarkConditionSpec:
    """One benchmark condition or realism track."""

    key: str
    title: str
    kind: str
    base_config: str
    preset: str
    dataset_dir: str
    dataset_generation_config: str
    baseline_run_name: str
    visibility_run_name: str
    residual_run_name: str
    ccrr_run_name: str
    emc_run_name: str
    protocol_run_name: str
    visibility_config: str
    residual_config: str
    ccrr_config: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="all",
        choices=[
            "all",
            "families",
            "baseline_tracks",
            "scan_segments",
            "station_dropout",
            "realism",
            "challenge_inspired_realism",
        ],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--data-root", default="data/generated")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dps-checkpoint", default=None)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(f"\n[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _resolve_output_path(output_root: Path, run_name: str) -> Path:
    return output_root / run_name / "checkpoints" / "best.pt"


def _repo_relative(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        return resolved.as_posix()
    try:
        return resolved.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _ensure_dataset(
    *,
    python_bin: str,
    base_config: str,
    preset: str,
    dataset_dir: Path,
    skip_existing: bool,
) -> None:
    if skip_existing and (dataset_dir / "train.npz").exists():
        return
    if (dataset_dir / "train.npz").exists():
        return
    _run(
        [
            python_bin,
            "scripts/generate_toy_dataset.py",
            "--base-config",
            base_config,
            "--preset",
            preset,
            "--output-dir",
            str(dataset_dir),
        ]
    )


def _ensure_train_run(
    *,
    python_bin: str,
    run_name: str,
    output_root: Path,
    data_dir: Path,
    preset: str,
    skip_existing: bool,
    base_config: str | None = None,
    backbone_checkpoint: Path | None = None,
) -> Path:
    checkpoint_path = _resolve_output_path(output_root, run_name)
    if skip_existing and checkpoint_path.exists():
        return checkpoint_path
    if checkpoint_path.exists():
        return checkpoint_path
    command = [
        python_bin,
        "scripts/train_baseline.py",
        "--preset",
        preset,
        "--data-dir",
        str(data_dir),
        "--run-name",
        run_name,
    ]
    if base_config is not None:
        command.extend(["--base-config", base_config])
    if backbone_checkpoint is not None:
        command.extend(["--backbone-checkpoint", str(backbone_checkpoint)])
    _run(command)
    return checkpoint_path


def _load_config(base_config: str, preset: str):
    config = load_experiment_config(
        base_path=ROOT / base_config,
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset=preset,
        default_base_path=ROOT / DEFAULT_BASE_CONFIG_PATH,
    )
    set_seed(config.project.seed)
    return config


def _protocol_summary_path(output_root: Path, run_name: str) -> Path:
    return output_root / run_name / "logs" / "emc_protocol_summary.json"


def _write_results_manifest(
    *,
    manifest_path: Path,
    payload: dict[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(manifest_path, payload)


def _run_condition(
    *,
    spec: BenchmarkConditionSpec,
    output_root: Path,
    data_root: Path,
    python_bin: str,
    skip_existing: bool,
    release_root: Path,
    dps_checkpoint: Path | None = None,
) -> dict[str, Any]:
    dataset_dir = data_root / spec.dataset_dir
    protocol_output_dir = output_root / spec.protocol_run_name
    summary_path = _protocol_summary_path(output_root, spec.protocol_run_name)
    config = _load_config(spec.base_config, spec.preset)

    _ensure_dataset(
        python_bin=python_bin,
        base_config=spec.dataset_generation_config,
        preset=spec.preset,
        dataset_dir=dataset_dir,
        skip_existing=skip_existing,
    )

    baseline_checkpoint = _ensure_train_run(
        python_bin=python_bin,
        run_name=spec.baseline_run_name,
        output_root=output_root,
        data_dir=dataset_dir,
        preset=spec.preset,
        skip_existing=skip_existing,
    )
    residual_checkpoint = _ensure_train_run(
        python_bin=python_bin,
        base_config=spec.residual_config,
        run_name=spec.residual_run_name,
        output_root=output_root,
        data_dir=dataset_dir,
        preset=spec.preset,
        skip_existing=skip_existing,
        backbone_checkpoint=baseline_checkpoint,
    )
    visibility_checkpoint = _ensure_train_run(
        python_bin=python_bin,
        base_config=spec.visibility_config,
        run_name=spec.visibility_run_name,
        output_root=output_root,
        data_dir=dataset_dir,
        preset=spec.preset,
        skip_existing=skip_existing,
        backbone_checkpoint=baseline_checkpoint,
    )
    ccrr_checkpoint = _ensure_train_run(
        python_bin=python_bin,
        base_config=spec.ccrr_config,
        run_name=spec.ccrr_run_name,
        output_root=output_root,
        data_dir=dataset_dir,
        preset=spec.preset,
        skip_existing=skip_existing,
        backbone_checkpoint=baseline_checkpoint,
    )
    emc_checkpoint = _ensure_train_run(
        python_bin=python_bin,
        base_config=spec.base_config,
        run_name=spec.emc_run_name,
        output_root=output_root,
        data_dir=dataset_dir,
        preset=spec.preset,
        skip_existing=skip_existing,
        backbone_checkpoint=baseline_checkpoint,
    )

    if skip_existing and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        comparator_specs = [
            ComparatorSpec("dirty", "Dirty", "dirty"),
            ComparatorSpec("tikhonov", "Tikhonov", "tikhonov"),
            ComparatorSpec("baseline_learned", "Baseline 3D U-Net", "baseline", baseline_checkpoint),
            ComparatorSpec("residual_refinement", "Residual Refinement", "phase2", residual_checkpoint),
            ComparatorSpec("ccrr", "CCRR", "phase2", ccrr_checkpoint),
            ComparatorSpec("emc", "EMC", "phase2", emc_checkpoint),
        ]
        if dps_checkpoint is not None and dps_checkpoint.exists():
            comparator_specs.append(ComparatorSpec("dps", "DPS", "dps", dps_checkpoint))
        comparators = load_comparators(comparator_specs, device=get_device())
        summary = evaluate_emc_condition(
            config=config,
            dataset_dir=dataset_dir,
            comparators=comparators,
            output_dir=protocol_output_dir,
            support_fractions=config.holdout.eval_support_fractions,
        )

    split_manifest_dir = release_root / "split_manifests" / spec.key
    split_manifest = export_split_manifests(
        config=config,
        dataset_dir=dataset_dir,
        output_dir=split_manifest_dir,
    )
    config_manifest_path = release_root / "config_manifests" / f"{spec.key}.json"
    config_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(config_manifest_path, config.to_dict())

    condition_manifest = {
        "key": spec.key,
        "title": spec.title,
        "kind": spec.kind,
        "notes": spec.notes,
        "seed": int(config.project.seed),
        "base_config": spec.base_config,
        "dataset_generation_config": spec.dataset_generation_config,
        "preset": spec.preset,
        "dataset_dir": _repo_relative(dataset_dir),
        "protocol_output_dir": _repo_relative(protocol_output_dir),
        "summary_path": _repo_relative(summary_path),
        "split_manifest_path": _repo_relative(split_manifest_dir / "split_manifest.json"),
        "config_manifest_path": _repo_relative(config_manifest_path),
        "checkpoints": {
            "baseline": _repo_relative(baseline_checkpoint),
            "visibility_conditioned": _repo_relative(visibility_checkpoint),
            "residual_refinement": _repo_relative(residual_checkpoint),
            "ccrr": _repo_relative(ccrr_checkpoint),
            "emc": _repo_relative(emc_checkpoint),
            **(
                {"dps": _repo_relative(dps_checkpoint)}
                if dps_checkpoint is not None and dps_checkpoint.exists()
                else {}
            ),
        },
        "expected_protocol_files": {
            "summary_json": _repo_relative(summary_path),
            "metrics_csv": _repo_relative(protocol_output_dir / "logs" / "support_fraction_metrics.csv"),
            "support_fraction_predictions": {
                f"{int(round(float(fraction) * 100.0)):02d}": _repo_relative(
                    protocol_output_dir / "predictions" / f"support_{int(round(float(fraction) * 100.0)):02d}.npz"
                )
                for fraction in config.holdout.eval_support_fractions
            },
        },
        "split_manifest": split_manifest,
    }
    results_manifest = {
        "kind": "synthetic_benchmark_condition",
        "condition_key": spec.key,
        "condition_title": spec.title,
        "condition_kind": spec.kind,
        "seed": int(config.project.seed),
        "support_fractions": [float(value) for value in config.holdout.eval_support_fractions],
        "dataset_dir": _repo_relative(dataset_dir),
        "protocol_output_dir": _repo_relative(protocol_output_dir),
        "summary_path": _repo_relative(summary_path),
        "config_manifest_path": _repo_relative(config_manifest_path),
        "split_manifest_path": _repo_relative(split_manifest_dir / "split_manifest.json"),
        "checkpoint_paths": condition_manifest["checkpoints"],
        "metrics_csv": condition_manifest["expected_protocol_files"]["metrics_csv"],
        "prediction_paths": condition_manifest["expected_protocol_files"]["support_fraction_predictions"],
    }
    protocol_results_manifest_path = protocol_output_dir / "logs" / "results_manifest.json"
    release_results_manifest_path = release_root / "results_manifests" / f"{spec.key}.json"
    _write_results_manifest(manifest_path=protocol_results_manifest_path, payload=results_manifest)
    _write_results_manifest(manifest_path=release_results_manifest_path, payload=results_manifest)
    condition_manifest["results_manifest_path"] = _repo_relative(release_results_manifest_path)
    return condition_manifest


def _benchmark_specs() -> list[BenchmarkConditionSpec]:
    return [
        BenchmarkConditionSpec(
            key="baseline_tracks",
            title="Default64 baseline-track holdout benchmark",
            kind="benchmark_family",
            base_config="configs/emc_benchmark_baseline_tracks_default64.yaml",
            preset="default64",
            dataset_dir="ccrr_default64_seed7_shared",
            dataset_generation_config="configs/emc_benchmark_baseline_tracks_default64.yaml",
            baseline_run_name="ccrr_default64_seed7_baseline_ref",
            visibility_run_name="ccrr_default64_seed7_visibility_ref",
            residual_run_name="ccrr_default64_seed7_residual_ref",
            ccrr_run_name="ccrr_default64_seed7_main",
            emc_run_name="emc_benchmark_baseline_tracks_default64_main_noclosure",
            protocol_run_name="emc_benchmark_baseline_tracks_default64_protocol",
            visibility_config="configs/phase2_visibility_default64.yaml",
            residual_config="configs/phase2_residual_refine_default64.yaml",
            ccrr_config="configs/ccrr_default64.yaml",
            notes="Central benchmark family: deterministic contiguous baseline-track blocks across time.",
        ),
        BenchmarkConditionSpec(
            key="scan_segments",
            title="Default64 scan-segment holdout benchmark",
            kind="benchmark_family",
            base_config="configs/emc_benchmark_scan_segments_default64.yaml",
            preset="default64",
            dataset_dir="ccrr_default64_seed7_shared",
            dataset_generation_config="configs/emc_benchmark_baseline_tracks_default64.yaml",
            baseline_run_name="ccrr_default64_seed7_baseline_ref",
            visibility_run_name="ccrr_default64_seed7_visibility_ref",
            residual_run_name="ccrr_default64_seed7_residual_ref",
            ccrr_run_name="ccrr_default64_seed7_main",
            emc_run_name="emc_benchmark_scan_segments_default64_main_noclosure",
            protocol_run_name="emc_benchmark_scan_segments_default64_protocol",
            visibility_config="configs/phase2_visibility_default64.yaml",
            residual_config="configs/phase2_residual_refine_default64.yaml",
            ccrr_config="configs/ccrr_default64.yaml",
            notes="Temporal benchmark family: deterministic contiguous scan-like frame removal with DC retained at the origin.",
        ),
        BenchmarkConditionSpec(
            key="station_dropout",
            title="Default64 station-dropout holdout benchmark",
            kind="benchmark_family",
            base_config="configs/emc_benchmark_station_dropout_default64.yaml",
            preset="default64",
            dataset_dir="ccrr_default64_seed7_shared",
            dataset_generation_config="configs/emc_benchmark_baseline_tracks_default64.yaml",
            baseline_run_name="ccrr_default64_seed7_baseline_ref",
            visibility_run_name="ccrr_default64_seed7_visibility_ref",
            residual_run_name="ccrr_default64_seed7_residual_ref",
            ccrr_run_name="ccrr_default64_seed7_main",
            emc_run_name="emc_benchmark_station_dropout_default64_main_noclosure",
            protocol_run_name="emc_benchmark_station_dropout_default64_protocol",
            visibility_config="configs/phase2_visibility_default64.yaml",
            residual_config="configs/phase2_residual_refine_default64.yaml",
            ccrr_config="configs/ccrr_default64.yaml",
            notes="Station-structured benchmark family: deterministic subsets of stations are withheld together with all incident baselines.",
        ),
        BenchmarkConditionSpec(
            key="challenge_inspired_realism",
            title="Challenge-inspired realism track",
            kind="challenge_inspired_realism",
            base_config="configs/emc_benchmark_challenge_inspired_realism_default64.yaml",
            preset="default64",
            dataset_dir="ccrr_realism_bridge2_default64_shared",
            dataset_generation_config="configs/emc_benchmark_challenge_inspired_realism_default64.yaml",
            baseline_run_name="ccrr_realism_bridge2_default64_baseline_ref",
            visibility_run_name="ccrr_realism_bridge2_default64_visibility_ref",
            residual_run_name="ccrr_realism_bridge2_default64_residual_ref",
            ccrr_run_name="ccrr_realism_bridge2_default64_main",
            emc_run_name="emc_benchmark_challenge_inspired_realism_default64_main_noclosure",
            protocol_run_name="emc_benchmark_challenge_inspired_realism_default64_protocol",
            visibility_config="configs/phase2_visibility_default64.yaml",
            residual_config="configs/phase2_residual_refine_default64.yaml",
            ccrr_config="configs/ccrr_realism_bridge2_default64.yaml",
            notes=(
                "Challenge-inspired realism track built only from public-style ingredients already supported in the repository: "
                "station-track sampling, scan gaps, baseline-dependent noise heterogeneity, and station gain corruption. "
                "This is not the private ngEHT Challenge #2 dataset."
            ),
        ),
    ]


def main() -> int:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    data_root = (ROOT / args.data_root).resolve()
    release_root = (output_root / "emc_benchmark_release").resolve()
    release_root.mkdir(parents=True, exist_ok=True)

    specs = _benchmark_specs()
    target_to_keys = {
        "all": {spec.key for spec in specs},
        "families": {"baseline_tracks", "scan_segments", "station_dropout"},
        "baseline_tracks": {"baseline_tracks"},
        "scan_segments": {"scan_segments"},
        "station_dropout": {"station_dropout"},
        "realism": {"challenge_inspired_realism"},
        "challenge_inspired_realism": {"challenge_inspired_realism"},
    }
    selected_keys = target_to_keys[args.target]

    manifests: dict[str, Any] = {}
    for spec in specs:
        if spec.key not in selected_keys:
            continue
        manifests[spec.key] = _run_condition(
            spec=spec,
            output_root=output_root,
            data_root=data_root,
            python_bin=args.python,
            skip_existing=args.skip_existing,
            release_root=release_root,
            dps_checkpoint=((ROOT / args.dps_checkpoint).resolve() if args.dps_checkpoint else None),
        )

    release_manifest = {
        "name": "EMC benchmark release",
        "benchmark_claim": (
            "EMC is proposed here as a reproducible benchmark protocol for earned versus enforced "
            "measurement consistency under structured sparse dynamic VLBI-inspired sampling."
        ),
        "one_command_reproduction": (
            f"{args.python} scripts/run_emc_benchmark.py --target all --skip-existing && "
            f"{args.python} scripts/generate_emc_benchmark_artifacts.py"
        ),
        "deterministic_factors": {
            "project_seed": 7,
            "support_fractions": [0.8, 0.6, 0.4, 0.2],
            "holdout_families": [
                "baseline_track_blocks",
                "scan_segment_blocks",
                "station_dropout",
            ],
        },
        "results_manifest_dir": _repo_relative(release_root / "results_manifests"),
        "conditions": manifests,
    }
    write_benchmark_output_manifest(
        output_path=release_root / "benchmark_output_manifest.json",
        payload=release_manifest,
    )
    print(json.dumps({"completed_conditions": list(manifests.keys()), "release_root": str(release_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
