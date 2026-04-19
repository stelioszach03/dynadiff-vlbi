#!/usr/bin/env python3
"""Run the reproducible CCRR comparison protocol on shared datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="all",
        choices=["all", "seed_repeats", "conditions", "ablations"],
        help="Which part of the CCRR protocol to run.",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--data-root", default="data/generated")
    parser.add_argument("--preset-default32", default="default32")
    parser.add_argument("--preset-exp64", default="exp64")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seeds", default="7,19,31,43,59")
    return parser.parse_args()


def _run(command: list[str], cwd: Path) -> None:
    print(f"\n[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _write_seeded_config(source_path: str | Path, output_path: Path, seed: int) -> Path:
    payload = yaml.safe_load(Path(source_path).read_text(encoding="utf-8")) or {}
    project = dict(payload.get("project", {}))
    project["seed"] = int(seed)
    payload["project"] = project
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def _needs_file(path: Path, skip_existing: bool) -> bool:
    return not (skip_existing and path.exists())


def _train_run_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def _checkpoint_path(output_root: Path, run_name: str) -> Path:
    return output_root / run_name / "checkpoints" / "best.pt"


def _summary_path(output_root: Path, run_name: str) -> Path:
    return output_root / run_name / "logs" / "evaluation_summary.json"


def _seed_protocol(
    *,
    python_bin: str,
    seed: int,
    preset: str,
    output_root: Path,
    data_root: Path,
    skip_existing: bool,
) -> None:
    temp_dir = output_root / "ccrr_protocol" / "temp_configs"
    ccrr_override = _write_seeded_config("configs/ccrr_default32.yaml", temp_dir / f"ccrr_seed_{seed}.yaml", seed)
    visibility_override = _write_seeded_config(
        "configs/phase2_visibility_default32.yaml",
        temp_dir / f"phase2_visibility_seed_{seed}.yaml",
        seed,
    )
    residual_override = _write_seeded_config(
        "configs/phase2_residual_refine_default32.yaml",
        temp_dir / f"phase2_residual_seed_{seed}.yaml",
        seed,
    )

    dataset_dir = data_root / f"ccrr_default32_seed{seed}_shared"
    run_prefix = f"ccrr_seed{seed}"

    if _needs_file(dataset_dir / "test.npz", skip_existing):
        _run(
            [
                python_bin,
                "scripts/generate_toy_dataset.py",
                "--base-config",
                str(ccrr_override),
                "--preset",
                preset,
                "--output-dir",
                str(dataset_dir),
            ],
            cwd=ROOT,
        )

    baseline_run = _train_run_name(run_prefix, "baseline_ref")
    baseline_ckpt = _checkpoint_path(output_root, baseline_run)
    if _needs_file(baseline_ckpt, skip_existing):
        _run(
            [
                python_bin,
                "scripts/train_baseline.py",
                "--base-config",
                str(ccrr_override),
                "--preset",
                preset,
                "--model-type",
                "baseline",
                "--data-dir",
                str(dataset_dir),
                "--run-name",
                baseline_run,
            ],
            cwd=ROOT,
        )

    visibility_run = _train_run_name(run_prefix, "visibility_ref")
    visibility_ckpt = _checkpoint_path(output_root, visibility_run)
    if _needs_file(visibility_ckpt, skip_existing):
        _run(
            [
                python_bin,
                "scripts/train_baseline.py",
                "--base-config",
                str(visibility_override),
                "--preset",
                preset,
                "--data-dir",
                str(dataset_dir),
                "--run-name",
                visibility_run,
            ],
            cwd=ROOT,
        )

    residual_run = _train_run_name(run_prefix, "residual_ref")
    residual_ckpt = _checkpoint_path(output_root, residual_run)
    if _needs_file(residual_ckpt, skip_existing):
        _run(
            [
                python_bin,
                "scripts/train_baseline.py",
                "--base-config",
                str(residual_override),
                "--preset",
                preset,
                "--data-dir",
                str(dataset_dir),
                "--run-name",
                residual_run,
                "--backbone-checkpoint",
                str(baseline_ckpt),
            ],
            cwd=ROOT,
        )

    ccrr_run = _train_run_name(run_prefix, "main")
    ccrr_ckpt = _checkpoint_path(output_root, ccrr_run)
    if _needs_file(ccrr_ckpt, skip_existing):
        _run(
            [
                python_bin,
                "scripts/train_baseline.py",
                "--base-config",
                str(ccrr_override),
                "--preset",
                preset,
                "--data-dir",
                str(dataset_dir),
                "--run-name",
                ccrr_run,
                "--backbone-checkpoint",
                str(baseline_ckpt),
            ],
            cwd=ROOT,
        )

    summary_path = _summary_path(output_root, ccrr_run)
    if _needs_file(summary_path, skip_existing):
        _run(
            [
                python_bin,
                "scripts/evaluate_model.py",
                "--base-config",
                str(ccrr_override),
                "--preset",
                preset,
                "--data-dir",
                str(dataset_dir),
                "--run-name",
                ccrr_run,
                "--reference-baseline-checkpoint",
                str(baseline_ckpt),
                "--reference-visibility-checkpoint",
                str(visibility_ckpt),
                "--reference-residual-checkpoint",
                str(residual_ckpt),
            ],
            cwd=ROOT,
        )


def _condition_protocol(
    *,
    python_bin: str,
    name: str,
    ccrr_config: str,
    visibility_config: str,
    residual_config: str,
    preset: str,
    output_root: Path,
    data_root: Path,
    skip_existing: bool,
) -> None:
    dataset_dir = data_root / f"{name}_shared"
    baseline_run = f"{name}_baseline_ref"
    visibility_run = f"{name}_visibility_ref"
    residual_run = f"{name}_residual_ref"
    ccrr_run = f"{name}_main"

    if _needs_file(dataset_dir / "test.npz", skip_existing):
        _run(
            [
                python_bin,
                "scripts/generate_toy_dataset.py",
                "--base-config",
                ccrr_config,
                "--preset",
                preset,
                "--output-dir",
                str(dataset_dir),
            ],
            cwd=ROOT,
        )

    baseline_ckpt = _checkpoint_path(output_root, baseline_run)
    if _needs_file(baseline_ckpt, skip_existing):
        _run(
            [
                python_bin,
                "scripts/train_baseline.py",
                "--base-config",
                ccrr_config,
                "--preset",
                preset,
                "--model-type",
                "baseline",
                "--data-dir",
                str(dataset_dir),
                "--run-name",
                baseline_run,
            ],
            cwd=ROOT,
        )

    visibility_ckpt = _checkpoint_path(output_root, visibility_run)
    if _needs_file(visibility_ckpt, skip_existing):
        _run(
            [
                python_bin,
                "scripts/train_baseline.py",
                "--base-config",
                visibility_config,
                "--preset",
                preset,
                "--data-dir",
                str(dataset_dir),
                "--run-name",
                visibility_run,
            ],
            cwd=ROOT,
        )

    residual_ckpt = _checkpoint_path(output_root, residual_run)
    if _needs_file(residual_ckpt, skip_existing):
        _run(
            [
                python_bin,
                "scripts/train_baseline.py",
                "--base-config",
                residual_config,
                "--preset",
                preset,
                "--data-dir",
                str(dataset_dir),
                "--run-name",
                residual_run,
                "--backbone-checkpoint",
                str(baseline_ckpt),
            ],
            cwd=ROOT,
        )

    ccrr_ckpt = _checkpoint_path(output_root, ccrr_run)
    if _needs_file(ccrr_ckpt, skip_existing):
        _run(
            [
                python_bin,
                "scripts/train_baseline.py",
                "--base-config",
                ccrr_config,
                "--preset",
                preset,
                "--data-dir",
                str(dataset_dir),
                "--run-name",
                ccrr_run,
                "--backbone-checkpoint",
                str(baseline_ckpt),
            ],
            cwd=ROOT,
        )

    summary_path = _summary_path(output_root, ccrr_run)
    if _needs_file(summary_path, skip_existing):
        _run(
            [
                python_bin,
                "scripts/evaluate_model.py",
                "--base-config",
                ccrr_config,
                "--preset",
                preset,
                "--data-dir",
                str(dataset_dir),
                "--run-name",
                ccrr_run,
                "--reference-baseline-checkpoint",
                str(baseline_ckpt),
                "--reference-visibility-checkpoint",
                str(visibility_ckpt),
                "--reference-residual-checkpoint",
                str(residual_ckpt),
            ],
            cwd=ROOT,
        )


def _ablation_protocol(
    *,
    python_bin: str,
    output_root: Path,
    data_root: Path,
    skip_existing: bool,
) -> None:
    dataset_dir = data_root / "ccrr_default32_seed7_shared"
    baseline_ckpt = _checkpoint_path(output_root, "ccrr_seed7_baseline_ref")
    visibility_ckpt = _checkpoint_path(output_root, "ccrr_seed7_visibility_ref")
    residual_ckpt = _checkpoint_path(output_root, "ccrr_seed7_residual_ref")
    ablations = {
        "ccrr_ablation_no_dc": "configs/ccrr_ablation_no_dc_default32.yaml",
        "ccrr_ablation_no_closure": "configs/ccrr_ablation_no_closure_default32.yaml",
        "ccrr_ablation_no_metadata": "configs/ccrr_ablation_no_metadata_default32.yaml",
        "ccrr_ablation_no_uncertainty": "configs/ccrr_ablation_no_uncertainty_default32.yaml",
    }
    for run_name, config_path in ablations.items():
        checkpoint_path = _checkpoint_path(output_root, run_name)
        if _needs_file(checkpoint_path, skip_existing):
            _run(
                [
                    python_bin,
                    "scripts/train_baseline.py",
                    "--base-config",
                    config_path,
                    "--preset",
                    "default32",
                    "--data-dir",
                    str(dataset_dir),
                    "--run-name",
                    run_name,
                    "--backbone-checkpoint",
                    str(baseline_ckpt),
                ],
                cwd=ROOT,
            )
        summary_path = _summary_path(output_root, run_name)
        if _needs_file(summary_path, skip_existing):
            _run(
                [
                    python_bin,
                    "scripts/evaluate_model.py",
                    "--base-config",
                    config_path,
                    "--preset",
                    "default32",
                    "--data-dir",
                    str(dataset_dir),
                    "--run-name",
                    run_name,
                    "--reference-baseline-checkpoint",
                    str(baseline_ckpt),
                    "--reference-visibility-checkpoint",
                    str(visibility_ckpt),
                    "--reference-residual-checkpoint",
                    str(residual_ckpt),
                ],
                cwd=ROOT,
            )


def _parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    data_root = (ROOT / args.data_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.target in {"all", "seed_repeats"}:
        for seed in _parse_seeds(args.seeds):
            _seed_protocol(
                python_bin=args.python,
                seed=seed,
                preset=args.preset_default32,
                output_root=output_root,
                data_root=data_root,
                skip_existing=args.skip_existing,
            )

    if args.target in {"all", "conditions"}:
        condition_specs = [
            {
                "name": "ccrr_noise_high",
                "ccrr_config": "configs/ccrr_noise_high_default32.yaml",
                "visibility_config": "configs/phase2_visibility_default32.yaml",
                "residual_config": "configs/phase2_residual_refine_default32.yaml",
                "preset": args.preset_default32,
            },
            {
                "name": "ccrr_sparse_uv",
                "ccrr_config": "configs/ccrr_sparse_uv_default32.yaml",
                "visibility_config": "configs/phase2_visibility_default32.yaml",
                "residual_config": "configs/phase2_residual_refine_default32.yaml",
                "preset": args.preset_default32,
            },
            {
                "name": "ccrr_exp64",
                "ccrr_config": "configs/ccrr_exp64.yaml",
                "visibility_config": "configs/phase2_visibility_exp64.yaml",
                "residual_config": "configs/phase2_residual_refine_exp64.yaml",
                "preset": args.preset_exp64,
            },
            {
                "name": "ccrr_realism_bridge2",
                "ccrr_config": "configs/ccrr_realism_bridge2_default32.yaml",
                "visibility_config": "configs/phase2_visibility_default32.yaml",
                "residual_config": "configs/phase2_residual_refine_default32.yaml",
                "preset": args.preset_default32,
            },
        ]
        for spec in condition_specs:
            _condition_protocol(
                python_bin=args.python,
                name=spec["name"],
                ccrr_config=spec["ccrr_config"],
                visibility_config=spec["visibility_config"],
                residual_config=spec["residual_config"],
                preset=spec["preset"],
                output_root=output_root,
                data_root=data_root,
                skip_existing=args.skip_existing,
            )

    if args.target in {"all", "ablations"}:
        _ablation_protocol(
            python_bin=args.python,
            output_root=output_root,
            data_root=data_root,
            skip_existing=args.skip_existing,
        )

    manifest = {
        "target": args.target,
        "seeds": _parse_seeds(args.seeds),
        "output_root": str(output_root),
        "data_root": str(data_root),
        "skip_existing": bool(args.skip_existing),
    }
    manifest_path = output_root / "ccrr_protocol" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
