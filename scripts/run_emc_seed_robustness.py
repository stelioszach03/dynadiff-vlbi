#!/usr/bin/env python3
"""Run a bounded seed-robustness protocol for the EMC default64 baseline-track benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.benchmark_release import export_split_manifests  # noqa: E402
from dynadiff_vlbi.evaluation.emc_protocol import ComparatorSpec, evaluate_emc_condition, load_comparators  # noqa: E402
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, load_experiment_config  # noqa: E402
from dynadiff_vlbi.utils.device import get_device  # noqa: E402
from dynadiff_vlbi.utils.logging_utils import save_json  # noqa: E402
from dynadiff_vlbi.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--data-root", default="data/generated")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--preset", default="default64")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seeds", default="7,19,31,42,137")
    return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
    return [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def _run(command: list[str]) -> None:
    print(f"\n[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _ensure_dataset(
    *,
    python_bin: str,
    base_config: Path,
    preset: str,
    dataset_dir: Path,
    skip_existing: bool,
) -> None:
    if (dataset_dir / "test.npz").exists():
        return
    if skip_existing and (dataset_dir / "test.npz").exists():
        return
    _run(
        [
            python_bin,
            "scripts/generate_toy_dataset.py",
            "--base-config",
            str(base_config),
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
    base_config: str | Path | None = None,
    backbone_checkpoint: Path | None = None,
) -> Path:
    checkpoint_path = _checkpoint_path(output_root, run_name)
    if checkpoint_path.exists():
        return checkpoint_path
    if skip_existing and checkpoint_path.exists():
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
        command.extend(["--base-config", str(base_config)])
    if backbone_checkpoint is not None:
        command.extend(["--backbone-checkpoint", str(backbone_checkpoint)])
    _run(command)
    return checkpoint_path


def _write_seeded_config(source_path: str | Path, output_path: Path, seed: int) -> Path:
    payload = yaml.safe_load(Path(source_path).read_text(encoding="utf-8")) or {}
    project = dict(payload.get("project", {}))
    project["seed"] = int(seed)
    payload["project"] = project
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def _checkpoint_path(output_root: Path, run_name: str) -> Path:
    return output_root / run_name / "checkpoints" / "best.pt"


def _protocol_summary_path(output_root: Path, run_name: str) -> Path:
    return output_root / run_name / "logs" / "emc_protocol_summary.json"


def _repo_relative(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        return resolved.as_posix()
    try:
        return resolved.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _seed_paths(output_root: Path, seed: int) -> dict[str, Path | str]:
    if seed == 7:
        return {
            "emc_run_name": "emc_benchmark_baseline_tracks_default64_main_noclosure",
            "protocol_run_name": "emc_benchmark_baseline_tracks_default64_protocol",
            "baseline_run_name": "ccrr_default64_seed7_baseline_ref",
            "residual_run_name": "ccrr_default64_seed7_residual_ref",
            "ccrr_run_name": "ccrr_default64_seed7_main",
        }
    return {
        "emc_run_name": f"emc_default64_seed{seed}_baseline_tracks_main_noclosure",
        "protocol_run_name": f"emc_default64_seed{seed}_baseline_tracks_protocol",
        "baseline_run_name": f"ccrr_default64_seed{seed}_baseline_ref",
        "residual_run_name": f"ccrr_default64_seed{seed}_residual_ref",
        "ccrr_run_name": f"ccrr_default64_seed{seed}_main",
    }


def _load_seeded_config(base_config: Path, preset: str):
    config = load_experiment_config(
        base_path=base_config,
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset=preset,
        default_base_path=ROOT / DEFAULT_BASE_CONFIG_PATH,
    )
    set_seed(config.project.seed)
    return config


def main() -> int:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    data_root = (ROOT / args.data_root).resolve()
    temp_dir = output_root / "emc_seed_robustness" / "temp_configs"
    config_manifest_dir = output_root / "emc_seed_robustness" / "config_manifests"
    split_manifest_root = output_root / "emc_seed_robustness" / "split_manifests"
    results_manifest_dir = output_root / "emc_seed_robustness" / "results_manifests"
    summary_runs: list[dict[str, object]] = []

    for seed in _parse_seeds(args.seeds):
        seed_config = _write_seeded_config("configs/emc_default64.yaml", temp_dir / f"emc_seed_{seed}.yaml", seed)
        benchmark_seed_config = _write_seeded_config(
            "configs/emc_benchmark_baseline_tracks_default64.yaml",
            temp_dir / f"emc_benchmark_seed_{seed}.yaml",
            seed,
        )
        config = _load_seeded_config(seed_config, args.preset)
        dataset_dir = data_root / f"ccrr_default64_seed{seed}_shared"
        _ensure_dataset(
            python_bin=args.python,
            base_config=benchmark_seed_config,
            preset=args.preset,
            dataset_dir=dataset_dir,
            skip_existing=args.skip_existing,
        )

        names = _seed_paths(output_root, seed)
        config_manifest_path = config_manifest_dir / f"seed_{seed}.json"
        config_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(config_manifest_path, config.to_dict())
        split_manifest_dir = split_manifest_root / f"seed_{seed}"
        split_manifest = export_split_manifests(
            config=config,
            dataset_dir=dataset_dir,
            output_dir=split_manifest_dir,
        )
        baseline_checkpoint = _ensure_train_run(
            python_bin=args.python,
            run_name=str(names["baseline_run_name"]),
            output_root=output_root,
            data_dir=dataset_dir,
            preset=args.preset,
            skip_existing=args.skip_existing,
        )
        residual_checkpoint = _ensure_train_run(
            python_bin=args.python,
            run_name=str(names["residual_run_name"]),
            output_root=output_root,
            data_dir=dataset_dir,
            preset=args.preset,
            skip_existing=args.skip_existing,
            base_config="configs/phase2_residual_refine_default64.yaml",
            backbone_checkpoint=baseline_checkpoint,
        )
        ccrr_checkpoint = _ensure_train_run(
            python_bin=args.python,
            run_name=str(names["ccrr_run_name"]),
            output_root=output_root,
            data_dir=dataset_dir,
            preset=args.preset,
            skip_existing=args.skip_existing,
            base_config="configs/ccrr_default64.yaml",
            backbone_checkpoint=baseline_checkpoint,
        )

        emc_run_name = str(names["emc_run_name"])
        protocol_run_name = str(names["protocol_run_name"])
        emc_checkpoint = _checkpoint_path(output_root, emc_run_name)
        if not (args.skip_existing and emc_checkpoint.exists()) and not emc_checkpoint.exists():
            _run(
                [
                    args.python,
                    "scripts/train_baseline.py",
                    "--base-config",
                    str(seed_config),
                    "--preset",
                    args.preset,
                    "--data-dir",
                    str(dataset_dir),
                    "--run-name",
                    emc_run_name,
                    "--backbone-checkpoint",
                    str(baseline_checkpoint),
                ]
            )

        summary_path = _protocol_summary_path(output_root, protocol_run_name)
        protocol_output_dir = output_root / protocol_run_name
        if not (args.skip_existing and summary_path.exists()) and not summary_path.exists():
            comparators = load_comparators(
                [
                    ComparatorSpec("dirty", "Dirty", "dirty"),
                    ComparatorSpec("tikhonov", "Tikhonov", "tikhonov"),
                    ComparatorSpec("baseline_learned", "Baseline 3D U-Net", "baseline", baseline_checkpoint),
                    ComparatorSpec("residual_refinement", "Residual Refinement", "phase2", residual_checkpoint),
                    ComparatorSpec("ccrr", "CCRR", "phase2", ccrr_checkpoint),
                    ComparatorSpec("emc", "EMC", "phase2", emc_checkpoint),
                ],
                device=get_device(),
            )
            evaluate_emc_condition(
                config=config,
                dataset_dir=dataset_dir,
                comparators=comparators,
                output_dir=protocol_output_dir,
                support_fractions=config.holdout.eval_support_fractions,
            )
        protocol_results_manifest_path = protocol_output_dir / "logs" / "results_manifest.json"
        aggregate_results_manifest_path = results_manifest_dir / f"seed_{seed}.json"
        aggregate_results_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        results_manifest = {
            "kind": "seed_robustness_run",
            "seed": int(seed),
            "preset": args.preset,
            "dataset_dir": _repo_relative(dataset_dir),
            "config_manifest_path": _repo_relative(config_manifest_path),
            "split_manifest_path": _repo_relative(split_manifest_dir / "split_manifest.json"),
            "summary_path": _repo_relative(summary_path),
            "protocol_dir": _repo_relative(protocol_output_dir),
            "checkpoints": {
                "baseline": _repo_relative(baseline_checkpoint),
                "residual_refinement": _repo_relative(residual_checkpoint),
                "ccrr": _repo_relative(ccrr_checkpoint),
                "emc": _repo_relative(emc_checkpoint),
            },
            "support_fractions": [float(value) for value in config.holdout.eval_support_fractions],
        }
        protocol_results_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(protocol_results_manifest_path, results_manifest)
        save_json(aggregate_results_manifest_path, results_manifest)

        summary_runs.append(
            {
                "seed": seed,
                "dataset_dir": _repo_relative(dataset_dir),
                "config_path": _repo_relative(seed_config),
                "config_manifest_path": _repo_relative(config_manifest_path),
                "split_manifest_path": _repo_relative(split_manifest_dir / "split_manifest.json"),
                "results_manifest_path": _repo_relative(aggregate_results_manifest_path),
                "emc_checkpoint": _repo_relative(emc_checkpoint),
                "protocol_summary": _repo_relative(summary_path),
                "protocol_dir": _repo_relative(protocol_output_dir),
                "baseline_checkpoint": _repo_relative(baseline_checkpoint),
                "residual_checkpoint": _repo_relative(residual_checkpoint),
                "ccrr_checkpoint": _repo_relative(ccrr_checkpoint),
                "split_manifest": split_manifest,
            }
        )

    save_json(
        output_root / "emc_seed_robustness" / "seed_robustness_manifest.json",
        {"seeds": _parse_seeds(args.seeds), "runs": summary_runs},
    )
    print(json.dumps({"seeds": _parse_seeds(args.seeds), "runs": summary_runs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
