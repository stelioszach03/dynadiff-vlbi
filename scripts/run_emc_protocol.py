#!/usr/bin/env python3
"""Run the earned measurement consistency protocol on shared comparator splits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.emc_protocol import ComparatorSpec, evaluate_emc_condition, load_comparators
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, load_experiment_config
from dynadiff_vlbi.utils.device import get_device
from dynadiff_vlbi.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="all", choices=["all", "default32", "sparse_uv"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--data-root", default="data/generated")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(f"\n[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _ensure_emc_checkpoint(
    *,
    python_bin: str,
    base_config: str,
    preset: str,
    data_dir: Path,
    run_name: str,
    backbone_checkpoint: Path,
    output_root: Path,
    skip_existing: bool,
) -> Path:
    checkpoint_path = output_root / run_name / "checkpoints" / "best.pt"
    if skip_existing and checkpoint_path.exists():
        return checkpoint_path
    if checkpoint_path.exists():
        return checkpoint_path
    _run(
        [
            python_bin,
            "scripts/train_baseline.py",
            "--base-config",
            base_config,
            "--preset",
            preset,
            "--data-dir",
            str(data_dir),
            "--run-name",
            run_name,
            "--backbone-checkpoint",
            str(backbone_checkpoint),
        ]
    )
    return checkpoint_path


def _condition_protocol(
    *,
    condition_name: str,
    base_config: str,
    preset: str,
    dataset_dir: Path,
    output_root: Path,
    existing_baseline_checkpoint: Path,
    existing_residual_checkpoint: Path,
    existing_ccrr_checkpoint: Path,
    emc_run_name: str,
    protocol_run_name: str,
    python_bin: str,
    skip_existing: bool,
) -> dict[str, object]:
    config = load_experiment_config(
        base_path=ROOT / base_config,
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset=preset,
        default_base_path=ROOT / DEFAULT_BASE_CONFIG_PATH,
    )
    set_seed(config.project.seed)
    emc_checkpoint = _ensure_emc_checkpoint(
        python_bin=python_bin,
        base_config=base_config,
        preset=preset,
        data_dir=dataset_dir,
        run_name=emc_run_name,
        backbone_checkpoint=existing_baseline_checkpoint,
        output_root=output_root,
        skip_existing=skip_existing,
    )
    protocol_output_dir = output_root / protocol_run_name
    summary_path = protocol_output_dir / "logs" / "emc_protocol_summary.json"
    if skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    comparators = load_comparators(
        [
            ComparatorSpec("dirty", "Dirty", "dirty"),
            ComparatorSpec("tikhonov", "Tikhonov", "tikhonov"),
            ComparatorSpec("baseline_learned", "Baseline 3D U-Net", "baseline", existing_baseline_checkpoint),
            ComparatorSpec(
                "residual_refinement",
                "Residual Refinement",
                "phase2",
                existing_residual_checkpoint,
            ),
            ComparatorSpec("ccrr", "CCRR", "phase2", existing_ccrr_checkpoint),
            ComparatorSpec("emc", "EMC", "phase2", emc_checkpoint),
        ],
        device=get_device(),
    )
    summary = evaluate_emc_condition(
        config=config,
        dataset_dir=dataset_dir,
        comparators=comparators,
        output_dir=protocol_output_dir,
        support_fractions=config.holdout.eval_support_fractions,
    )
    print(json.dumps({"condition": condition_name, "output_dir": str(protocol_output_dir)}, indent=2))
    return summary


def main() -> int:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    data_root = (ROOT / args.data_root).resolve()

    results: dict[str, object] = {}
    if args.target in {"all", "default32"}:
        results["default32"] = _condition_protocol(
            condition_name="default32",
            base_config="configs/emc_default32.yaml",
            preset="default32",
            dataset_dir=data_root / "ccrr_default32_seed7_shared",
            output_root=output_root,
            existing_baseline_checkpoint=output_root / "ccrr_seed7_baseline_ref" / "checkpoints" / "best.pt",
            existing_residual_checkpoint=output_root / "ccrr_seed7_residual_ref" / "checkpoints" / "best.pt",
            existing_ccrr_checkpoint=output_root / "ccrr_seed7_main" / "checkpoints" / "best.pt",
            emc_run_name="emc_default32_main",
            protocol_run_name="emc_default32_protocol",
            python_bin=args.python,
            skip_existing=args.skip_existing,
        )
    if args.target in {"all", "sparse_uv"}:
        results["sparse_uv"] = _condition_protocol(
            condition_name="sparse_uv",
            base_config="configs/emc_sparse_uv_default32.yaml",
            preset="default32",
            dataset_dir=data_root / "ccrr_sparse_uv_shared",
            output_root=output_root,
            existing_baseline_checkpoint=output_root / "ccrr_sparse_uv_baseline_ref" / "checkpoints" / "best.pt",
            existing_residual_checkpoint=output_root / "ccrr_sparse_uv_residual_ref" / "checkpoints" / "best.pt",
            existing_ccrr_checkpoint=output_root / "ccrr_sparse_uv_main" / "checkpoints" / "best.pt",
            emc_run_name="emc_sparse_uv_main",
            protocol_run_name="emc_sparse_uv_protocol",
            python_bin=args.python,
            skip_existing=args.skip_existing,
        )
    print(json.dumps({"completed": list(results.keys())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
