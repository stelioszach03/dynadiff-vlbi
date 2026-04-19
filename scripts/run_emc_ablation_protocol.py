#!/usr/bin/env python3
"""Run compact EMC ablations on the shared default32 earned-consistency benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.emc_protocol import ComparatorSpec, evaluate_emc_condition, load_comparators  # noqa: E402
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, load_experiment_config  # noqa: E402
from dynadiff_vlbi.utils.device import get_device  # noqa: E402
from dynadiff_vlbi.utils.seed import set_seed  # noqa: E402


@dataclass(frozen=True)
class AblationSpec:
    key: str
    label: str
    base_config: str
    run_name: str


ABLATIONS = [
    AblationSpec(
        "emc_no_dc",
        "EMC w/o DC",
        "configs/emc_ablation_no_dc_default32.yaml",
        "emc_ablation_no_dc_noclosure",
    ),
    AblationSpec(
        "emc_with_closure",
        "EMC + closure auxiliary",
        "configs/emc_ablation_with_closure_default32.yaml",
        "emc_ablation_with_closure_aux",
    ),
    AblationSpec(
        "emc_no_metadata",
        "EMC w/o Metadata",
        "configs/emc_ablation_no_metadata_default32.yaml",
        "emc_ablation_no_metadata_noclosure",
    ),
    AblationSpec(
        "emc_no_uncertainty",
        "EMC w/o Uncertainty",
        "configs/emc_ablation_no_uncertainty_default32.yaml",
        "emc_ablation_no_uncertainty_noclosure",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--data-dir", default="data/generated/ccrr_default32_seed7_shared")
    parser.add_argument("--preset", default="default32")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(f"\n[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _checkpoint_path(output_root: Path, run_name: str) -> Path:
    return output_root / run_name / "checkpoints" / "best.pt"


def _ensure_checkpoint(
    *,
    python_bin: str,
    output_root: Path,
    data_dir: Path,
    preset: str,
    base_config: str,
    run_name: str,
    backbone_checkpoint: Path,
    skip_existing: bool,
) -> Path:
    checkpoint_path = _checkpoint_path(output_root, run_name)
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


def main() -> int:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    data_dir = (ROOT / args.data_dir).resolve()
    backbone_checkpoint = output_root / "ccrr_seed7_baseline_ref" / "checkpoints" / "best.pt"
    baseline_checkpoint = backbone_checkpoint
    residual_checkpoint = output_root / "ccrr_seed7_residual_ref" / "checkpoints" / "best.pt"
    ccrr_checkpoint = output_root / "ccrr_seed7_main" / "checkpoints" / "best.pt"
    emc_checkpoint = output_root / "emc_benchmark_baseline_tracks_main_noclosure" / "checkpoints" / "best.pt"

    config = load_experiment_config(
        base_path=ROOT / "configs/emc_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset=args.preset,
        default_base_path=ROOT / DEFAULT_BASE_CONFIG_PATH,
    )
    set_seed(config.project.seed)

    ablation_checkpoints: dict[str, Path] = {}
    for spec in ABLATIONS:
        ablation_checkpoints[spec.key] = _ensure_checkpoint(
            python_bin=args.python,
            output_root=output_root,
            data_dir=data_dir,
            preset=args.preset,
            base_config=spec.base_config,
            run_name=spec.run_name,
            backbone_checkpoint=backbone_checkpoint,
            skip_existing=args.skip_existing,
        )

    protocol_output_dir = output_root / "emc_ablation_protocol"
    summary_path = protocol_output_dir / "logs" / "emc_protocol_summary.json"
    if args.skip_existing and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(summary_path)
        return 0

    comparators = load_comparators(
        [
            ComparatorSpec("dirty", "Dirty", "dirty"),
            ComparatorSpec("tikhonov", "Tikhonov", "tikhonov"),
            ComparatorSpec("baseline_learned", "Baseline 3D U-Net", "baseline", baseline_checkpoint),
            ComparatorSpec("residual_refinement", "Residual Refinement", "phase2", residual_checkpoint),
            ComparatorSpec("ccrr", "CCRR", "phase2", ccrr_checkpoint),
            ComparatorSpec("emc", "EMC", "phase2", emc_checkpoint),
            *[
                ComparatorSpec(spec.key, spec.label, "phase2", ablation_checkpoints[spec.key])
                for spec in ABLATIONS
            ],
        ],
        device=get_device(),
    )
    summary = evaluate_emc_condition(
        config=config,
        dataset_dir=data_dir,
        comparators=comparators,
        output_dir=protocol_output_dir,
        support_fractions=config.holdout.eval_support_fractions,
    )
    print(json.dumps({"output_dir": str(protocol_output_dir), "models": list(summary["comparator_labels"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
