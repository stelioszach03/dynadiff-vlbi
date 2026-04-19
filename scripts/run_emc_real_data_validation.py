#!/usr/bin/env python3
"""Run observation-domain EMC validation on one prepared public EHT dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.emc_protocol import ComparatorSpec, load_comparators  # noqa: E402
from dynadiff_vlbi.evaluation.real_data_protocol import evaluate_real_data_condition  # noqa: E402
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, load_experiment_config  # noqa: E402
from dynadiff_vlbi.utils.device import get_device  # noqa: E402
from dynadiff_vlbi.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/emc_real_public_eht_validation.yaml")
    parser.add_argument("--preset", default="default32")
    parser.add_argument("--data-dir", default="data/real/eht_public_m87_2017_2019-d01-01_emc")
    parser.add_argument("--run-name", default="emc_real_public_eht_validation")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--baseline-checkpoint", default="outputs/ccrr_seed7_baseline_ref/checkpoints/best.pt")
    parser.add_argument("--visibility-checkpoint", default="outputs/ccrr_seed7_visibility_ref/checkpoints/best.pt")
    parser.add_argument("--residual-checkpoint", default="outputs/ccrr_seed7_residual_ref/checkpoints/best.pt")
    parser.add_argument("--ccrr-checkpoint", default="outputs/ccrr_seed7_main/checkpoints/best.pt")
    parser.add_argument(
        "--emc-checkpoint",
        default="outputs/emc_benchmark_baseline_tracks_main_noclosure/checkpoints/best.pt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment_config(
        base_path=ROOT / args.base_config,
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset=args.preset,
        default_base_path=ROOT / DEFAULT_BASE_CONFIG_PATH,
    )
    set_seed(config.project.seed)
    output_dir = (ROOT / args.output_root / args.run_name).resolve()
    comparators = load_comparators(
        [
            ComparatorSpec("dirty", "Dirty", "dirty"),
            ComparatorSpec("tikhonov", "Tikhonov", "tikhonov"),
            ComparatorSpec("ehtim_bridge", "eht-imaging bridge", "ehtim_bridge"),
            ComparatorSpec("baseline_learned", "Baseline 3D U-Net", "baseline", (ROOT / args.baseline_checkpoint)),
            ComparatorSpec(
                "visibility_conditioned",
                "Standalone Visibility",
                "phase2",
                (ROOT / args.visibility_checkpoint),
            ),
            ComparatorSpec(
                "residual_refinement",
                "Residual Refinement",
                "phase2",
                (ROOT / args.residual_checkpoint),
            ),
            ComparatorSpec("ccrr", "CCRR", "phase2", (ROOT / args.ccrr_checkpoint)),
            ComparatorSpec("emc", "EMC", "phase2", (ROOT / args.emc_checkpoint)),
        ],
        device=get_device(),
    )
    summary = evaluate_real_data_condition(
        config=config,
        dataset_dir=(ROOT / args.data_dir).resolve(),
        comparators=comparators,
        output_dir=output_dir,
        support_fractions=config.holdout.eval_support_fractions,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
