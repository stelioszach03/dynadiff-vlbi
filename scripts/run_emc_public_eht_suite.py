#!/usr/bin/env python3
"""Prepare and evaluate the multi-release public-EHT EMC validation suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.data.eht_public_data import (  # noqa: E402
    get_public_eht_release_spec,
    prepare_public_eht_validation_dataset,
    ensure_public_eht_repo,
)
from dynadiff_vlbi.evaluation.emc_protocol import ComparatorSpec, load_comparators  # noqa: E402
from dynadiff_vlbi.evaluation.real_data_protocol import evaluate_real_data_condition  # noqa: E402
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, load_experiment_config  # noqa: E402
from dynadiff_vlbi.utils.device import get_device  # noqa: E402
from dynadiff_vlbi.utils.logging_utils import save_json  # noqa: E402
from dynadiff_vlbi.utils.seed import set_seed  # noqa: E402


REAL_DATA_CONFIGS = {
    "baseline_track_blocks": "configs/emc_real_public_eht_validation.yaml",
    "station_dropout": "configs/emc_real_public_eht_station_dropout.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-codes",
        default="2019-D01-01,2024-D01-01,2020-D01-01,2021-D03-01",
        help="Comma-separated official public EHT release codes to include.",
    )
    parser.add_argument(
        "--families",
        default="baseline_track_blocks,station_dropout",
        help="Comma-separated real-data holdout families to run.",
    )
    parser.add_argument("--preset", default="default32")
    parser.add_argument("--data-root", default="data/real/public_eht_suite")
    parser.add_argument("--external-root", default="data/external")
    parser.add_argument("--output-root", default="outputs/public_eht_suite")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--baseline-checkpoint", default="outputs/ccrr_seed7_baseline_ref/checkpoints/best.pt")
    parser.add_argument("--visibility-checkpoint", default="outputs/ccrr_seed7_visibility_ref/checkpoints/best.pt")
    parser.add_argument("--residual-checkpoint", default="outputs/ccrr_seed7_residual_ref/checkpoints/best.pt")
    parser.add_argument("--ccrr-checkpoint", default="outputs/ccrr_seed7_main/checkpoints/best.pt")
    parser.add_argument(
        "--emc-checkpoint",
        default="outputs/emc_benchmark_baseline_tracks_main_noclosure/checkpoints/best.pt",
    )
    # Phase 4 / 5: adaptive-partition plumbing.
    parser.add_argument(
        "--partition-mode",
        default="deterministic",
        choices=["deterministic", "adaptive"],
    )
    parser.add_argument(
        "--oracle-ckpt",
        default=None,
        help="Path to a trained HeavyHitterOracle checkpoint; required when --partition-mode=adaptive.",
    )
    args = parser.parse_args()
    if args.partition_mode == "adaptive" and args.oracle_ckpt is None:
        parser.error("--partition-mode=adaptive requires --oracle-ckpt <path>")
    return args


def _track_slug(release_code: str) -> str:
    spec = get_public_eht_release_spec(release_code)
    return f"{spec.target_id.lower()}_{spec.campaign_year}_{spec.release_code.lower()}"


def main() -> int:
    args = parse_args()
    # Expose adaptive-partition selection to the real-data protocol via the
    # same env-var channel the synthetic benchmark uses. No oracle load here;
    # the protocol evaluator does the lazy load via resolve_partition_strategy.
    import os as _os

    _os.environ["DYNADIFF_PARTITION_MODE"] = args.partition_mode
    if args.oracle_ckpt is not None:
        _os.environ["DYNADIFF_ORACLE_CKPT"] = str((ROOT / args.oracle_ckpt).resolve())
    elif "DYNADIFF_ORACLE_CKPT" in _os.environ and args.partition_mode != "adaptive":
        del _os.environ["DYNADIFF_ORACLE_CKPT"]

    release_codes = [item.strip() for item in args.release_codes.split(",") if item.strip()]
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    device = get_device()
    output_root = (ROOT / args.output_root).resolve()
    data_root = (ROOT / args.data_root).resolve()
    external_root = (ROOT / args.external_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    comparators = load_comparators(
        [
            ComparatorSpec("dirty", "Dirty", "dirty"),
            ComparatorSpec("tikhonov", "Tikhonov", "tikhonov"),
            ComparatorSpec("ehtim_bridge", "eht-imaging bridge", "ehtim_bridge"),
            ComparatorSpec("baseline_learned", "Baseline 3D U-Net", "baseline", ROOT / args.baseline_checkpoint),
            ComparatorSpec(
                "visibility_conditioned",
                "Standalone Visibility",
                "phase2",
                ROOT / args.visibility_checkpoint,
            ),
            ComparatorSpec("residual_refinement", "Residual Refinement", "phase2", ROOT / args.residual_checkpoint),
            ComparatorSpec("ccrr", "CCRR", "phase2", ROOT / args.ccrr_checkpoint),
            ComparatorSpec("emc", "EMC", "phase2", ROOT / args.emc_checkpoint),
        ],
        device=device,
    )

    suite_manifest: dict[str, object] = {
        "release_codes": release_codes,
        "families": families,
        "tracks": [],
        "runs": [],
        "config_manifests": {},
    }

    for family in families:
        if family not in REAL_DATA_CONFIGS:
            raise KeyError(f"Unsupported real-data family '{family}'. Supported: {sorted(REAL_DATA_CONFIGS)}")
        config = load_experiment_config(
            base_path=ROOT / REAL_DATA_CONFIGS[family],
            train_path=ROOT / "configs/train.yaml",
            eval_path=ROOT / "configs/eval.yaml",
            preset=args.preset,
            default_base_path=ROOT / DEFAULT_BASE_CONFIG_PATH,
        )
        set_seed(config.project.seed)
        config_manifest_path = output_root / "config_manifests" / f"{family}.json"
        config_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(config_manifest_path, asdict(config))
        suite_manifest["config_manifests"][family] = str(config_manifest_path)

    for release_code in release_codes:
        spec = get_public_eht_release_spec(release_code)
        track_slug = _track_slug(release_code)
        repo_dir = ensure_public_eht_repo(
            destination=external_root / f"eht-public-{spec.release_code}",
            release_code=spec.release_code,
        )
        dataset_dir = data_root / track_slug
        manifest_path = dataset_dir / "real_data_manifest.json"
        if manifest_path.exists() and args.skip_existing:
            dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            dataset_manifest = prepare_public_eht_validation_dataset(
                source_root=repo_dir,
                output_dir=dataset_dir,
                release_code=spec.release_code,
                image_size=args.image_size,
                sequence_length=args.sequence_length,
            )
        suite_manifest["tracks"].append(
            {
                "release_code": spec.release_code,
                "target": spec.target_id,
                "campaign_year": spec.campaign_year,
                "dataset_dir": str(dataset_dir),
                "manifest_path": str(manifest_path),
                "sample_count": dataset_manifest["sample_count"],
            }
        )

        for family in families:
            config = load_experiment_config(
                base_path=ROOT / REAL_DATA_CONFIGS[family],
                train_path=ROOT / "configs/train.yaml",
                eval_path=ROOT / "configs/eval.yaml",
                preset=args.preset,
                default_base_path=ROOT / DEFAULT_BASE_CONFIG_PATH,
            )
            set_seed(config.project.seed)
            run_dir = output_root / family / track_slug
            summary_path = run_dir / "logs" / "real_data_protocol_summary.json"
            skipped = bool(args.skip_existing and summary_path.exists())
            if not skipped:
                evaluate_real_data_condition(
                    config=config,
                    dataset_dir=dataset_dir,
                    comparators=comparators,
                    output_dir=run_dir,
                    support_fractions=config.holdout.eval_support_fractions,
                )
            suite_manifest["runs"].append(
                {
                    "family": family,
                    "release_code": spec.release_code,
                    "target": spec.target_id,
                    "campaign_year": spec.campaign_year,
                    "dataset_dir": str(dataset_dir),
                    "output_dir": str(run_dir),
                    "summary_path": str(summary_path),
                    "skipped_existing": skipped,
                }
            )

    save_json(output_root / "suite_manifest.json", suite_manifest)
    print(suite_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
