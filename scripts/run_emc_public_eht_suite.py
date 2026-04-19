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
from dynadiff_vlbi.evaluation.benchmark_release import export_split_manifests  # noqa: E402
from dynadiff_vlbi.evaluation.emc_protocol import ComparatorSpec, load_comparators  # noqa: E402
from dynadiff_vlbi.evaluation.real_data_protocol import evaluate_real_data_condition  # noqa: E402
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, load_experiment_config  # noqa: E402
from dynadiff_vlbi.utils.device import get_device  # noqa: E402
from dynadiff_vlbi.utils.logging_utils import save_json  # noqa: E402
from dynadiff_vlbi.utils.seed import set_seed  # noqa: E402


REAL_DATA_CONFIGS = {
    "baseline_track_blocks": "configs/emc_real_public_eht_validation_default64.yaml",
    "station_dropout": "configs/emc_real_public_eht_station_dropout_default64.yaml",
}


def _repo_relative(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        return resolved.as_posix()
    try:
        return resolved.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


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
    parser.add_argument("--preset", default="default64")
    parser.add_argument("--data-root", default="data/real/public_eht_suite")
    parser.add_argument("--external-root", default="data/external")
    parser.add_argument("--output-root", default="outputs/public_eht_suite")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--baseline-checkpoint", default="outputs/ccrr_default64_seed7_baseline_ref/checkpoints/best.pt")
    parser.add_argument("--visibility-checkpoint", default="outputs/ccrr_default64_seed7_visibility_ref/checkpoints/best.pt")
    parser.add_argument("--residual-checkpoint", default="outputs/ccrr_default64_seed7_residual_ref/checkpoints/best.pt")
    parser.add_argument("--ccrr-checkpoint", default="outputs/ccrr_default64_seed7_main/checkpoints/best.pt")
    parser.add_argument(
        "--emc-checkpoint",
        default="outputs/emc_benchmark_baseline_tracks_default64_main_noclosure/checkpoints/best.pt",
    )
    parser.add_argument("--dps-checkpoint", default=None)
    return parser.parse_args()


def _track_slug(release_code: str) -> str:
    spec = get_public_eht_release_spec(release_code)
    return f"{spec.target_id.lower()}_{spec.campaign_year}_{spec.release_code.lower()}"


def main() -> int:
    args = parse_args()
    release_codes = [item.strip() for item in args.release_codes.split(",") if item.strip()]
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    device = get_device()
    output_root = (ROOT / args.output_root).resolve()
    data_root = (ROOT / args.data_root).resolve()
    external_root = (ROOT / args.external_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    comparator_specs = [
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
        ComparatorSpec("emc_tto", "EMC-TTO", "phase2", ROOT / args.emc_checkpoint),
    ]
    if args.dps_checkpoint:
        comparator_specs.append(ComparatorSpec("dps", "DPS", "dps", ROOT / args.dps_checkpoint))
    comparators = load_comparators(comparator_specs, device=device)
    comparator_manifest = {
        "baseline_learned": _repo_relative(ROOT / args.baseline_checkpoint),
        "visibility_conditioned": _repo_relative(ROOT / args.visibility_checkpoint),
        "residual_refinement": _repo_relative(ROOT / args.residual_checkpoint),
        "ccrr": _repo_relative(ROOT / args.ccrr_checkpoint),
        "emc": _repo_relative(ROOT / args.emc_checkpoint),
        "emc_tto": _repo_relative(ROOT / args.emc_checkpoint),
        **({"dps": _repo_relative(ROOT / args.dps_checkpoint)} if args.dps_checkpoint else {}),
    }

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
        suite_manifest["config_manifests"][family] = _repo_relative(config_manifest_path)

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
            split_manifest_dir = output_root / "split_manifests" / family / track_slug
            split_manifest = export_split_manifests(
                config=config,
                dataset_dir=dataset_dir,
                output_dir=split_manifest_dir,
            )
            skipped = bool(args.skip_existing and summary_path.exists())
            if not skipped:
                evaluate_real_data_condition(
                    config=config,
                    dataset_dir=dataset_dir,
                    comparators=comparators,
                    output_dir=run_dir,
                    support_fractions=config.holdout.eval_support_fractions,
                    use_domain_adaptation=True,
                )
            results_manifest = {
                "kind": "public_eht_real_data_condition",
                "family": family,
                "release_code": spec.release_code,
                "target": spec.target_id,
                "campaign_year": spec.campaign_year,
                "seed": int(config.project.seed),
                "support_fractions": [float(value) for value in config.holdout.eval_support_fractions],
                "dataset_dir": _repo_relative(dataset_dir),
                "dataset_manifest_path": _repo_relative(manifest_path),
                "config_manifest_path": _repo_relative(output_root / "config_manifests" / f"{family}.json"),
                "split_manifest_path": _repo_relative(split_manifest_dir / "split_manifest.json"),
                "summary_path": _repo_relative(summary_path),
                "output_dir": _repo_relative(run_dir),
                "metrics_csv": _repo_relative(run_dir / "logs" / "support_fraction_metrics.csv"),
                "prediction_paths": {
                    f"{int(round(float(fraction) * 100.0)):02d}": _repo_relative(
                        run_dir / "predictions" / f"support_{int(round(float(fraction) * 100.0)):02d}.npz"
                    )
                    for fraction in config.holdout.eval_support_fractions
                },
                "domain_adaptation": {
                    "enabled": True,
                    "tto_steps": 50,
                    "tto_lr": 1.0e-4,
                },
                "comparators": comparator_manifest,
            }
            run_results_manifest_path = run_dir / "logs" / "results_manifest.json"
            suite_results_manifest_path = output_root / "results_manifests" / f"{family}__{track_slug}.json"
            run_results_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            suite_results_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(run_results_manifest_path, results_manifest)
            save_json(suite_results_manifest_path, results_manifest)
            suite_manifest["runs"].append(
                {
                    "family": family,
                    "release_code": spec.release_code,
                    "target": spec.target_id,
                    "campaign_year": spec.campaign_year,
                    "dataset_dir": _repo_relative(dataset_dir),
                    "output_dir": _repo_relative(run_dir),
                    "summary_path": _repo_relative(summary_path),
                    "split_manifest_path": _repo_relative(split_manifest_dir / "split_manifest.json"),
                    "results_manifest_path": _repo_relative(suite_results_manifest_path),
                    "skipped_existing": skipped,
                    "split_manifest": split_manifest,
                }
            )

    save_json(output_root / "suite_manifest.json", suite_manifest)
    print(suite_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
