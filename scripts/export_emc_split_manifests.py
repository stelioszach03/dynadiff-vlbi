#!/usr/bin/env python3
"""Export deterministic EMC support-target split manifests for one dataset/config pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynadiff_vlbi.evaluation.benchmark_release import export_split_manifests
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        required=True,
        help="Path to the EMC benchmark base config.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Optional preset name. Omit to use the base config exactly as written.",
    )
    parser.add_argument(
        "--train-config",
        default="configs/train.yaml",
        help="Path to the train preset file.",
    )
    parser.add_argument(
        "--eval-config",
        default="configs/eval.yaml",
        help="Path to the evaluation config file.",
    )
    parser.add_argument(
        "--default-base-config",
        default=str(DEFAULT_BASE_CONFIG_PATH),
        help="Default shared base config used as the merge root.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Generated dataset directory containing train/val/test .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where deterministic split manifests will be written.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to export. Default: test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(
        base_path=args.base_config,
        train_path=args.train_config,
        eval_path=args.eval_config,
        preset=args.preset,
        default_base_path=args.default_base_config,
    )
    manifest = export_split_manifests(
        config=config,
        dataset_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        split_name=args.split,
    )
    summary = {
        "config": str(Path(args.base_config).resolve()),
        "dataset_dir": str(Path(args.data_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "split_manifest": str((Path(args.output_dir) / "split_manifest.json").resolve()),
        "strategy": manifest["strategy"],
        "support_tags": manifest["support_tags"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
