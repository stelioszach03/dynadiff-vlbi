#!/usr/bin/env python3
"""Generate synthetic train/validation/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.data.synthetic_generator import generate_dataset_splits
from dynadiff_vlbi.utils.config import load_experiment_config
from dynadiff_vlbi.utils.logging_utils import save_yaml
from dynadiff_vlbi.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--preset", default="smoke", choices=["smoke", "default32", "exp64"])
    parser.add_argument("--output-dir", default=None, help="Optional override for the dataset output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(
        base_path=ROOT / args.base_config,
        train_path=ROOT / args.train_config,
        eval_path=ROOT / args.eval_config,
        preset=args.preset,
    )
    set_seed(config.project.seed)
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / config.paths.data_root / args.preset
    saved_paths = generate_dataset_splits(
        output_dir=output_dir,
        dataset_config=config.dataset,
        sampling_config=config.sampling,
        noise_config=config.noise,
        base_seed=config.project.seed,
    )
    save_yaml(output_dir / "config_snapshot.yaml", config.to_dict())
    print(f"Saved synthetic dataset to: {output_dir}")
    for split_name, split_path in saved_paths.items():
        print(f"  {split_name}: {split_path}")


if __name__ == "__main__":
    main()
