#!/usr/bin/env python3
"""Generate synthetic train/validation/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_BASE_CONFIG = "configs/base.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.data.synthetic_generator import generate_dataset_splits
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, load_experiment_config
from dynadiff_vlbi.utils.logging_utils import save_yaml
from dynadiff_vlbi.utils.seed import set_seed


def _explicit_arg(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def _resolve_preset(raw_preset: str | None, base_config_explicit: bool) -> str | None:
    if raw_preset is not None:
        return raw_preset
    if base_config_explicit:
        return None
    return "smoke"


def _experiment_label(base_config_path: Path, preset: str | None) -> str:
    is_default_base = base_config_path.resolve() == (ROOT / DEFAULT_BASE_CONFIG_PATH).resolve()
    base_name = base_config_path.stem
    if preset is not None and is_default_base:
        return preset
    if preset is not None:
        return f"{preset}_{base_name}"
    return base_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--preset", default=None, choices=["smoke", "default32", "exp64"])
    parser.add_argument("--output-dir", default=None, help="Optional override for the dataset output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_explicit = _explicit_arg("--base-config")
    preset = _resolve_preset(args.preset, base_config_explicit=base_config_explicit)
    base_config_path = ROOT / args.base_config
    config = load_experiment_config(
        base_path=base_config_path,
        train_path=ROOT / args.train_config,
        eval_path=ROOT / args.eval_config,
        preset=preset,
        default_base_path=ROOT / DEFAULT_BASE_CONFIG_PATH,
    )
    set_seed(config.project.seed)
    output_label = _experiment_label(base_config_path=base_config_path, preset=preset)
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / config.paths.data_root / output_label
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
