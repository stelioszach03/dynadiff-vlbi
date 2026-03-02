#!/usr/bin/env python3
"""Train the compact temporal U-Net baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_BASE_CONFIG = "configs/base.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.data.dataset import build_dataloaders
from dynadiff_vlbi.models.temporal_unet import TemporalUNet3D
from dynadiff_vlbi.training.trainer import Trainer
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, load_experiment_config
from dynadiff_vlbi.utils.device import get_device
from dynadiff_vlbi.utils.logging_utils import prepare_output_dirs
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
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for quick experiments.")
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
    if args.epochs is not None:
        config.training.epochs = int(args.epochs)
    set_seed(config.project.seed)

    experiment_label = _experiment_label(base_config_path=base_config_path, preset=preset)
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / config.paths.data_root / experiment_label
    if not (data_dir / "train.npz").exists():
        raise FileNotFoundError(
            f"Missing dataset split under {data_dir}. Run scripts/generate_toy_dataset.py first."
        )

    output_root = Path(args.output_root) if args.output_root else ROOT / config.paths.output_root
    run_name = args.run_name or f"train_{experiment_label}"
    output_dirs = prepare_output_dirs(str(output_root), run_name=run_name, config=config)
    train_loader, val_loader, _ = build_dataloaders(
        data_dir=data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    model = TemporalUNet3D(config.model)
    trainer = Trainer(model=model, config=config, device=get_device(), output_dirs=output_dirs)
    summary = trainer.fit(train_loader=train_loader, val_loader=val_loader)
    print(f"Training complete. Best checkpoint: {summary['best_checkpoint']}")
    print(f"Best validation loss: {summary['best_val_total']:.6f}")


if __name__ == "__main__":
    main()
