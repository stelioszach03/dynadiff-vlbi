#!/usr/bin/env python3
"""Train the compact temporal U-Net baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.data.dataset import build_dataloaders
from dynadiff_vlbi.models.temporal_unet import TemporalUNet3D
from dynadiff_vlbi.training.trainer import Trainer
from dynadiff_vlbi.utils.config import load_experiment_config
from dynadiff_vlbi.utils.device import get_device
from dynadiff_vlbi.utils.logging_utils import prepare_output_dirs
from dynadiff_vlbi.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--preset", default="smoke", choices=["smoke", "default32", "exp64"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for quick experiments.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(
        base_path=ROOT / args.base_config,
        train_path=ROOT / args.train_config,
        eval_path=ROOT / args.eval_config,
        preset=args.preset,
    )
    if args.epochs is not None:
        config.training.epochs = int(args.epochs)
    set_seed(config.project.seed)

    data_dir = Path(args.data_dir) if args.data_dir else ROOT / config.paths.data_root / args.preset
    if not (data_dir / "train.npz").exists():
        raise FileNotFoundError(
            f"Missing dataset split under {data_dir}. Run scripts/generate_toy_dataset.py first."
        )

    output_root = Path(args.output_root) if args.output_root else ROOT / config.paths.output_root
    run_name = args.run_name or f"train_{args.preset}"
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
