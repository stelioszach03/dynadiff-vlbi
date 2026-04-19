#!/usr/bin/env python3
"""Train the baseline or phase 2 reconstruction model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_BASE_CONFIG = "configs/base.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.data.dataset import build_dataloaders
from dynadiff_vlbi.data.visibility_dataset import build_visibility_dataloaders
from dynadiff_vlbi.models.factory import build_model
from dynadiff_vlbi.training.phase2_trainer import Phase2Trainer
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


def _resolve_optional_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def _initialize_backbone_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {}).get("model", {})
    if checkpoint_config.get("model_type") != "baseline":
        raise ValueError(
            "This model expects a baseline 3D U-Net checkpoint for backbone initialization."
        )
    if not hasattr(model, "backbone"):
        raise ValueError("The selected model does not expose a backbone for initialization.")
    model.backbone.load_state_dict(checkpoint["model_state_dict"])
    if getattr(model, "freeze_backbone", False):
        model.backbone.eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--preset", default=None, choices=["smoke", "default32", "default64", "exp64"])
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["baseline", "visibility_conditioned", "residual_visibility_refinement", "ccrr", "emc_ccrr"],
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--backbone-checkpoint", default=None)
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
    if args.model_type is not None:
        config.model.model_type = args.model_type
    if args.epochs is not None:
        config.training.epochs = int(args.epochs)
    backbone_checkpoint = _resolve_optional_path(args.backbone_checkpoint or config.training.backbone_checkpoint)
    if backbone_checkpoint is not None:
        config.training.backbone_checkpoint = str(backbone_checkpoint)
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
    if config.model.model_type == "baseline":
        train_loader, val_loader, _ = build_dataloaders(
            data_dir=data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
        )
        trainer_cls = Trainer
    elif config.model.model_type in {"visibility_conditioned", "residual_visibility_refinement", "ccrr", "emc_ccrr"}:
        train_loader, val_loader, _ = build_visibility_dataloaders(
            data_dir=data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            model_config=config.model,
        )
        trainer_cls = Phase2Trainer
    else:
        raise ValueError(f"Unsupported model_type '{config.model.model_type}'.")

    model = build_model(config.model)
    if config.model.model_type in {"residual_visibility_refinement", "ccrr", "emc_ccrr"}:
        if backbone_checkpoint is None:
            raise FileNotFoundError(
                "This model requires --backbone-checkpoint or training.backbone_checkpoint."
            )
        if not backbone_checkpoint.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_checkpoint}")
        _initialize_backbone_from_checkpoint(model=model, checkpoint_path=backbone_checkpoint)
    trainer = trainer_cls(model=model, config=config, device=get_device(), output_dirs=output_dirs)
    summary = trainer.fit(train_loader=train_loader, val_loader=val_loader)
    print(f"Training complete. Best checkpoint: {summary['best_checkpoint']}")
    print(f"Best validation loss: {summary['best_val_total']:.6f}")


if __name__ == "__main__":
    main()
