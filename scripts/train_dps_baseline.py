#!/usr/bin/env python3
"""Train the lightweight DPS baseline on the shared default64 synthetic split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.emc.baselines.dps_baseline import (  # noqa: E402
    DPSBaseline,
    DPSCheckpointConfig,
    DPSScoreUNet,
    save_dps_checkpoint,
)
from dynadiff_vlbi.utils.device import get_device  # noqa: E402
from dynadiff_vlbi.utils.seed import set_seed  # noqa: E402


class FrameDataset(Dataset):
    """Flatten dynamic frame stacks into individual 64x64 grayscale frames."""

    def __init__(self, npz_path: str | Path) -> None:
        with np.load(Path(npz_path)) as payload:
            frames = payload["ground_truth"].astype(np.float32)
        self.frames = frames.reshape(-1, 1, frames.shape[-2], frames.shape[-1])

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.frames[index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/generated/ccrr_default64_seed7_shared")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name", default="dps_default64_baseline_tracks")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--lambda-start", type=float, default=0.3)
    parser.add_argument("--lambda-end", type=float, default=0.05)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _write_history_row(path: Path, row: dict[str, object]) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "best_val_loss"]
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    data_dir = (ROOT / args.data_dir).resolve()
    output_root = (ROOT / args.output_root).resolve()
    run_root = output_root / args.run_name
    checkpoints_dir = run_root / "checkpoints"
    logs_dir = run_root / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = FrameDataset(data_dir / "train.npz")
    val_dataset = FrameDataset(data_dir / "val.npz")
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    device = get_device()
    config = DPSCheckpointConfig(
        image_size=64,
        base_channels=int(args.base_channels),
        timesteps=int(args.timesteps),
        ddim_steps=int(args.ddim_steps),
        lambda_start=float(args.lambda_start),
        lambda_end=float(args.lambda_end),
    )
    score_model = DPSScoreUNet(image_channels=1, base_channels=config.base_channels).to(device)
    baseline = DPSBaseline(
        score_model,
        total_timesteps=config.timesteps,
        ddim_steps=config.ddim_steps,
        lambda_start=config.lambda_start,
        lambda_end=config.lambda_end,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    optimizer = torch.optim.AdamW(
        score_model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    best_val_loss = float("inf")
    history_path = logs_dir / "history.csv"
    for epoch in range(1, int(args.epochs) + 1):
        score_model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = baseline.training_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().item()))

        score_model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_loss = baseline.training_loss(batch)
                val_losses.append(float(val_loss.detach().item()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dps_checkpoint(
                path=checkpoints_dir / "best.pt",
                score_model=score_model,
                config=config,
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch,
                best_val_loss=best_val_loss,
            )
        save_dps_checkpoint(
            path=checkpoints_dir / "latest.pt",
            score_model=score_model,
            config=config,
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            best_val_loss=best_val_loss,
        )
        _write_history_row(
            history_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            },
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                }
            ),
            flush=True,
        )

    (logs_dir / "config_snapshot.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "run_name": args.run_name,
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "seed": int(args.seed),
                "dps_config": config.__dict__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"best_checkpoint": str(checkpoints_dir / "best.pt"), "best_val_loss": best_val_loss}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
