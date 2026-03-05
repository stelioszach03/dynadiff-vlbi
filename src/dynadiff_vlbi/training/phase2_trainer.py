"""Trainer for visibility-conditioned phase 2 experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dynadiff_vlbi.training.losses import compute_phase2_loss_dict
from dynadiff_vlbi.utils.config import ExperimentConfig
from dynadiff_vlbi.utils.logging_utils import append_csv_row, save_json, save_yaml


class Phase2Trainer:
    """Trainer for visibility-conditioned reconstruction models."""

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device,
        output_dirs: dict[str, Path],
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dirs = output_dirs
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.history_path = output_dirs["logs"] / "history.csv"
        self.summary_path = output_dirs["logs"] / "training_summary.json"
        save_yaml(output_dirs["logs"] / "config_snapshot.yaml", config.to_dict())

    def _prepare_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        visibility_input = batch["visibility_input"].to(self.device, non_blocking=True)
        dirty_input = batch["dirty_input"].to(self.device, non_blocking=True)
        if not self.config.model.include_dirty_input:
            dirty_input = None
        targets = batch["target"].to(self.device, non_blocking=True).unsqueeze(1)
        return visibility_input, dirty_input, targets

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        totals = {"total": 0.0, "reconstruction": 0.0, "temporal": 0.0, "heteroscedastic": 0.0}
        num_batches = 0
        progress = tqdm(loader, leave=False, desc="train" if training else "val")
        for batch in progress:
            visibility_input, dirty_input, targets = self._prepare_batch(batch)
            with torch.set_grad_enabled(training):
                outputs = self.model(visibility_input=visibility_input, dirty_input=dirty_input)
                losses = compute_phase2_loss_dict(
                    prediction=outputs.mean,
                    target=targets,
                    temporal_loss_weight=self.config.training.temporal_loss_weight,
                    log_variance=outputs.log_variance,
                    heteroscedastic_loss_weight=self.config.training.heteroscedastic_loss_weight,
                )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["total"].backward()
                    clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.grad_clip_norm)
                    self.optimizer.step()
            for key, value in losses.items():
                totals[key] += float(value.detach().item())
            num_batches += 1
            progress.set_postfix(loss=f"{losses['total'].detach().item():.4f}")

        if num_batches == 0:
            return {key: 0.0 for key in totals}
        return {key: value / num_batches for key, value in totals.items()}

    def _checkpoint_payload(self, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "metrics": metrics,
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        """Run the phase 2 training loop and save best/latest checkpoints."""

        best_val = float("inf")
        best_epoch = 0
        best_checkpoint_path = self.output_dirs["checkpoints"] / "best.pt"
        latest_checkpoint_path = self.output_dirs["checkpoints"] / "latest.pt"
        fields = [
            "epoch",
            "train_total",
            "train_reconstruction",
            "train_temporal",
            "train_heteroscedastic",
            "val_total",
            "val_reconstruction",
            "val_temporal",
            "val_heteroscedastic",
        ]

        for epoch in range(1, self.config.training.epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)
            row = {
                "epoch": epoch,
                "train_total": train_metrics["total"],
                "train_reconstruction": train_metrics["reconstruction"],
                "train_temporal": train_metrics["temporal"],
                "train_heteroscedastic": train_metrics["heteroscedastic"],
                "val_total": val_metrics["total"],
                "val_reconstruction": val_metrics["reconstruction"],
                "val_temporal": val_metrics["temporal"],
                "val_heteroscedastic": val_metrics["heteroscedastic"],
            }
            append_csv_row(self.history_path, fieldnames=fields, row=row)
            torch.save(self._checkpoint_payload(epoch, row), latest_checkpoint_path)
            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                best_epoch = epoch
                torch.save(self._checkpoint_payload(epoch, row), best_checkpoint_path)

        summary = {
            "best_val_total": best_val,
            "best_epoch": best_epoch,
            "best_checkpoint": str(best_checkpoint_path),
            "latest_checkpoint": str(latest_checkpoint_path),
        }
        save_json(self.summary_path, summary)
        return summary
