"""Trainer for visibility-conditioned phase 2 experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dynadiff_vlbi.data.feature_formatting import format_dirty_input, format_visibility_tensor
from dynadiff_vlbi.data.measurement_holdout import (
    build_structured_holdout_split,
    select_epoch_support_fraction,
)
from dynadiff_vlbi.training.losses import compute_emc_loss_dict, compute_phase2_loss_dict
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

    def _prepare_standard_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
        visibility_input = batch["visibility_input"].to(self.device, non_blocking=True)
        dirty_input = batch["dirty_input"].to(self.device, non_blocking=True)
        if (
            not self.config.model.include_dirty_input
            and self.config.model.model_type not in {"residual_visibility_refinement", "ccrr", "emc_ccrr"}
        ):
            dirty_input = None
        targets = batch["target"].to(self.device, non_blocking=True).unsqueeze(1)
        measurements = (
            batch["vis_real"].to(self.device, non_blocking=True).to(torch.float32)
            + 1j * batch["vis_imag"].to(self.device, non_blocking=True).to(torch.float32)
        ).to(torch.complex64)
        mask = batch["mask"].to(self.device, non_blocking=True).to(torch.float32)
        baseline_pairs = batch.get("baseline_pairs")
        if baseline_pairs is not None:
            baseline_pairs = baseline_pairs.to(self.device, non_blocking=True)
        frame_uv_indices = batch.get("frame_uv_indices")
        if frame_uv_indices is not None:
            frame_uv_indices = frame_uv_indices.to(self.device, non_blocking=True)
        return {
            "visibility_input": visibility_input,
            "dirty_input": dirty_input,
            "targets": targets,
            "model_measurements": measurements,
            "full_measurements": measurements,
            "model_mask": mask,
            "support_mask": mask,
            "target_mask": torch.zeros_like(mask),
            "baseline_pairs": baseline_pairs,
            "frame_uv_indices": frame_uv_indices,
        }

    def _prepare_emc_batch(
        self,
        batch: dict[str, torch.Tensor],
        support_fraction: float,
    ) -> dict[str, torch.Tensor | None]:
        vis_real_np = batch["vis_real"].detach().cpu().numpy().astype(np.float32)
        vis_imag_np = batch["vis_imag"].detach().cpu().numpy().astype(np.float32)
        mask_np = batch["mask"].detach().cpu().numpy().astype(np.float32)
        uv_coords_np = batch["uv_coords"].detach().cpu().numpy().astype(np.float32)
        frame_uv_indices_np = batch["frame_uv_indices"].detach().cpu().numpy().astype(np.int64)
        frame_uv_coords_np = batch["frame_uv_coords"].detach().cpu().numpy().astype(np.float32)
        sample_indices_np = batch["sample_index"].detach().cpu().numpy().astype(np.int64)
        baseline_pairs_np = None
        if "baseline_pairs" in batch:
            baseline_pairs_np = batch["baseline_pairs"].detach().cpu().numpy().astype(np.int64)
        station_positions_np = None
        if "station_positions" in batch:
            station_positions_np = batch["station_positions"].detach().cpu().numpy().astype(np.float32)

        visibility_inputs: list[torch.Tensor] = []
        dirty_inputs: list[torch.Tensor] = []
        support_measurements: list[np.ndarray] = []
        support_masks: list[np.ndarray] = []
        target_masks: list[np.ndarray] = []

        for batch_index in range(vis_real_np.shape[0]):
            full_measurements = (vis_real_np[batch_index] + 1j * vis_imag_np[batch_index]).astype(np.complex64)
            split = build_structured_holdout_split(
                measurements=full_measurements,
                observed_mask=mask_np[batch_index],
                frame_uv_indices=frame_uv_indices_np[batch_index],
                frame_uv_coords=frame_uv_coords_np[batch_index],
                baseline_pairs=None if baseline_pairs_np is None else baseline_pairs_np[batch_index],
                station_positions=None if station_positions_np is None else station_positions_np[batch_index],
                base_seed=self.config.project.seed,
                sample_index=int(sample_indices_np[batch_index]),
                support_fraction=support_fraction,
                strategy=self.config.holdout.strategy,
            )
            visibility_tensor = format_visibility_tensor(
                vis_real=vis_real_np[batch_index],
                vis_imag=vis_imag_np[batch_index],
                mask=split.support_mask,
                representation=self.config.model.visibility_representation,
                include_mask_channel=self.config.model.include_mask_channel,
                include_uv_coords=self.config.model.include_uv_coords,
                uv_coords=uv_coords_np[batch_index],
                include_observation_metadata=self.config.model.include_observation_metadata,
                frame_uv_coords=frame_uv_coords_np[batch_index],
                frame_uv_indices=frame_uv_indices_np[batch_index],
            )
            visibility_inputs.append(torch.from_numpy(visibility_tensor))
            dirty_inputs.append(torch.from_numpy(format_dirty_input(split.support_dirty)))
            support_measurements.append(split.support_measurements.astype(np.complex64))
            support_masks.append(split.support_mask.astype(np.float32))
            target_masks.append(split.target_mask.astype(np.float32))

        full_measurements = (
            batch["vis_real"].to(self.device, non_blocking=True).to(torch.float32)
            + 1j * batch["vis_imag"].to(self.device, non_blocking=True).to(torch.float32)
        ).to(torch.complex64)
        baseline_pairs = batch.get("baseline_pairs")
        if baseline_pairs is not None:
            baseline_pairs = baseline_pairs.to(self.device, non_blocking=True)
        frame_uv_indices = batch.get("frame_uv_indices")
        if frame_uv_indices is not None:
            frame_uv_indices = frame_uv_indices.to(self.device, non_blocking=True)
        return {
            "visibility_input": torch.stack(visibility_inputs, dim=0).to(self.device, non_blocking=True),
            "dirty_input": torch.stack(dirty_inputs, dim=0).to(self.device, non_blocking=True),
            "targets": batch["target"].to(self.device, non_blocking=True).unsqueeze(1),
            "model_measurements": torch.from_numpy(np.stack(support_measurements, axis=0)).to(
                self.device,
                non_blocking=True,
            ),
            "full_measurements": full_measurements,
            "model_mask": torch.from_numpy(np.stack(support_masks, axis=0)).to(self.device, non_blocking=True),
            "support_mask": torch.from_numpy(np.stack(support_masks, axis=0)).to(self.device, non_blocking=True),
            "target_mask": torch.from_numpy(np.stack(target_masks, axis=0)).to(self.device, non_blocking=True),
            "baseline_pairs": baseline_pairs,
            "frame_uv_indices": frame_uv_indices,
        }

    def _prepare_batch(
        self,
        batch: dict[str, torch.Tensor],
        support_fraction: float | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if self.config.model.model_type == "emc_ccrr" and self.config.holdout.enabled:
            return self._prepare_emc_batch(batch=batch, support_fraction=support_fraction or 1.0)
        return self._prepare_standard_batch(batch=batch)

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        temporal_loss_weight: float,
        heteroscedastic_loss_weight: float,
        visibility_loss_weight: float,
        closure_loss_weight: float,
        target_visibility_loss_weight: float,
        target_closure_loss_weight: float,
        support_fraction: float | None = None,
    ) -> dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        totals = {
            "total": 0.0,
            "reconstruction": 0.0,
            "temporal": 0.0,
            "heteroscedastic": 0.0,
            "visibility": 0.0,
            "closure": 0.0,
            "support_visibility": 0.0,
            "target_visibility": 0.0,
            "target_closure": 0.0,
        }
        num_batches = 0
        progress = tqdm(loader, leave=False, desc="train" if training else "val")
        for batch in progress:
            prepared = self._prepare_batch(batch, support_fraction=support_fraction)
            with torch.set_grad_enabled(training):
                outputs = self.model(
                    visibility_input=prepared["visibility_input"],
                    dirty_input=prepared["dirty_input"],
                    measurements=prepared["model_measurements"],
                    mask=prepared["model_mask"],
                    baseline_pairs=prepared["baseline_pairs"],
                    frame_uv_indices=prepared["frame_uv_indices"],
                )
                if self.config.model.model_type == "emc_ccrr":
                    losses = compute_emc_loss_dict(
                        prediction=outputs.mean,
                        target=prepared["targets"],
                        temporal_loss_weight=temporal_loss_weight,
                        measurements=prepared["full_measurements"],
                        support_mask=prepared["support_mask"],
                        target_mask=prepared["target_mask"],
                        log_variance=outputs.log_variance,
                        heteroscedastic_loss_weight=heteroscedastic_loss_weight,
                        consistency_prediction=getattr(outputs, "pre_dc_prediction", outputs.mean),
                        support_visibility_loss_weight=visibility_loss_weight,
                        target_visibility_loss_weight=target_visibility_loss_weight,
                        baseline_pairs=prepared["baseline_pairs"],
                        frame_uv_indices=prepared["frame_uv_indices"],
                        target_closure_loss_weight=target_closure_loss_weight,
                        closure_max_triangles=self.config.training.closure_max_triangles,
                    )
                else:
                    losses = compute_phase2_loss_dict(
                        prediction=outputs.mean,
                        target=prepared["targets"],
                        temporal_loss_weight=temporal_loss_weight,
                        log_variance=outputs.log_variance,
                        heteroscedastic_loss_weight=heteroscedastic_loss_weight,
                        consistency_prediction=getattr(outputs, "pre_dc_prediction", outputs.mean),
                        measurements=prepared["full_measurements"],
                        mask=prepared["model_mask"],
                        visibility_loss_weight=visibility_loss_weight,
                        baseline_pairs=prepared["baseline_pairs"],
                        frame_uv_indices=prepared["frame_uv_indices"],
                        closure_loss_weight=closure_loss_weight,
                        closure_max_triangles=self.config.training.closure_max_triangles,
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
            "active_temporal_weight",
            "active_heteroscedastic_weight",
            "active_visibility_weight",
            "active_closure_weight",
            "active_target_visibility_weight",
            "active_target_closure_weight",
            "active_support_fraction",
            "train_total",
            "train_reconstruction",
            "train_temporal",
            "train_heteroscedastic",
            "train_visibility",
            "train_closure",
            "train_support_visibility",
            "train_target_visibility",
            "train_target_closure",
            "val_total",
            "val_reconstruction",
            "val_temporal",
            "val_heteroscedastic",
            "val_visibility",
            "val_closure",
            "val_support_visibility",
            "val_target_visibility",
            "val_target_closure",
        ]

        for epoch in range(1, self.config.training.epochs + 1):
            warmup_active = epoch <= self.config.training.reconstruction_warmup_epochs
            temporal_weight = 0.0 if warmup_active else self.config.training.temporal_loss_weight
            heteroscedastic_weight = 0.0 if warmup_active else self.config.training.heteroscedastic_loss_weight
            visibility_weight = 0.0 if warmup_active else self.config.training.visibility_loss_weight
            closure_weight = 0.0 if warmup_active else self.config.training.closure_loss_weight
            target_visibility_weight = (
                0.0 if warmup_active else self.config.training.target_visibility_loss_weight
            )
            target_closure_weight = 0.0 if warmup_active else self.config.training.target_closure_loss_weight
            active_support_fraction = (
                select_epoch_support_fraction(self.config.holdout.train_support_fractions, epoch)
                if self.config.model.model_type == "emc_ccrr"
                else 1.0
            )
            train_metrics = self._run_epoch(
                train_loader,
                training=True,
                temporal_loss_weight=temporal_weight,
                heteroscedastic_loss_weight=heteroscedastic_weight,
                visibility_loss_weight=visibility_weight,
                closure_loss_weight=closure_weight,
                target_visibility_loss_weight=target_visibility_weight,
                target_closure_loss_weight=target_closure_weight,
                support_fraction=active_support_fraction,
            )
            val_metrics = self._run_epoch(
                val_loader,
                training=False,
                temporal_loss_weight=temporal_weight,
                heteroscedastic_loss_weight=heteroscedastic_weight,
                visibility_loss_weight=visibility_weight,
                closure_loss_weight=closure_weight,
                target_visibility_loss_weight=target_visibility_weight,
                target_closure_loss_weight=target_closure_weight,
                support_fraction=active_support_fraction,
            )
            row = {
                "epoch": epoch,
                "active_temporal_weight": temporal_weight,
                "active_heteroscedastic_weight": heteroscedastic_weight,
                "active_visibility_weight": visibility_weight,
                "active_closure_weight": closure_weight,
                "active_target_visibility_weight": target_visibility_weight,
                "active_target_closure_weight": target_closure_weight,
                "active_support_fraction": active_support_fraction,
                "train_total": train_metrics["total"],
                "train_reconstruction": train_metrics["reconstruction"],
                "train_temporal": train_metrics["temporal"],
                "train_heteroscedastic": train_metrics["heteroscedastic"],
                "train_visibility": train_metrics["visibility"],
                "train_closure": train_metrics["closure"],
                "train_support_visibility": train_metrics["support_visibility"],
                "train_target_visibility": train_metrics["target_visibility"],
                "train_target_closure": train_metrics["target_closure"],
                "val_total": val_metrics["total"],
                "val_reconstruction": val_metrics["reconstruction"],
                "val_temporal": val_metrics["temporal"],
                "val_heteroscedastic": val_metrics["heteroscedastic"],
                "val_visibility": val_metrics["visibility"],
                "val_closure": val_metrics["closure"],
                "val_support_visibility": val_metrics["support_visibility"],
                "val_target_visibility": val_metrics["target_visibility"],
                "val_target_closure": val_metrics["target_closure"],
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
