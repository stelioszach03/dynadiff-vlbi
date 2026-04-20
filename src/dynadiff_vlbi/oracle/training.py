"""Distillation training loop for :class:`HeavyHitterOracle`.

Given a dataset of measurement operators and support partitions, the
trainer runs teacher forcing:

    1. Teacher: ``I*(j | S) = a_j^H Sigma_post(S) a_j`` (signal-agnostic,
       depends only on UV coverage and the variance ratio).
    2. Student: ``I_hat = oracle(y_S, uv_S, uv_T)``.
    3. Loss: MSE in log space + ranking auxiliary on the top-k
       heavy-hitters.

The log-MSE term aligns the student with the teacher's ordinal structure
across several decades of importance values, while the ranking margin
loss sharpens the top-k decision boundary that governs adaptive support
selection at inference time.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


# ---------------------------------------------------------------------------
# Seed discipline
# ---------------------------------------------------------------------------


def set_oracle_seed(seed: int) -> None:
    """Seed every RNG source used by oracle training.

    Covers: ``random``, ``numpy``, PyTorch CPU, PyTorch CUDA (all devices),
    and the ``PYTHONHASHSEED`` env var (relevant when CPython creates new
    worker processes for the DataLoader).

    Call this once at the top of any oracle-training entry point. It makes
    fresh ``OracleTrainingConfig``-seeded runs byte-reproducible modulo
    non-determinism in cuDNN kernels, which is controlled separately by
    ``torch.use_deterministic_algorithms`` -- not enabled here because the
    multi-head attention kernels we rely on don't have deterministic
    implementations at the batch sizes we use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class OracleTrainingConfig:
    """Hyperparameters for oracle distillation."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 16
    num_epochs: int = 50
    ranking_weight: float = 0.1
    top_k: int = 8
    log_clip_min: float = 1e-8
    grad_clip_norm: float | None = 1.0

    # Synthetic-data generation (used by the default dataset in train_oracle).
    signal_var: float = 1.0
    noise_var: float = 0.01
    signal_dim: int = 256          # n
    num_measurements: int = 128    # m
    support_fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _safe_log(x: torch.Tensor, clip_min: float) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_min))


def log_mse_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    clip_min: float = 1e-8,
) -> torch.Tensor:
    """MSE in log space; robust to the wide dynamic range of I*."""
    return F.mse_loss(_safe_log(student, clip_min), _safe_log(teacher, clip_min))


def topk_ranking_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    top_k: int,
    margin: float = 0.1,
) -> torch.Tensor:
    """Margin-ranking auxiliary over heavy/light pairs from the teacher."""
    B, M = teacher.shape
    k = min(top_k, M // 2) if M > 1 else 1
    if k <= 0:
        return torch.zeros((), device=teacher.device, dtype=teacher.dtype)
    _, heavy_idx = torch.topk(teacher, k=k, dim=-1)
    _, light_idx = torch.topk(-teacher, k=k, dim=-1)

    batch_range = torch.arange(B, device=teacher.device).unsqueeze(-1)
    heavy_scores = student[batch_range, heavy_idx]   # [B, k]
    light_scores = student[batch_range, light_idx]   # [B, k]
    diff = heavy_scores.unsqueeze(-1) - light_scores.unsqueeze(-2)  # [B, k, k]
    hinge = F.relu(margin - diff)
    return hinge.mean()


# ---------------------------------------------------------------------------
# One gradient step
# ---------------------------------------------------------------------------


def distill_oracle_step(
    oracle: nn.Module,
    batch: Mapping[str, torch.Tensor],
    optimizer: optim.Optimizer,
    config: OracleTrainingConfig,
) -> dict[str, float]:
    """Single distillation step; returns scalar loss components."""
    oracle.train()
    optimizer.zero_grad()
    student = oracle(
        support_visibilities=batch["support_visibilities"],
        support_uv=batch["support_uv"],
        target_uv=batch["target_uv"],
        support_padding_mask=batch.get("support_padding_mask"),
    )
    teacher = batch["teacher_importance"]
    loss_mse = log_mse_loss(student, teacher, clip_min=config.log_clip_min)
    loss_rank = topk_ranking_loss(student, teacher, top_k=config.top_k)
    loss = loss_mse + config.ranking_weight * loss_rank
    loss.backward()
    if config.grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(oracle.parameters(), config.grad_clip_norm)
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "loss_mse": float(loss_mse.detach().item()),
        "loss_rank": float(loss_rank.detach().item()),
    }


# ---------------------------------------------------------------------------
# Evaluation: top-k recall of heavy hitters
# ---------------------------------------------------------------------------


@torch.no_grad()
def topk_recall(
    student: torch.Tensor,
    teacher: torch.Tensor,
    top_k: int,
) -> float:
    """Mean recall of the student's top-k against the teacher's top-k."""
    B, M = teacher.shape
    k = min(top_k, M)
    if k <= 0:
        return 0.0
    _, teacher_top = torch.topk(teacher, k=k, dim=-1)
    _, student_top = torch.topk(student, k=k, dim=-1)
    recalls = []
    for b in range(B):
        t = set(teacher_top[b].tolist())
        s = set(student_top[b].tolist())
        recalls.append(len(t & s) / float(k))
    return float(sum(recalls) / len(recalls))


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


def _run_validation(
    oracle: nn.Module,
    iterator_factory: Callable[[int], Iterable[Mapping[str, torch.Tensor]]],
    epoch: int,
    device: torch.device,
    config: OracleTrainingConfig,
) -> dict:
    """Return validation metrics over one validation epoch.

    Includes:
      - ``val_loss``: mean log-MSE across all batches
      - ``val_topk_recall``: mean top-k recall across all batches
      - ``recall_by_alpha``: dict mapping each training support fraction
        (e.g. 0.2, 0.4, 0.6, 0.8) to its bucketed top-k recall. Buckets
        are populated from ``batch['support_fraction']`` if the iterator
        emits it; missing tags collapse into an ``unknown`` bucket. This
        is what flags a partial-coverage failure mode (e.g. oracle good
        at alpha=0.8 but useless at alpha=0.2).
    """
    oracle.train(False)
    val_losses: list[float] = []
    recalls: list[float] = []
    alpha_recalls: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in iterator_factory(epoch):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            student = oracle(
                support_visibilities=batch["support_visibilities"],
                support_uv=batch["support_uv"],
                target_uv=batch["target_uv"],
                support_padding_mask=batch.get("support_padding_mask"),
            )
            teacher = batch["teacher_importance"]
            val_losses.append(
                float(log_mse_loss(student, teacher, clip_min=config.log_clip_min))
            )
            recall = topk_recall(student, teacher, config.top_k)
            recalls.append(recall)

            alpha_tag = batch.get("support_fraction")
            if isinstance(alpha_tag, torch.Tensor):
                alpha_tag = float(alpha_tag.flatten()[0].item())
            key = f"{alpha_tag:.2f}" if alpha_tag is not None else "unknown"
            alpha_recalls.setdefault(key, []).append(recall)

    val_loss = float(sum(val_losses) / max(len(val_losses), 1))
    val_recall = float(sum(recalls) / max(len(recalls), 1))
    recall_by_alpha = {
        k: float(sum(vs) / len(vs)) for k, vs in sorted(alpha_recalls.items())
    }
    return {
        "val_loss": val_loss,
        "val_topk_recall": val_recall,
        "recall_by_alpha": recall_by_alpha,
    }


def train_oracle(
    oracle: nn.Module,
    data_iterator_factory: Callable[[int], Iterable[Mapping[str, torch.Tensor]]],
    config: OracleTrainingConfig,
    validation_iterator_factory: Callable[[int], Iterable[Mapping[str, torch.Tensor]]] | None = None,
    device: torch.device | None = None,
    checkpoint_dir: Path | None = None,
    progress_callback: Callable[[int, dict], None] | None = None,
) -> dict:
    """Train the oracle to distill the teacher signal.

    Learning-rate schedule: fixed AdamW at ``config.learning_rate`` for
    the entire run. We deliberately do **not** add a warmup or cosine
    schedule in Phase 3 because (a) the target is a static posterior-
    variance vector, not a noisy gradient target, so warmup provides
    no stability gain, and (b) leaving the schedule flat keeps the
    50-epoch training run easy to reason about when we compare against
    the adaptive-partition benchmark in Phase 5. If training diverges
    on real VLBI data in Phase 4, reintroduce a linear warmup over the
    first 5 % of steps before adding cosine decay.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle.to(device)
    optimizer = optim.AdamW(
        oracle.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_loss = math.inf
    best_epoch = -1
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(config.num_epochs):
        train_losses = []
        for batch in data_iterator_factory(epoch):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            metrics = distill_oracle_step(oracle, batch, optimizer, config)
            train_losses.append(metrics["loss"])

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))

        val_loss = float("nan")
        val_recall = float("nan")
        recall_by_alpha: dict[str, float] = {}
        if validation_iterator_factory is not None:
            val_metrics = _run_validation(
                oracle, validation_iterator_factory, epoch, device, config
            )
            val_loss = val_metrics["val_loss"]
            val_recall = val_metrics["val_topk_recall"]
            recall_by_alpha = val_metrics["recall_by_alpha"]

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_topk_recall": val_recall,
            "recall_by_alpha": recall_by_alpha,
        }
        history.append(epoch_summary)
        if progress_callback is not None:
            progress_callback(epoch, epoch_summary)

        tracked = val_loss if not math.isnan(val_loss) else train_loss
        if tracked < best_loss:
            best_loss = tracked
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in oracle.state_dict().items()}

    if checkpoint_dir is not None and best_state is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / "best.ckpt"
        torch.save(
            {
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "history": history,
                "config": config.__dict__,
            },
            ckpt_path,
        )

    return {
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "history": history,
        "num_parameters": sum(p.numel() for p in oracle.parameters()),
    }
