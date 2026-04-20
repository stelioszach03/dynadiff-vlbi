#!/usr/bin/env python3
"""Train the HeavyHitterOracle via teacher distillation.

Usage (A100, full 50 epochs):
    python scripts/train_oracle.py \\
        --config configs/thesis_extension/oracle_default.yaml \\
        --output checkpoints/oracle/v1 \\
        --epochs 50

Usage (smoke test, 1 epoch, tiny synthetic data):
    python scripts/train_oracle.py --config ... --epochs 1 --samples-per-epoch 32
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.oracle.heavy_hitter_oracle import (
    HeavyHitterOracle,
    HeavyHitterOracleConfig,
)
from dynadiff_vlbi.oracle.teacher import compute_importance_teacher_batched
from dynadiff_vlbi.oracle.training import (
    OracleTrainingConfig,
    train_oracle,
)


# ---------------------------------------------------------------------------
# Synthetic-data generator used when no real VLBI dataset is bound to the
# oracle yet (i.e., Phase 3 standalone training). It emits a random partial
# DFT and random k-sparse signals per sample, then derives the teacher.
# ---------------------------------------------------------------------------


class _SyntheticPartialDFTEpoch:
    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        signal_dim: int,
        num_measurements: int,
        support_fractions: tuple[float, ...],
        signal_var: float,
        noise_var: float,
        device: torch.device,
        rng: np.random.Generator,
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.n = signal_dim
        self.m = num_measurements
        self.support_fractions = support_fractions
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.device = device
        self.rng = rng

    def _make_partial_dft(self, batch_size: int) -> torch.Tensor:
        # Shared frequency pool across batch to let self-attention learn
        # structure; per-sample mask lives in the support indices.
        freq_idx = self.rng.choice(self.n, size=self.m, replace=False)
        freqs = np.arange(self.n)
        angles = -2.0 * np.pi * np.outer(freq_idx, freqs) / self.n
        A_np = np.exp(1j * angles) / math.sqrt(self.n)
        A = torch.from_numpy(A_np).to(self.device).to(torch.complex64)
        return A.unsqueeze(0).expand(batch_size, -1, -1).contiguous(), freq_idx

    def _uv_of_freq(self, freq_idx: np.ndarray) -> torch.Tensor:
        # Fold 1-D frequency indices to a 2-D UV grid for consistency with
        # VLBI conventions; the third channel encodes frame time (=0 here).
        side = int(math.sqrt(self.n))
        if side * side != self.n:
            side = self.n  # fallback: all on a single axis
        u = (freq_idx % max(side, 1)) / max(side, 1) * 2.0 - 1.0
        v = (freq_idx // max(side, 1)) / max(side, 1) * 2.0 - 1.0
        t = np.zeros_like(u, dtype=float)
        return torch.tensor(np.stack([u, v, t], axis=-1), dtype=torch.float32, device=self.device)

    def __iter__(self):
        num_batches = max(1, self.num_samples // self.batch_size)
        for _ in range(num_batches):
            alpha = float(self.rng.choice(self.support_fractions))
            m_sup = max(1, int(round(self.m * alpha)))
            m_tgt = self.m - m_sup
            if m_tgt <= 0:
                continue

            A, freq_idx = self._make_partial_dft(self.batch_size)
            uv = self._uv_of_freq(freq_idx)  # [m, 3]

            perm = self.rng.permutation(self.m)
            support_idx_np = perm[:m_sup]
            target_idx_np = perm[m_sup:]
            support_idx = torch.tensor(support_idx_np, dtype=torch.long, device=self.device)
            target_idx = torch.tensor(target_idx_np, dtype=torch.long, device=self.device)

            # Draw k-sparse signals per batch item, compute y_S.
            k = max(1, self.m // 16)
            x = torch.zeros(self.batch_size, self.n, dtype=torch.float32, device=self.device)
            for b in range(self.batch_size):
                idx = self.rng.choice(self.n, size=k, replace=False)
                x[b, idx] = torch.randn(k, device=self.device) * math.sqrt(
                    self.signal_var * self.n / k
                )
            A_sup = A[:, support_idx, :]  # [B, m_sup, n]
            noise = (
                torch.randn(self.batch_size, m_sup, dtype=torch.complex64, device=self.device)
                * math.sqrt(self.noise_var / 2.0)
                + 1j
                * torch.randn(self.batch_size, m_sup, dtype=torch.complex64, device=self.device).real
                * math.sqrt(self.noise_var / 2.0)
            )
            y_sup = torch.einsum("bmn,bn->bm", A_sup, x.to(torch.complex64)) + noise

            # Teacher importance at every target row.
            support_idx_b = support_idx.unsqueeze(0).expand(self.batch_size, -1)
            target_idx_b = target_idx.unsqueeze(0).expand(self.batch_size, -1)
            teacher = compute_importance_teacher_batched(
                A,
                support_idx_b,
                target_idx_b,
                signal_var=self.signal_var,
                noise_var=self.noise_var,
            )

            support_vis = torch.stack(
                [y_sup.real, y_sup.imag, y_sup.abs()], dim=-1
            ).to(torch.float32)
            support_uv = uv[support_idx].unsqueeze(0).expand(self.batch_size, -1, -1)
            target_uv = uv[target_idx].unsqueeze(0).expand(self.batch_size, -1, -1)

            yield {
                "support_visibilities": support_vis,
                "support_uv": support_uv,
                "target_uv": target_uv,
                "teacher_importance": teacher,
            }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/thesis_extension/oracle_default.yaml",
        help="Path to the oracle YAML config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override checkpoint output directory (default: from config).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--val-samples-per-epoch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _oracle_config_from_yaml(cfg: dict) -> HeavyHitterOracleConfig:
    o = cfg.get("oracle", {})
    return HeavyHitterOracleConfig(
        hidden_dim=int(o.get("hidden_dim", 256)),
        num_heads=int(o.get("num_heads", 4)),
        num_self_layers=int(o.get("num_self_layers", 2)),
        num_cross_layers=int(o.get("num_cross_layers", 2)),
        dropout=float(o.get("dropout", 0.1)),
        mlp_ratio=float(o.get("mlp_ratio", 2.0)),
        vis_feature_dim=int(o.get("vis_feature_dim", 3)),
        uv_feature_dim=int(o.get("uv_feature_dim", 3)),
    )


def _training_config_from_yaml(cfg: dict) -> OracleTrainingConfig:
    t = cfg.get("training", {})
    d = cfg.get("data", {})
    return OracleTrainingConfig(
        learning_rate=float(t.get("learning_rate", 1e-4)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        batch_size=int(t.get("batch_size", 16)),
        num_epochs=int(t.get("num_epochs", 50)),
        ranking_weight=float(t.get("ranking_weight", 0.1)),
        top_k=int(t.get("top_k", 8)),
        log_clip_min=float(t.get("log_clip_min", 1e-8)),
        grad_clip_norm=(
            None if t.get("grad_clip_norm") is None else float(t.get("grad_clip_norm"))
        ),
        signal_var=float(d.get("signal_var", 1.0)),
        noise_var=float(d.get("noise_var", 0.01)),
        signal_dim=int(d.get("signal_dim", 256)),
        num_measurements=int(d.get("num_measurements", 128)),
        support_fractions=tuple(d.get("support_fractions", (0.2, 0.4, 0.6, 0.8))),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = _load_config(config_path)

    oracle_cfg = _oracle_config_from_yaml(cfg)
    training_cfg = _training_config_from_yaml(cfg)

    if args.epochs is not None:
        training_cfg.num_epochs = int(args.epochs)
    if args.batch_size is not None:
        training_cfg.batch_size = int(args.batch_size)

    samples_per_epoch = int(cfg.get("data", {}).get("samples_per_epoch", 256))
    val_samples_per_epoch = int(cfg.get("data", {}).get("val_samples_per_epoch", 64))
    if args.samples_per_epoch is not None:
        samples_per_epoch = int(args.samples_per_epoch)
    if args.val_samples_per_epoch is not None:
        val_samples_per_epoch = int(args.val_samples_per_epoch)

    seed = args.seed if args.seed is not None else int(cfg.get("project", {}).get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    output_dir = args.output or cfg.get("paths", {}).get("checkpoint_dir", "checkpoints/oracle/v1")
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    log_dir = cfg.get("paths", {}).get("log_dir", f"runs/{output_dir.name}")
    log_dir = Path(log_dir)
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    def train_factory(epoch: int):
        epoch_rng = np.random.default_rng(seed + epoch * 1000 + 1)
        return _SyntheticPartialDFTEpoch(
            num_samples=samples_per_epoch,
            batch_size=training_cfg.batch_size,
            signal_dim=training_cfg.signal_dim,
            num_measurements=training_cfg.num_measurements,
            support_fractions=training_cfg.support_fractions,
            signal_var=training_cfg.signal_var,
            noise_var=training_cfg.noise_var,
            device=device,
            rng=epoch_rng,
        )

    def val_factory(epoch: int):
        epoch_rng = np.random.default_rng(seed + 999983)  # fixed across epochs
        return _SyntheticPartialDFTEpoch(
            num_samples=val_samples_per_epoch,
            batch_size=training_cfg.batch_size,
            signal_dim=training_cfg.signal_dim,
            num_measurements=training_cfg.num_measurements,
            support_fractions=training_cfg.support_fractions,
            signal_var=training_cfg.signal_var,
            noise_var=training_cfg.noise_var,
            device=device,
            rng=epoch_rng,
        )

    oracle = HeavyHitterOracle(oracle_cfg)
    num_params = oracle.num_parameters()
    print(f"Oracle parameters: {num_params:,}")

    history_path = log_dir / "history.jsonl"

    def _cb(epoch: int, metrics: dict) -> None:
        with open(history_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(
            f"[epoch {epoch:03d}] train_loss={metrics['train_loss']:.6f} "
            f"val_loss={metrics['val_loss']:.6f} "
            f"val_topk_recall={metrics['val_topk_recall']:.4f}"
        )

    summary = train_oracle(
        oracle=oracle,
        data_iterator_factory=train_factory,
        config=training_cfg,
        validation_iterator_factory=val_factory,
        device=device,
        checkpoint_dir=output_dir,
        progress_callback=_cb,
    )

    summary_path = output_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "best_loss": summary["best_loss"],
                "best_epoch": summary["best_epoch"],
                "num_parameters": summary["num_parameters"],
                "training_config": asdict(training_cfg),
                "oracle_config": asdict(oracle_cfg),
            },
            f,
            indent=2,
        )

    print(f"Training complete. Best loss={summary['best_loss']:.6f} "
          f"at epoch {summary['best_epoch']}.")
    print(f"Checkpoint dir: {output_dir}")
    print(f"Log dir:        {log_dir}")
    _ = rng  # silence unused warning if ever reintroduced at top level


if __name__ == "__main__":
    main()
