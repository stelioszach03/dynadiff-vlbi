"""Tests for the learning-augmented heavy-hitter oracle (Phase 3).

Covers:
  1. Forward-pass shape correctness.
  2. Teacher determinism given fixed (A, support, variances).
  3. A single training step reduces loss on a tiny synthetic batch.
  4. Trained-oracle top-k recall exceeds random baseline.

All tests are deterministic (seeded) and CPU-only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.oracle import (
    HeavyHitterOracle,
    HeavyHitterOracleConfig,
    OracleTrainingConfig,
    compute_importance_teacher,
    compute_importance_teacher_batched,
    compute_posterior_covariance,
    distill_oracle_step,
    set_oracle_seed,
    train_oracle,
)
from dynadiff_vlbi.oracle.training import log_mse_loss, topk_recall, _run_validation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _partial_dft(n: int, m: int, rng: np.random.Generator) -> torch.Tensor:
    idx = rng.choice(n, size=m, replace=False)
    angles = -2 * math.pi * np.outer(idx, np.arange(n)) / n
    return torch.from_numpy(np.exp(1j * angles) / math.sqrt(n)).to(torch.complex64)


def _make_tiny_batch(
    oracle_cfg: HeavyHitterOracleConfig,
    rng: np.random.Generator,
    batch_size: int = 4,
    n: int = 64,
    m: int = 32,
    signal_var: float = 1.0,
    noise_var: float = 0.01,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Construct a (batch, support_idx, target_idx) triple for testing."""
    A = _partial_dft(n, m, rng)

    perm = rng.permutation(m)
    m_sup = m // 2
    sup_idx = torch.tensor(perm[:m_sup], dtype=torch.long)
    tgt_idx = torch.tensor(perm[m_sup:], dtype=torch.long)

    # Signal, noise -> support visibilities (one per batch item).
    x = torch.zeros(batch_size, n, dtype=torch.float32)
    k = max(1, m // 8)
    for b in range(batch_size):
        idx = rng.choice(n, size=k, replace=False)
        x[b, idx] = torch.randn(k) * math.sqrt(signal_var * n / k)

    A_sup = A[sup_idx]
    noise = (
        torch.randn(batch_size, m_sup, dtype=torch.complex64)
        * math.sqrt(noise_var / 2.0)
        + 1j * torch.randn(batch_size, m_sup, dtype=torch.complex64).real
        * math.sqrt(noise_var / 2.0)
    )
    y_sup = (A_sup.to(torch.complex64) @ x.to(torch.complex64).t()).t() + noise

    # Real-valued UV tokens: here we just encode the 1-D frequency index
    # folded to 2-D + a time channel of 0.
    side = int(math.sqrt(n)) if int(math.sqrt(n)) ** 2 == n else n
    idx_pool = np.arange(m)  # placeholder — actual UV geometry is irrelevant for shape tests
    u = (idx_pool % max(side, 1)) / max(side, 1) * 2.0 - 1.0
    v = (idx_pool // max(side, 1)) / max(side, 1) * 2.0 - 1.0
    t = np.zeros_like(u, dtype=float)
    uv_all = torch.tensor(np.stack([u, v, t], axis=-1), dtype=torch.float32)
    uv_sup = uv_all[sup_idx].unsqueeze(0).expand(batch_size, -1, -1)
    uv_tgt = uv_all[tgt_idx].unsqueeze(0).expand(batch_size, -1, -1)

    support_vis = torch.stack([y_sup.real, y_sup.imag, y_sup.abs()], dim=-1).to(torch.float32)

    # Teacher importance (same for all batch items since A/support are shared).
    teacher_single = compute_importance_teacher(
        A[tgt_idx], A[sup_idx], signal_var=signal_var, noise_var=noise_var
    )
    teacher = teacher_single.unsqueeze(0).expand(batch_size, -1).clone()

    batch = {
        "support_visibilities": support_vis,
        "support_uv": uv_sup,
        "target_uv": uv_tgt,
        "teacher_importance": teacher,
    }
    return batch, sup_idx, tgt_idx


# ---------------------------------------------------------------------------
# 1. Forward pass shape
# ---------------------------------------------------------------------------


def test_oracle_forward_output_shape_matches_target_dim():
    cfg = HeavyHitterOracleConfig(
        hidden_dim=128, num_heads=4, num_self_layers=1, num_cross_layers=1, mlp_ratio=1.5
    )
    oracle = HeavyHitterOracle(cfg)
    B, M_sup, M_tgt = 3, 40, 12
    out = oracle(
        support_visibilities=torch.randn(B, M_sup, 3),
        support_uv=torch.randn(B, M_sup, 3),
        target_uv=torch.randn(B, M_tgt, 3),
    )
    assert out.shape == (B, M_tgt)
    assert torch.all(out >= 0), "importance must be non-negative"
    assert torch.all(torch.isfinite(out))


def test_oracle_parameter_count_in_target_band():
    """Default-config oracle has 0.5M - 2M parameters."""
    cfg = HeavyHitterOracleConfig(
        hidden_dim=256, num_heads=4, num_self_layers=2, num_cross_layers=1, mlp_ratio=1.5
    )
    oracle = HeavyHitterOracle(cfg)
    n = oracle.num_parameters()
    assert 500_000 <= n <= 2_000_000, f"parameter count {n:,} outside [0.5M, 2M]"


def test_oracle_rejects_malformed_inputs():
    cfg = HeavyHitterOracleConfig(hidden_dim=32, num_heads=4, num_self_layers=1, num_cross_layers=1)
    oracle = HeavyHitterOracle(cfg)
    # wrong support visibility feature dim
    with pytest.raises(ValueError):
        oracle(torch.randn(1, 4, 5), torch.randn(1, 4, 3), torch.randn(1, 2, 3))
    # batch mismatch
    with pytest.raises(ValueError):
        oracle(torch.randn(1, 4, 3), torch.randn(1, 4, 3), torch.randn(2, 2, 3))


# ---------------------------------------------------------------------------
# 2. Teacher determinism
# ---------------------------------------------------------------------------


def test_teacher_is_deterministic_for_fixed_inputs():
    rng = np.random.default_rng(7)
    A = _partial_dft(n=64, m=32, rng=rng)
    sup = torch.arange(16)
    tgt = torch.arange(16, 32)
    t1 = compute_importance_teacher(A[tgt], A[sup], signal_var=1.0, noise_var=0.01)
    t2 = compute_importance_teacher(A[tgt], A[sup], signal_var=1.0, noise_var=0.01)
    assert torch.equal(t1, t2)
    # Teacher depends only on the operator + variances, so different signals
    # (implicit here since compute_importance_teacher does not see x) should
    # still produce identical outputs.
    t3 = compute_importance_teacher(A[tgt], A[sup], signal_var=1.0, noise_var=0.01)
    assert torch.equal(t1, t3)


def test_teacher_importance_is_positive_and_finite():
    rng = np.random.default_rng(13)
    A = _partial_dft(n=128, m=64, rng=rng)
    sup = torch.arange(32)
    tgt = torch.arange(32, 64)
    I = compute_importance_teacher(A[tgt], A[sup], signal_var=1.0, noise_var=0.01)
    assert torch.all(I > 0)
    assert torch.all(torch.isfinite(I))
    # Posterior covariance must be SPD.
    Sigma = compute_posterior_covariance(A[sup], signal_var=1.0, noise_var=0.01)
    eigvals = torch.linalg.eigvalsh(Sigma)
    assert torch.all(eigvals > 0)


def test_teacher_batched_matches_single():
    rng = np.random.default_rng(42)
    A = _partial_dft(n=64, m=32, rng=rng)
    B = 3
    A_batch = A.unsqueeze(0).expand(B, -1, -1).contiguous()
    sup_idx = torch.arange(16).unsqueeze(0).expand(B, -1)
    tgt_idx = torch.arange(16, 32).unsqueeze(0).expand(B, -1)
    batched = compute_importance_teacher_batched(
        A_batch, sup_idx, tgt_idx, signal_var=1.0, noise_var=0.01
    )
    single = compute_importance_teacher(
        A[torch.arange(16, 32)], A[torch.arange(16)], signal_var=1.0, noise_var=0.01
    )
    for b in range(B):
        torch.testing.assert_close(batched[b], single, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Training step reduces loss
# ---------------------------------------------------------------------------


def test_single_training_step_reduces_loss():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = HeavyHitterOracleConfig(
        hidden_dim=64, num_heads=4, num_self_layers=1, num_cross_layers=1, mlp_ratio=1.5,
        dropout=0.0,
    )
    oracle = HeavyHitterOracle(cfg)
    batch, _, _ = _make_tiny_batch(cfg, rng, batch_size=4, n=64, m=32)

    optim = torch.optim.AdamW(oracle.parameters(), lr=1e-2)
    training_cfg = OracleTrainingConfig(batch_size=4, top_k=4, grad_clip_norm=None)

    # Reference loss BEFORE any step (deterministic forward pass).
    oracle.train(False)
    with torch.no_grad():
        student0 = oracle(
            batch["support_visibilities"], batch["support_uv"], batch["target_uv"]
        )
        loss_before = float(log_mse_loss(student0, batch["teacher_importance"]))

    # Run a handful of steps on the same batch.
    losses = []
    for _ in range(5):
        metrics = distill_oracle_step(oracle, batch, optim, training_cfg)
        losses.append(metrics["loss"])

    oracle.train(False)
    with torch.no_grad():
        student_after = oracle(
            batch["support_visibilities"], batch["support_uv"], batch["target_uv"]
        )
        loss_after = float(log_mse_loss(student_after, batch["teacher_importance"]))

    assert loss_after < loss_before, (
        f"training did not reduce loss: before={loss_before:.4f} "
        f"after={loss_after:.4f} (traj={losses})"
    )


# ---------------------------------------------------------------------------
# 4. Top-k recall exceeds random baseline after training
# ---------------------------------------------------------------------------


def _random_topk_recall(teacher: torch.Tensor, top_k: int, num_samples: int, rng: np.random.Generator) -> float:
    """Estimate the chance-rate top-k recall by drawing random top-k sets."""
    B, M = teacher.shape
    k = min(top_k, M)
    _, teacher_top = torch.topk(teacher, k=k, dim=-1)
    recalls = []
    for _ in range(num_samples):
        for b in range(B):
            picked = set(rng.choice(M, size=k, replace=False).tolist())
            truth = set(teacher_top[b].tolist())
            recalls.append(len(picked & truth) / float(k))
    return float(sum(recalls) / max(len(recalls), 1))


def test_trained_oracle_topk_recall_exceeds_random_baseline():
    """After training the oracle should pick the heavy hitters materially
    above chance. We train on a fixed UV geometry + support partition so
    the teacher importance is stable across samples; the oracle must learn
    to produce a consistent ranking that agrees with the teacher on the
    top-k set.
    """
    torch.manual_seed(123)
    rng = np.random.default_rng(123)

    cfg = HeavyHitterOracleConfig(
        hidden_dim=64, num_heads=4, num_self_layers=1, num_cross_layers=1, mlp_ratio=1.5,
        dropout=0.0,
    )
    oracle = HeavyHitterOracle(cfg)
    # Use m = n (full DFT, different partition) so the teacher has a
    # well-defined ordering on the target rows rather than being dominated
    # by the null-space degeneracy of a rank-deficient support operator.
    n, m = 32, 32
    signal_var, noise_var = 1.0, 0.01
    top_k = 4

    batch, _, _ = _make_tiny_batch(
        cfg, rng, batch_size=8, n=n, m=m, signal_var=signal_var, noise_var=noise_var
    )
    optim = torch.optim.AdamW(oracle.parameters(), lr=5e-3)
    training_cfg = OracleTrainingConfig(batch_size=8, top_k=top_k, grad_clip_norm=None)
    for _ in range(200):
        distill_oracle_step(oracle, batch, optim, training_cfg)

    oracle.train(False)
    with torch.no_grad():
        student = oracle(
            batch["support_visibilities"], batch["support_uv"], batch["target_uv"]
        )
    recall = topk_recall(student, batch["teacher_importance"], top_k=top_k)
    random_baseline = _random_topk_recall(
        batch["teacher_importance"], top_k=top_k, num_samples=200, rng=rng
    )

    # The teacher importance has a high-value ("unobserved") cluster and a
    # low-value ("well-observed") cluster, so a correctly trained oracle
    # should recall the heavy hitters materially above chance; 0.15 is a
    # conservative margin relative to the ~0.25 lift observed empirically.
    assert recall > random_baseline + 0.15, (
        f"oracle top-k recall ({recall:.3f}) does not meaningfully exceed "
        f"random baseline ({random_baseline:.3f})"
    )


# ---------------------------------------------------------------------------
# Bonus: end-to-end train_oracle runs and produces a checkpoint
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 4 pre-items: seed discipline + per-alpha recall bucketing
# ---------------------------------------------------------------------------


def test_set_oracle_seed_reproduces_oracle_weights():
    """Same seed -> bit-identical oracle initial weights + identical forward."""
    cfg = HeavyHitterOracleConfig(
        hidden_dim=32, num_heads=4, num_self_layers=1, num_cross_layers=1, mlp_ratio=1.5,
        dropout=0.0,
    )
    set_oracle_seed(424242)
    oracle_a = HeavyHitterOracle(cfg)
    x_a = torch.randn(2, 8, 3)
    set_oracle_seed(424242)
    oracle_b = HeavyHitterOracle(cfg)
    x_b = torch.randn(2, 8, 3)
    # Tensor inputs drawn after seed reset should match.
    torch.testing.assert_close(x_a, x_b)
    # Every parameter should match bit-for-bit.
    for (na, pa), (nb, pb) in zip(oracle_a.named_parameters(), oracle_b.named_parameters()):
        assert na == nb
        torch.testing.assert_close(pa, pb, rtol=0.0, atol=0.0)


def test_set_oracle_seed_different_seeds_diverge():
    """Different seeds -> different oracle weights (sanity)."""
    cfg = HeavyHitterOracleConfig(
        hidden_dim=32, num_heads=4, num_self_layers=1, num_cross_layers=1, mlp_ratio=1.5,
        dropout=0.0,
    )
    set_oracle_seed(0)
    oracle_a = HeavyHitterOracle(cfg)
    set_oracle_seed(1)
    oracle_b = HeavyHitterOracle(cfg)
    # At least one parameter tensor should differ.
    any_different = False
    for (_, pa), (_, pb) in zip(oracle_a.named_parameters(), oracle_b.named_parameters()):
        if not torch.equal(pa, pb):
            any_different = True
            break
    assert any_different, "seed change did not perturb oracle weights"


def test_validation_bucketing_reports_recall_per_support_fraction():
    """``_run_validation`` should bucket top-k recall by the
    ``support_fraction`` tag emitted by the data iterator, so we can
    read per-alpha quality (0.2 / 0.4 / 0.6 / 0.8) at a glance.
    """
    cfg = HeavyHitterOracleConfig(
        hidden_dim=32, num_heads=4, num_self_layers=1, num_cross_layers=1, mlp_ratio=1.5,
        dropout=0.0,
    )
    oracle = HeavyHitterOracle(cfg)
    rng = np.random.default_rng(0)
    batches_by_alpha = {}
    for alpha in [0.2, 0.4, 0.6, 0.8]:
        batch, _, _ = _make_tiny_batch(cfg, rng, batch_size=2, n=32, m=16)
        batch["support_fraction"] = torch.tensor(alpha)
        batches_by_alpha[alpha] = batch

    def factory(epoch: int):
        # Return one batch per training alpha in a deterministic order.
        return [batches_by_alpha[a] for a in [0.2, 0.4, 0.6, 0.8]]

    training_cfg = OracleTrainingConfig(batch_size=2, top_k=3, grad_clip_norm=None)
    metrics = _run_validation(
        oracle, factory, epoch=0, device=torch.device("cpu"), config=training_cfg
    )
    assert set(metrics.keys()) == {"val_loss", "val_topk_recall", "recall_by_alpha"}
    # We should have one bucket per distinct alpha we emitted.
    buckets = metrics["recall_by_alpha"]
    assert set(buckets.keys()) == {"0.20", "0.40", "0.60", "0.80"}
    # All recalls finite in [0, 1].
    for a, r in buckets.items():
        assert 0.0 <= r <= 1.0, f"recall for alpha={a} out of range: {r}"


def test_train_oracle_produces_checkpoint(tmp_path: Path):
    torch.manual_seed(2024)
    rng = np.random.default_rng(2024)
    cfg = HeavyHitterOracleConfig(
        hidden_dim=32, num_heads=4, num_self_layers=1, num_cross_layers=1, mlp_ratio=1.5,
        dropout=0.0,
    )
    oracle = HeavyHitterOracle(cfg)
    batch, _, _ = _make_tiny_batch(cfg, rng, batch_size=4, n=32, m=16)

    def factory(epoch: int):
        # Two identical mini-batches per epoch so the loop has something to do.
        return [batch, batch]

    training_cfg = OracleTrainingConfig(
        batch_size=4, num_epochs=2, grad_clip_norm=None, top_k=3
    )
    summary = train_oracle(
        oracle=oracle,
        data_iterator_factory=factory,
        config=training_cfg,
        validation_iterator_factory=factory,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
    )
    ckpt = tmp_path / "best.ckpt"
    assert ckpt.exists()
    assert summary["best_epoch"] >= 0
    assert summary["num_parameters"] == oracle.num_parameters()
