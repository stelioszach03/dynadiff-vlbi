"""Tests for Phase 5 prerequisite plumbing.

Covers:
  * ``resolve_partition_strategy`` deterministic / adaptive + env override
  * ``load_oracle_from_checkpoint`` round-trip (save, reload, same forward)
  * ``_inherit_from`` directive in ``load_experiment_config``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.data.measurement_holdout import (
    _ORACLE_CACHE,
    resolve_partition_strategy,
)
from dynadiff_vlbi.oracle import (
    HeavyHitterOracle,
    HeavyHitterOracleConfig,
    load_oracle_from_checkpoint,
)
from dynadiff_vlbi.utils.config import HoldoutConfig, load_experiment_config


def _mini_oracle():
    cfg = HeavyHitterOracleConfig(
        hidden_dim=32, num_heads=4, num_self_layers=1, num_cross_layers=1,
        mlp_ratio=1.5, dropout=0.0,
    )
    return HeavyHitterOracle(cfg), cfg


# ---------------------------------------------------------------------------
# resolve_partition_strategy
# ---------------------------------------------------------------------------


def test_resolver_returns_deterministic_strategy_by_default(monkeypatch):
    """Default HoldoutConfig -> (strategy, None). No oracle loaded."""
    monkeypatch.delenv("DYNADIFF_PARTITION_MODE", raising=False)
    monkeypatch.delenv("DYNADIFF_ORACLE_CKPT", raising=False)
    _ORACLE_CACHE.clear()
    cfg = HoldoutConfig(strategy="baseline_track_blocks", partition_mode="deterministic")
    strategy, oracle = resolve_partition_strategy(cfg)
    assert strategy == "baseline_track_blocks"
    assert oracle is None


def test_resolver_adaptive_requires_checkpoint(monkeypatch):
    monkeypatch.delenv("DYNADIFF_PARTITION_MODE", raising=False)
    monkeypatch.delenv("DYNADIFF_ORACLE_CKPT", raising=False)
    cfg = HoldoutConfig(strategy="baseline_track_blocks", partition_mode="adaptive")
    with pytest.raises(ValueError, match="adaptive.*checkpoint"):
        resolve_partition_strategy(cfg)


def test_resolver_env_override_beats_config(monkeypatch, tmp_path: Path):
    """DYNADIFF_PARTITION_MODE=adaptive + DYNADIFF_ORACLE_CKPT overrides config."""
    oracle, _ = _mini_oracle()
    ckpt_path = tmp_path / "mini.ckpt"
    torch.save(
        {
            "model_state_dict": oracle.state_dict(),
            "config": {"num_heads": 4},
        },
        ckpt_path,
    )
    monkeypatch.setenv("DYNADIFF_PARTITION_MODE", "adaptive")
    monkeypatch.setenv("DYNADIFF_ORACLE_CKPT", str(ckpt_path))
    _ORACLE_CACHE.clear()

    cfg = HoldoutConfig(strategy="baseline_track_blocks", partition_mode="deterministic")
    strategy, model = resolve_partition_strategy(cfg)
    assert strategy == "learned_oracle_importance"
    assert model is not None
    # Cache hit: second call returns the same object.
    strategy2, model2 = resolve_partition_strategy(cfg)
    assert model2 is model


def test_resolver_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("DYNADIFF_PARTITION_MODE", "nonsense")
    with pytest.raises(ValueError, match="Unsupported partition_mode"):
        resolve_partition_strategy(HoldoutConfig())


# ---------------------------------------------------------------------------
# load_oracle_from_checkpoint
# ---------------------------------------------------------------------------


def test_oracle_checkpoint_roundtrip_preserves_forward(tmp_path: Path):
    torch.manual_seed(0)
    oracle, cfg = _mini_oracle()
    oracle.train(False)
    ckpt_path = tmp_path / "oracle.ckpt"
    torch.save(
        {
            "model_state_dict": oracle.state_dict(),
            "config": {"num_heads": cfg.num_heads},
        },
        ckpt_path,
    )
    loaded = load_oracle_from_checkpoint(ckpt_path, device="cpu")
    # Same forward on a fixed input.
    sup_vis = torch.randn(2, 8, 3)
    sup_uv = torch.randn(2, 8, 3)
    tgt_uv = torch.randn(2, 4, 3)
    out_orig = oracle(sup_vis, sup_uv, tgt_uv)
    out_loaded = loaded(sup_vis, sup_uv, tgt_uv)
    torch.testing.assert_close(out_orig, out_loaded, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# _inherit_from
# ---------------------------------------------------------------------------


def test_inherit_from_merges_parent(tmp_path: Path):
    """A minimal overlay with _inherit_from: ... merges on top of the parent."""
    child = tmp_path / "child.yaml"
    child.write_text(
        "_inherit_from: configs/base.yaml\n"
        "project:\n"
        "  name: thesis_override_smoke\n"
        "holdout:\n"
        "  enabled: true\n"
        "  strategy: scan_segment_blocks\n"
        "  support_fraction: 0.6\n"
        "  partition_mode: adaptive\n"
    )
    cfg = load_experiment_config(
        base_path=child,
        train_path=ROOT / "configs" / "train.yaml",
        eval_path=ROOT / "configs" / "eval.yaml",
        preset=None,
        default_base_path=ROOT / "configs" / "base.yaml",
    )
    # Overlay fields applied.
    assert cfg.project.name == "thesis_override_smoke"
    assert cfg.holdout.enabled is True
    assert cfg.holdout.strategy == "scan_segment_blocks"
    assert cfg.holdout.partition_mode == "adaptive"
    assert cfg.holdout.support_fraction == pytest.approx(0.6)
    # Parent fields still present where overlay didn't override (e.g. image_size).
    assert cfg.dataset.image_size > 0  # came from parent


def test_inherit_from_detects_cycle(tmp_path: Path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(f"_inherit_from: {b}\nproject:\n  name: a\n")
    b.write_text(f"_inherit_from: {a}\nproject:\n  name: b\n")
    with pytest.raises(ValueError, match="Cyclic _inherit_from"):
        load_experiment_config(
            base_path=a,
            train_path=ROOT / "configs" / "train.yaml",
            eval_path=ROOT / "configs" / "eval.yaml",
            preset=None,
            default_base_path=a,
        )


# ---------------------------------------------------------------------------
# Integration: partition_mode='adaptive' drives learned_oracle_importance
# ---------------------------------------------------------------------------


def test_adaptive_config_triggers_learned_oracle_strategy(monkeypatch, tmp_path: Path):
    """When HoldoutConfig.partition_mode='adaptive' and oracle_checkpoint is set,
    the resolver hands back a HeavyHitterOracle and the learned_oracle_importance
    strategy -- that's exactly what evaluate_emc_condition needs to flip to the
    adaptive path."""
    monkeypatch.delenv("DYNADIFF_PARTITION_MODE", raising=False)
    monkeypatch.delenv("DYNADIFF_ORACLE_CKPT", raising=False)
    _ORACLE_CACHE.clear()

    oracle, _ = _mini_oracle()
    ckpt_path = tmp_path / "oracle.ckpt"
    torch.save({"model_state_dict": oracle.state_dict(), "config": {"num_heads": 4}}, ckpt_path)
    cfg = HoldoutConfig(
        strategy="baseline_track_blocks",
        partition_mode="adaptive",
        oracle_checkpoint=str(ckpt_path),
    )
    strategy, model = resolve_partition_strategy(cfg)
    assert strategy == "learned_oracle_importance"
    assert model is not None
