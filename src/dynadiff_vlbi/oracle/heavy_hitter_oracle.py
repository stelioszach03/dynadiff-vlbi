"""HeavyHitterOracle — learned importance predictor for adaptive partitions.

The oracle takes an already-observed (support) visibility set and a set of
query UV positions (candidate targets) and returns a scalar importance
score for each query, predicting how much the posterior variance at that
query would shrink if it were moved into the support.

Architecture (mirrors ``MeasurementCrossAttention`` in
``dynadiff_vlbi/models/score_diffusion.py``):

    1. Visibility tokenisation:
         y_sup  (real, imag, |y|)            -> [B, M_sup, 3]
         uv_sup (u, v, t)                    -> [B, M_sup, 3]
         concat, project                     -> [B, M_sup, hidden_dim]
    2. Self-attention stack over support tokens (``num_self_layers``).
    3. Cross-attention from query tokens (uv_tgt -> [B, M_tgt, hidden_dim])
       to support tokens (``num_cross_layers``).
    4. Scalar head: linear + softplus -> importance in (0, +inf).

The hidden dimension, attention heads, and depth are chosen so that the
parameter count sits between 0.5 M and 2 M with the default config
(hidden_dim=256, num_heads=4, num_self_layers=num_cross_layers=2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class HeavyHitterOracleConfig:
    """Hyperparameters for :class:`HeavyHitterOracle`."""

    hidden_dim: int = 256
    num_heads: int = 4
    num_self_layers: int = 2
    num_cross_layers: int = 2
    dropout: float = 0.1
    mlp_ratio: float = 2.0

    vis_feature_dim: int = 3  # (real, imag, |y|)
    uv_feature_dim: int = 3   # (u, v, t)

    def total_input_dim(self) -> int:
        return self.vis_feature_dim + self.uv_feature_dim

    def query_input_dim(self) -> int:
        return self.uv_feature_dim


# ---------------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------------


class _SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + MLP block, single-dim tokens [B, N, D]."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + MLP block. Queries attend to a context."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm_q(queries)
        ctx = self.norm_ctx(context)
        attn_out, _ = self.attn(
            q, ctx, ctx, key_padding_mask=context_key_padding_mask, need_weights=False
        )
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm2(queries))
        return queries


# ---------------------------------------------------------------------------
# HeavyHitterOracle
# ---------------------------------------------------------------------------


class HeavyHitterOracle(nn.Module):
    """Predict per-query importance scores from support visibilities.

    Inputs to :meth:`forward`:

    * ``support_visibilities``: real tensor of shape ``[B, M_sup, 3]``
      encoding ``(real(y), imag(y), |y|)`` at each support row.
    * ``support_uv``: real tensor of shape ``[B, M_sup, 3]`` encoding
      ``(u, v, t)`` of each support row (normalised to roughly [-1, 1]).
    * ``target_uv``: real tensor of shape ``[B, M_tgt, 3]`` with the
      ``(u, v, t)`` coordinates for each candidate target row.
    * ``support_padding_mask``: optional bool tensor ``[B, M_sup]``;
      ``True`` marks a padding entry that should be ignored.

    Output: ``importance`` of shape ``[B, M_tgt]`` in (0, +inf), trained to
    approximate the teacher signal
    ``I^*(j | S) = a_j^H Sigma_post(S) a_j`` (see
    ``dynadiff_vlbi.oracle.teacher``).
    """

    def __init__(self, config: HeavyHitterOracleConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeavyHitterOracleConfig()
        cfg = self.config

        self.support_embed = nn.Sequential(
            nn.Linear(cfg.total_input_dim(), cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.query_embed = nn.Sequential(
            nn.Linear(cfg.query_input_dim(), cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        self.self_blocks = nn.ModuleList(
            [
                _SelfAttentionBlock(
                    dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout,
                    mlp_ratio=cfg.mlp_ratio,
                )
                for _ in range(cfg.num_self_layers)
            ]
        )
        self.cross_blocks = nn.ModuleList(
            [
                _CrossAttentionBlock(
                    dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout,
                    mlp_ratio=cfg.mlp_ratio,
                )
                for _ in range(cfg.num_cross_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------
    # Input encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize_visibilities(
        support_visibilities: torch.Tensor,
        support_uv: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate visibility and UV features into a single token."""
        return torch.cat([support_visibilities, support_uv], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        support_visibilities: torch.Tensor,
        support_uv: torch.Tensor,
        target_uv: torch.Tensor,
        support_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return importance scores of shape ``[B, M_tgt]``."""
        cfg = self.config
        self._check_inputs(support_visibilities, support_uv, target_uv)

        support_tokens = self.support_embed(
            self.tokenize_visibilities(support_visibilities, support_uv)
        )
        query_tokens = self.query_embed(target_uv)

        for block in self.self_blocks:
            support_tokens = block(support_tokens, key_padding_mask=support_padding_mask)

        for block in self.cross_blocks:
            query_tokens = block(
                query_tokens,
                support_tokens,
                context_key_padding_mask=support_padding_mask,
            )

        logits = self.head(query_tokens).squeeze(-1)
        # Importance is non-negative; softplus keeps gradients non-zero near 0.
        importance = F.softplus(logits)
        _ = cfg  # reserved for future per-call dispatch
        return importance

    def _check_inputs(
        self,
        support_visibilities: torch.Tensor,
        support_uv: torch.Tensor,
        target_uv: torch.Tensor,
    ) -> None:
        cfg = self.config
        if support_visibilities.dim() != 3:
            raise ValueError(
                f"support_visibilities expected [B, M_sup, {cfg.vis_feature_dim}], "
                f"got shape {tuple(support_visibilities.shape)}"
            )
        if support_visibilities.shape[-1] != cfg.vis_feature_dim:
            raise ValueError(
                f"support_visibilities last dim must be {cfg.vis_feature_dim}, "
                f"got {support_visibilities.shape[-1]}"
            )
        if support_uv.shape[:2] != support_visibilities.shape[:2]:
            raise ValueError(
                "support_uv and support_visibilities must share [B, M_sup]; "
                f"got {tuple(support_uv.shape)[:2]} vs "
                f"{tuple(support_visibilities.shape)[:2]}"
            )
        if support_uv.shape[-1] != cfg.uv_feature_dim:
            raise ValueError(
                f"support_uv last dim must be {cfg.uv_feature_dim}, "
                f"got {support_uv.shape[-1]}"
            )
        if target_uv.dim() != 3 or target_uv.shape[-1] != cfg.uv_feature_dim:
            raise ValueError(
                f"target_uv expected [B, M_tgt, {cfg.uv_feature_dim}], "
                f"got shape {tuple(target_uv.shape)}"
            )
        if target_uv.shape[0] != support_uv.shape[0]:
            raise ValueError(
                "batch dimension of target_uv and support_uv must match; "
                f"got {target_uv.shape[0]} vs {support_uv.shape[0]}"
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_oracle_from_checkpoint(
    checkpoint_path,
    device=None,
):
    """Instantiate a :class:`HeavyHitterOracle` from a ``best.ckpt`` produced
    by :func:`dynadiff_vlbi.oracle.training.train_oracle`.

    The checkpoint is:
        {"model_state_dict": ..., "config": {...}, "history": ...}

    Architecture is reconstructed from the state-dict shapes plus the
    stored training config so the loader survives CLI / YAML drift.
    The returned module is on ``device`` and in ``eval`` mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    hidden_dim = int(state_dict["support_embed.0.weight"].shape[0])
    num_self = 1 + max(
        (int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("self_blocks.")),
        default=-1,
    )
    num_cross = 1 + max(
        (int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("cross_blocks.")),
        default=-1,
    )
    mlp_hidden = int(state_dict["self_blocks.0.mlp.0.weight"].shape[0])
    mlp_ratio = float(mlp_hidden) / float(hidden_dim)
    cfg_payload = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    num_heads = int(cfg_payload.get("num_heads", 4))

    cfg = HeavyHitterOracleConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_self_layers=max(1, num_self),
        num_cross_layers=max(1, num_cross),
        dropout=0.0,
        mlp_ratio=mlp_ratio,
        vis_feature_dim=3,
        uv_feature_dim=3,
    )
    oracle = HeavyHitterOracle(cfg)
    oracle.load_state_dict(state_dict)
    oracle.to(device)
    oracle.train(False)
    return oracle
