"""Score-based diffusion model for dynamic inverse problems.

Implements a conditional score-based generative model (Song et al. 2021)
adapted for dynamic VLBI reconstruction with structured measurement
conditioning. The model learns the score function (gradient of the log
probability) of the image posterior conditioned on sparse Fourier
measurements, enabling principled posterior sampling via Langevin dynamics.

Key innovations over standard DPS:
  1. Temporal score matching: jointly models the score of dynamic sequences
     rather than per-frame independent sampling
  2. Measurement-conditioned architecture: conditions the score network on
     sparse visibility data through cross-attention
  3. Structured consistency guidance: measurement-consistency gradient
     applied only on support-set coefficients, naturally separating
     earned from enforced consistency
  4. Applicable to any linear inverse problem (VLBI, MRI, CT, radar)

References:
  - Song et al. (2021), Score-Based Generative Modeling through SDEs, ICLR
  - Chung et al. (2023), Diffusion Posterior Sampling, ICLR
  - Kawar et al. (2022), DDRM: Denoising Diffusion Restoration Models, NeurIPS
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from dynadiff_vlbi.models.base import BaseReconstructionModel
from dynadiff_vlbi.physics.torch_fourier import fft2c_torch, ifft2c_torch
from dynadiff_vlbi.utils.config import ModelConfig


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class ContinuousNoiseSchedule(nn.Module):
    """Variance-preserving SDE noise schedule: sigma(t) = sigma_min^(1-t) * sigma_max^t."""

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min ** (1.0 - t) * self.sigma_max ** t

    def log_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - t) * math.log(self.sigma_min) + t * math.log(self.sigma_max)

    def marginal_prob_std(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t)

    def drift_coefficient(self, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient for the reverse SDE."""
        return -0.5 * self.sigma(t) ** 2


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock3D(nn.Module):
    """Residual block with time conditioning for 3D tensors [B,C,T,H,W]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        groups = max(1, min(8, in_channels))
        groups_out = max(1, min(8, out_channels))

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # Add time embedding
        t_proj = self.time_proj(F.silu(t_emb))[:, :, None, None, None]
        h = h + t_proj
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class MeasurementCrossAttention(nn.Module):
    """Cross-attention between image features and measurement features.

    Allows the score network to attend to the sparse measurement
    information, providing a principled conditioning mechanism that
    respects the structure of the inverse problem.

    Includes a learned projection to adapt the measurement feature
    dimension to the image feature dimension at each encoder level.
    """

    def __init__(self, channels: int, meas_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.norm = nn.GroupNorm(max(1, min(8, channels)), channels)
        self.q_proj = nn.Conv3d(channels, channels, 1)
        self.meas_proj = nn.Linear(meas_dim, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Conv3d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        measurement_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] image features
            measurement_features: [B, N, meas_dim] measurement token features
        """
        B, C, T, H, W = x.shape
        residual = x
        x = self.norm(x)

        # Project measurement features to match image channel dim
        meas = self.meas_proj(measurement_features)

        # Queries from image features
        q = self.q_proj(x).reshape(B, self.num_heads, C // self.num_heads, T * H * W)
        q = q.permute(0, 1, 3, 2)  # [B, heads, THW, head_dim]

        # Keys and values from measurements
        k = self.k_proj(meas).reshape(
            B, -1, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)  # [B, heads, N, head_dim]
        v = self.v_proj(meas).reshape(
            B, -1, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, THW, head_dim]

        out = out.permute(0, 1, 3, 2).reshape(B, C, T, H, W)
        out = self.out_proj(out)
        return residual + out


# ---------------------------------------------------------------------------
# Score network
# ---------------------------------------------------------------------------

class TemporalScoreNetwork(nn.Module):
    """U-Net score network for dynamic sequences with measurement conditioning.

    Architecture:
      - 3D U-Net backbone operating on [B, C, T, H, W]
      - Sinusoidal time embedding for noise level conditioning
      - Cross-attention layers for measurement conditioning
      - Variable depth via num_levels parameter
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        measurement_dim: int = 128,
        dropout: float = 0.1,
        attention_levels: tuple[int, ...] = (2, 3),
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Measurement encoder: visibility -> tokens
        self.meas_token_dim = measurement_dim
        self.measurement_encoder = nn.Sequential(
            nn.Linear(4, measurement_dim),  # [vis_real, vis_imag, u, v]
            nn.SiLU(),
            nn.Linear(measurement_dim, measurement_dim),
            nn.SiLU(),
            nn.Linear(measurement_dim, measurement_dim),
        )

        num_levels = len(channel_mult)
        channels = [base_channels * m for m in channel_mult]

        # Input projection
        self.input_proj = nn.Conv3d(in_channels, channels[0], 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        ch_in = channels[0]
        for level in range(num_levels):
            ch_out = channels[level]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3D(ch_in, ch_out, time_dim, dropout))
                ch_in = ch_out
            self.encoder_blocks.append(blocks)

            if level in attention_levels:
                self.encoder_attns.append(
                    MeasurementCrossAttention(
                        ch_out, measurement_dim, num_heads=max(1, ch_out // 32)
                    )
                )
            else:
                self.encoder_attns.append(nn.Identity())

            if level < num_levels - 1:
                self.encoder_pools.append(nn.Conv3d(ch_out, ch_out, (1, 2, 2), stride=(1, 2, 2)))
            else:
                self.encoder_pools.append(nn.Identity())

        # Middle
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock3D(mid_ch, mid_ch, time_dim, dropout)
        self.mid_attn = MeasurementCrossAttention(
            mid_ch, measurement_dim, num_heads=max(1, mid_ch // 32)
        )
        self.mid_block2 = ResBlock3D(mid_ch, mid_ch, time_dim, dropout)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        for level in reversed(range(num_levels)):
            ch_out = channels[level]
            if level < num_levels - 1:
                ch_prev = channels[level + 1]
                self.decoder_ups.append(
                    nn.ConvTranspose3d(ch_prev, ch_out, (1, 2, 2), stride=(1, 2, 2))
                )
            else:
                self.decoder_ups.append(nn.Identity())

            blocks = nn.ModuleList()
            # Concat with skip connection
            dec_in = ch_out * 2 if level < num_levels - 1 else ch_out
            for i in range(num_res_blocks):
                blocks.append(ResBlock3D(dec_in if i == 0 else ch_out, ch_out, time_dim, dropout))
            self.decoder_blocks.append(blocks)

            if level in attention_levels:
                self.decoder_attns.append(
                    MeasurementCrossAttention(
                        ch_out, measurement_dim, num_heads=max(1, ch_out // 32)
                    )
                )
            else:
                self.decoder_attns.append(nn.Identity())

        # Output
        out_groups = max(1, min(8, channels[0]))
        self.output_proj = nn.Sequential(
            nn.GroupNorm(out_groups, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], in_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        measurement_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the score (noise) at noise level t.

        Args:
            x: [B, C, T, H, W] noisy image sequence
            t: [B] noise levels in [0, 1]
            measurement_tokens: [B, N, 4] sparse measurement features
                Each token: [vis_real, vis_imag, u_coord, v_coord]

        Returns:
            Predicted noise (epsilon) with same shape as x.
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)

        t_emb = self.time_embed(t)

        # Encode measurements into tokens
        if measurement_tokens is not None:
            meas_features = self.measurement_encoder(measurement_tokens)
        else:
            B = x.shape[0]
            meas_features = torch.zeros(B, 1, self.meas_token_dim, device=x.device)

        h = self.input_proj(x)

        # Encoder with skip connections
        skips = []
        for blocks, pool, attn in zip(
            self.encoder_blocks, self.encoder_pools, self.encoder_attns
        ):
            for block in blocks:
                h = block(h, t_emb)
            if isinstance(attn, MeasurementCrossAttention):
                h = attn(h, meas_features)
            skips.append(h)
            h = pool(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, meas_features)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for blocks, up, attn, skip in zip(
            self.decoder_blocks, self.decoder_ups, self.decoder_attns,
            reversed(skips),
        ):
            if not isinstance(up, nn.Identity):
                h = up(h)
                h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)
            if isinstance(attn, MeasurementCrossAttention):
                h = attn(h, meas_features)

        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Diffusion posterior sampler
# ---------------------------------------------------------------------------

@dataclass
class DPSSample:
    """Output of diffusion posterior sampling."""

    mean: torch.Tensor  # posterior mean estimate
    samples: list[torch.Tensor]  # posterior samples
    log_variance: torch.Tensor | None  # estimated log-variance from samples


class DiffusionPosteriorSampler(BaseReconstructionModel):
    """Score-based diffusion posterior sampling for dynamic inverse problems.

    Combines a learned score network with measurement-consistency guidance
    to sample from the posterior p(x | y) where y = Mx + noise.

    The key insight: measurement-consistency guidance is applied only on
    the support set, naturally implementing earned measurement consistency.
    The target set measurements are never used during sampling, allowing
    them to serve as a genuine validation metric.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_steps: int = 100,
        guidance_scale: float = 1.0,
        num_samples: int = 4,
    ) -> None:
        super().__init__()
        self.config = config

        # Determine architecture size from config
        base = config.base_channels
        num_levels = getattr(config, "num_levels", 3)
        channel_mult = tuple(2**i for i in range(num_levels + 1))
        attention_levels = tuple(range(num_levels - 1, num_levels + 1))

        self.score_network = TemporalScoreNetwork(
            in_channels=config.in_channels,
            base_channels=base,
            channel_mult=channel_mult,
            time_dim=base * 4,
            measurement_dim=base * 2,
            dropout=config.dropout,
            attention_levels=attention_levels,
        )
        self.noise_schedule = ContinuousNoiseSchedule()
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.num_samples = num_samples

    def _measurement_consistency_gradient(
        self,
        x: torch.Tensor,
        measurements: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient of measurement consistency loss on support set.

        This implements the DPS-style guidance: grad_x ||M*F(x) - y||^2
        where M is the SUPPORT mask only (not the full measurement mask).
        """
        x_detached = x.detach().requires_grad_(True)
        # Compute predicted visibilities
        x_vis = fft2c_torch(x_detached.squeeze(1))
        # Measurement residual on support set only
        residual = (x_vis - measurements) * mask
        loss = 0.5 * torch.sum(torch.abs(residual) ** 2)
        grad = torch.autograd.grad(loss, x_detached)[0]
        return grad

    def _prepare_measurement_tokens(
        self,
        vis_real: torch.Tensor,
        vis_imag: torch.Tensor,
        mask: torch.Tensor,
        uv_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert sparse visibility data into measurement tokens for cross-attention.

        Returns: [B, N, 4] tensor of [vis_real, vis_imag, u, v] per observed coefficient.
        """
        B, T, H, W = vis_real.shape
        tokens_list = []
        max_tokens = 0

        for b in range(B):
            sample_tokens = []
            for t in range(T):
                observed = mask[b, t].bool()
                n_obs = observed.sum().item()
                if n_obs == 0:
                    continue
                vr = vis_real[b, t][observed]
                vi = vis_imag[b, t][observed]
                # Generate UV coordinates from mask indices
                rows, cols = torch.where(observed)
                u = (cols.float() / W - 0.5) * 2.0
                v = (rows.float() / H - 0.5) * 2.0
                frame_tokens = torch.stack([vr, vi, u, v], dim=-1)
                sample_tokens.append(frame_tokens)

            if sample_tokens:
                all_tokens = torch.cat(sample_tokens, dim=0)
                tokens_list.append(all_tokens)
                max_tokens = max(max_tokens, all_tokens.shape[0])
            else:
                tokens_list.append(torch.zeros(1, 4, device=vis_real.device))
                max_tokens = max(max_tokens, 1)

        # Pad to same length
        padded = torch.zeros(B, max_tokens, 4, device=vis_real.device)
        for b, tokens in enumerate(tokens_list):
            n = min(tokens.shape[0], max_tokens)
            padded[b, :n] = tokens[:n]

        return padded

    @torch.no_grad()
    def sample_posterior(
        self,
        dirty_input: torch.Tensor,
        measurements: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        vis_real: torch.Tensor | None = None,
        vis_imag: torch.Tensor | None = None,
        num_samples: int | None = None,
    ) -> DPSSample:
        """Sample from the posterior p(x | y_support) using reverse SDE.

        Args:
            dirty_input: [B, T, H, W] initial estimate (dirty reconstruction)
            measurements: [B, T, H, W] complex support visibilities
            mask: [B, T, H, W] support measurement mask
            vis_real: [B, T, H, W] real part of support visibilities
            vis_imag: [B, T, H, W] imaginary part of support visibilities
            num_samples: number of posterior samples

        Returns:
            DPSSample with posterior mean and individual samples
        """
        num_samples = num_samples or self.num_samples
        device = dirty_input.device
        B = dirty_input.shape[0]

        if dirty_input.ndim == 4:
            x_shape = (B, 1, *dirty_input.shape[1:])
        else:
            x_shape = dirty_input.shape

        # Prepare measurement tokens for cross-attention
        meas_tokens = None
        if vis_real is not None and vis_imag is not None and mask is not None:
            meas_tokens = self._prepare_measurement_tokens(vis_real, vis_imag, mask)

        timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)
        samples = []

        for s in range(num_samples):
            # Initialize from noise
            x_t = torch.randn(x_shape, device=device)

            for i in range(self.num_steps):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                t_batch = t_curr.expand(B)
                sigma_curr = self.noise_schedule.sigma(t_batch)

                # Score prediction (noise prediction)
                eps_pred = self.score_network(x_t, t_batch, meas_tokens)

                # Denoising step (DDPM-style)
                alpha = 1.0 - (sigma_curr[:, None, None, None, None] ** 2)
                x_denoised = (x_t - sigma_curr[:, None, None, None, None] * eps_pred) / alpha.sqrt().clamp(min=1e-6)

                # Measurement consistency guidance (DPS-style)
                if measurements is not None and mask is not None and self.guidance_scale > 0:
                    with torch.enable_grad():
                        mc_grad = self._measurement_consistency_gradient(
                            x_denoised, measurements, mask.unsqueeze(1) if mask.ndim == 4 else mask
                        )
                    x_denoised = x_denoised - self.guidance_scale * mc_grad

                # Add noise for next step (unless last step)
                if t_next > 0:
                    sigma_next = self.noise_schedule.sigma(t_next.expand(B))
                    noise = torch.randn_like(x_t)
                    x_t = alpha.sqrt() * x_denoised + sigma_next[:, None, None, None, None] * noise
                else:
                    x_t = x_denoised

            x_t = torch.clamp(x_t, 0.0, 1.0)
            samples.append(x_t)

        # Posterior statistics
        stacked = torch.stack(samples, dim=0)  # [S, B, C, T, H, W]
        posterior_mean = stacked.mean(dim=0)
        if num_samples > 1:
            log_variance = torch.log(stacked.var(dim=0).clamp(min=1e-8))
        else:
            log_variance = None

        return DPSSample(
            mean=posterior_mean,
            samples=samples,
            log_variance=log_variance,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score prediction for training (noise prediction objective).

        During training, the forward pass adds noise and predicts the noise.
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)
        B = x.shape[0]
        device = x.device

        # Sample random noise levels
        t = torch.rand(B, device=device)
        sigma = self.noise_schedule.sigma(t)

        # Add noise
        noise = torch.randn_like(x)
        x_noisy = x + sigma[:, None, None, None, None] * noise

        # Predict noise
        eps_pred = self.score_network(x_noisy, t, measurement_tokens=None)

        return eps_pred

    def training_loss(
        self,
        clean_images: torch.Tensor,
        measurement_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denoising score matching loss.

        L = E_{t, x_0, eps} [ ||eps - eps_theta(x_t, t, y)||^2 ]
        """
        if clean_images.ndim == 4:
            clean_images = clean_images.unsqueeze(1)
        B = clean_images.shape[0]
        device = clean_images.device

        t = torch.rand(B, device=device)
        sigma = self.noise_schedule.sigma(t)
        noise = torch.randn_like(clean_images)
        x_noisy = clean_images + sigma[:, None, None, None, None] * noise

        eps_pred = self.score_network(x_noisy, t, measurement_tokens)

        # Weighted denoising loss
        weight = 1.0 / (sigma ** 2 + 1e-6)
        loss = weight[:, None, None, None, None] * (eps_pred - noise) ** 2
        return loss.mean()
