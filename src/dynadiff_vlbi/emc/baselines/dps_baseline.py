"""Diffusion Posterior Sampling baseline for the EMC benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dynadiff_vlbi.physics.fourier_operator import fft2c, ifft2c
from dynadiff_vlbi.physics.torch_fourier import fft2c_torch


@dataclass(frozen=True)
class DPSCheckpointConfig:
    """Serializable DPS hyper-parameters."""

    image_size: int = 64
    base_channels: int = 32
    timesteps: int = 1000
    ddim_steps: int = 50
    lambda_start: float = 0.3
    lambda_end: float = 0.05
    beta_start: float = 1.0e-4
    beta_end: float = 0.02


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for diffusion timesteps."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale
        )
        angles = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualTimeBlock(nn.Module):
    """Residual convolution block with timestep conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_proj(F.silu(time_embedding)).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class DPSScoreUNet(nn.Module):
    """Small unconditional score network for 64x64 grayscale frames."""

    def __init__(self, image_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.time_embedding = SinusoidalTimeEmbedding(base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualTimeBlock(base_channels, base_channels, time_dim)
        self.pool1 = nn.AvgPool2d(2)
        self.down2 = ResidualTimeBlock(base_channels, base_channels * 2, time_dim)
        self.pool2 = nn.AvgPool2d(2)
        self.mid = ResidualTimeBlock(base_channels * 2, base_channels * 4, time_dim)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up_block1 = ResidualTimeBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up_block2 = ResidualTimeBlock(base_channels * 2, base_channels, time_dim)
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_proj = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(self.time_embedding(timesteps))
        x0 = self.input_proj(x)
        x1 = self.down1(x0, time_embedding)
        x2 = self.down2(self.pool1(x1), time_embedding)
        mid = self.mid(self.pool2(x2), time_embedding)
        up1 = self.up1(mid)
        up1 = self.up_block1(torch.cat([up1, x2], dim=1), time_embedding)
        up2 = self.up2(up1)
        up2 = self.up_block2(torch.cat([up2, x1], dim=1), time_embedding)
        return self.output_proj(F.silu(self.output_norm(up2)))


class DPSBaseline:
    """Unconditional DDPM prior with support-only posterior sampling guidance."""

    def __init__(
        self,
        score_model: DPSScoreUNet,
        *,
        total_timesteps: int = 1000,
        ddim_steps: int = 50,
        lambda_start: float = 0.3,
        lambda_end: float = 0.05,
        beta_start: float = 1.0e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.score_model = score_model
        self.score_model.eval()
        self.total_timesteps = int(total_timesteps)
        self.ddim_steps = int(ddim_steps)
        self.lambda_start = float(lambda_start)
        self.lambda_end = float(lambda_end)
        betas = torch.linspace(beta_start, beta_end, self.total_timesteps, dtype=torch.float32)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @property
    def device(self) -> torch.device:
        return next(self.score_model.parameters()).device

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.score_model.state_dict()

    @staticmethod
    def support_measurement_loss(
        prediction: torch.Tensor,
        support_vis: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        predicted_vis = fft2c_torch(prediction.squeeze(1))
        residual = (predicted_vis - support_vis) * support_mask
        denom = support_mask.sum().clamp_min(1.0)
        return (residual.real.pow(2).sum() + residual.imag.pow(2).sum()) / denom

    def training_loss(self, clean_frames: torch.Tensor) -> torch.Tensor:
        batch_size = int(clean_frames.shape[0])
        timesteps = torch.randint(
            low=0,
            high=self.total_timesteps,
            size=(batch_size,),
            device=clean_frames.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(clean_frames)
        alpha_bar = self.alpha_bars.to(clean_frames.device)[timesteps].view(-1, 1, 1, 1)
        noisy = alpha_bar.sqrt() * clean_frames + (1.0 - alpha_bar).sqrt() * noise
        predicted_noise = self.score_model(noisy, timesteps)
        return F.mse_loss(predicted_noise, noise)

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return values.to(reference.device)[timesteps].view(-1, 1, 1, 1)

    def _predict_x0(self, x_t: torch.Tensor, timesteps: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self._extract(self.alpha_bars, timesteps, x_t)
        return (x_t - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt().clamp_min(1.0e-8)

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        timesteps: torch.Tensor,
        next_timestep: int,
    ) -> torch.Tensor:
        alpha_bar_t = self._extract(self.alpha_bars, timesteps, x_t)
        if next_timestep < 0:
            return x0_pred
        next_timesteps = torch.full_like(timesteps, fill_value=next_timestep)
        alpha_bar_next = self._extract(self.alpha_bars, next_timesteps, x_t)
        predicted_noise = (x_t - alpha_bar_t.sqrt() * x0_pred) / (1.0 - alpha_bar_t).sqrt().clamp_min(1.0e-8)
        return alpha_bar_next.sqrt() * x0_pred + (1.0 - alpha_bar_next).sqrt() * predicted_noise

    def _guidance_lambda(self, step_index: int) -> float:
        if self.ddim_steps <= 1:
            return self.lambda_end
        fraction = float(step_index) / float(self.ddim_steps - 1)
        if fraction <= 0.6:
            return self.lambda_start
        tail_fraction = (fraction - 0.6) / 0.4
        return self.lambda_start + (self.lambda_end - self.lambda_start) * tail_fraction

    def sample(
        self,
        support_vis: np.ndarray,
        support_mask: np.ndarray,
        dirty_recon: np.ndarray,
    ) -> np.ndarray:
        """Run DDIM posterior sampling with support-only guidance."""

        self.score_model.eval()
        support_vis_tensor = torch.from_numpy(np.asarray(support_vis, dtype=np.complex64)).to(self.device)
        support_mask_tensor = torch.from_numpy(np.asarray(support_mask, dtype=np.float32)).to(self.device)
        dirty_tensor = torch.from_numpy(np.asarray(dirty_recon, dtype=np.float32)).to(self.device).unsqueeze(1)

        if dirty_tensor.ndim != 4:
            raise ValueError("DPS expects a dynamic sequence with shape [T, H, W].")

        timesteps = torch.linspace(
            self.total_timesteps - 1,
            0,
            self.ddim_steps,
            dtype=torch.long,
            device=self.device,
        )
        latent = torch.randn_like(dirty_tensor)
        latent = 0.92 * latent + 0.08 * dirty_tensor

        for step_index, timestep in enumerate(timesteps.tolist()):
            timestep_tensor = torch.full(
                (latent.shape[0],),
                fill_value=int(timestep),
                device=self.device,
                dtype=torch.long,
            )
            guided_latent = latent.detach().requires_grad_(True)
            predicted_noise = self.score_model(guided_latent, timestep_tensor)
            x0_pred = self._predict_x0(guided_latent, timestep_tensor, predicted_noise).clamp(0.0, 1.0)
            guidance_loss = self.support_measurement_loss(
                prediction=x0_pred,
                support_vis=support_vis_tensor,
                support_mask=support_mask_tensor,
            )
            guidance_grad = torch.autograd.grad(guidance_loss, guided_latent)[0]
            latent = (guided_latent - self._guidance_lambda(step_index) * guidance_grad).detach()

            with torch.no_grad():
                predicted_noise = self.score_model(latent, timestep_tensor)
                x0_pred = self._predict_x0(latent, timestep_tensor, predicted_noise).clamp(0.0, 1.0)
                next_timestep = int(timesteps[step_index + 1].item()) if step_index + 1 < len(timesteps) else -1
                latent = self._ddim_step(latent, x0_pred, timestep_tensor, next_timestep)

        prediction = x0_pred.squeeze(1).detach().cpu().numpy().astype(np.float32)
        support_vis_array = np.asarray(support_vis, dtype=np.complex64)
        support_mask_array = np.asarray(support_mask, dtype=np.float32)
        projected_vis = fft2c(prediction) * (1.0 - support_mask_array) + support_vis_array * support_mask_array
        return ifft2c(projected_vis).real.astype(np.float32)


def save_dps_checkpoint(
    *,
    path: str | Path,
    score_model: DPSScoreUNet,
    config: DPSCheckpointConfig,
    optimizer_state_dict: dict[str, Any] | None = None,
    epoch: int | None = None,
    best_val_loss: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "dps_config": asdict(config),
        "model_state_dict": score_model.state_dict(),
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if best_val_loss is not None:
        payload["best_val_loss"] = float(best_val_loss)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_dps_baseline(path: str | Path, device: torch.device) -> DPSBaseline:
    checkpoint = torch.load(Path(path), map_location=device)
    config = DPSCheckpointConfig(**checkpoint["dps_config"])
    model = DPSScoreUNet(image_channels=1, base_channels=config.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return DPSBaseline(
        model,
        total_timesteps=config.timesteps,
        ddim_steps=config.ddim_steps,
        lambda_start=config.lambda_start,
        lambda_end=config.lambda_end,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
