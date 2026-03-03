"""Dataset wrappers for visibility-conditioned training and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dynadiff_vlbi.data.dataset import DynamicVLBIDataset
from dynadiff_vlbi.data.feature_formatting import (
    build_temporal_uv_grid,
    format_dirty_input,
    format_visibility_tensor,
)
from dynadiff_vlbi.utils.config import ModelConfig


class VisibilityConditionedDataset(DynamicVLBIDataset):
    """Dataset that exposes both dirty-image and visibility-domain inputs."""

    def __init__(
        self,
        npz_path: str | Path,
        model_config: ModelConfig,
    ) -> None:
        super().__init__(npz_path=npz_path)
        self.model_config = model_config
        if self.uv_coords is None:
            self.uv_coords = build_temporal_uv_grid(
                image_size=self.ground_truth.shape[-1],
                sequence_length=self.ground_truth.shape[1],
            ).astype("float32")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        visibility_input = format_visibility_tensor(
            vis_real=self.vis_real[index],
            vis_imag=self.vis_imag[index],
            mask=self.mask[index],
            representation=self.model_config.visibility_representation,
            include_mask_channel=self.model_config.include_mask_channel,
            include_uv_coords=self.model_config.include_uv_coords,
            uv_coords=self.uv_coords,
        )
        dirty_input = format_dirty_input(self.dirty[index])
        sample.update(
            {
                "visibility_input": torch.from_numpy(visibility_input),
                "dirty_input": torch.from_numpy(dirty_input),
                "uv_coords": torch.from_numpy(self.uv_coords),
            }
        )
        return sample


def build_visibility_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    model_config: ModelConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build dataloaders for visibility-conditioned experiments."""

    data_dir = Path(data_dir)
    train_dataset = VisibilityConditionedDataset(data_dir / "train.npz", model_config=model_config)
    val_dataset = VisibilityConditionedDataset(data_dir / "val.npz", model_config=model_config)
    test_dataset = VisibilityConditionedDataset(data_dir / "test.npz", model_config=model_config)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
