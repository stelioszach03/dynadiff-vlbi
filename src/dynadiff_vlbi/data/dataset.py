"""Torch dataset wrappers for generated synthetic VLBI data."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from dynadiff_vlbi.data.io import load_npz


class DynamicVLBIDataset(Dataset):
    """Dataset wrapper for compressed split files."""

    def __init__(self, npz_path: str | Path) -> None:
        data = load_npz(npz_path)
        self.ground_truth = data["ground_truth"].astype("float32")
        self.dirty = data["dirty"].astype("float32")
        self.vis_real = data["vis_real"].astype("float32")
        self.vis_imag = data["vis_imag"].astype("float32")
        self.mask = data["mask"].astype("float32")
        self.ring_radius_px = data["ring_radius_px"].astype("float32")
        self.hotspot_coords_px = data["hotspot_coords_px"].astype("float32")
        self.uv_coords = data["uv_coords"].astype("float32") if "uv_coords" in data else None
        self.station_positions = (
            data["station_positions"].astype("float32") if "station_positions" in data else None
        )
        self.baseline_pairs = data["baseline_pairs"].astype("int64") if "baseline_pairs" in data else None
        self.frame_uv_indices = data["frame_uv_indices"].astype("int64") if "frame_uv_indices" in data else None
        self.frame_uv_coords = data["frame_uv_coords"].astype("float32") if "frame_uv_coords" in data else None

    def __len__(self) -> int:
        return int(self.ground_truth.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {
            "sample_index": torch.tensor(index, dtype=torch.int64),
            "input": torch.from_numpy(self.dirty[index]),
            "dirty": torch.from_numpy(self.dirty[index]),
            "target": torch.from_numpy(self.ground_truth[index]),
            "vis_real": torch.from_numpy(self.vis_real[index]),
            "vis_imag": torch.from_numpy(self.vis_imag[index]),
            "mask": torch.from_numpy(self.mask[index]),
            "ring_radius_px": torch.tensor(self.ring_radius_px[index], dtype=torch.float32),
            "hotspot_coords_px": torch.from_numpy(self.hotspot_coords_px[index]),
        }
        if self.station_positions is not None:
            sample["station_positions"] = torch.from_numpy(self.station_positions)
        if self.baseline_pairs is not None:
            sample["baseline_pairs"] = torch.from_numpy(self.baseline_pairs)
        if self.frame_uv_indices is not None:
            sample["frame_uv_indices"] = torch.from_numpy(self.frame_uv_indices)
        if self.frame_uv_coords is not None:
            sample["frame_uv_coords"] = torch.from_numpy(self.frame_uv_coords)
        return sample


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/validation/test dataloaders for a generated dataset directory."""

    data_dir = Path(data_dir)
    train_dataset = DynamicVLBIDataset(data_dir / "train.npz")
    val_dataset = DynamicVLBIDataset(data_dir / "val.npz")
    test_dataset = DynamicVLBIDataset(data_dir / "test.npz")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
