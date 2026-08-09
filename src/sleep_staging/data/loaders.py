"""Storage-independent array and DataLoader construction."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from .dataset import SleepDataset


def load_arrays(data_dir: str | Path, dataset_config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memory-map configured data, label, and group arrays from a directory."""
    root = Path(data_dir)
    names = [dataset_config.get(k) for k in ("data_file", "labels_file", "block_ids_file")]
    if any(name is None for name in names):
        raise ValueError("dataset filenames are incomplete in the configuration")
    mode = dataset_config.get("mmap_mode", "r")
    return tuple(np.load(root / name, mmap_mode=mode) for name in names)  # type: ignore[return-value]


def make_loader(X, y, groups, *, channel_indices=(0,), batch_size=128,
                shuffle=False, num_workers=2, transform=None, indices=None) -> DataLoader:
    """Build the notebook-equivalent loader without storage assumptions."""
    dataset = SleepDataset(X, y, groups, channel_indices, transform, mmap_mode=None, indices=indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)
