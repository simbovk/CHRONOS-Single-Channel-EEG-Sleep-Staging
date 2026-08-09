"""Dataset definitions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    """Expose samples shaped ``[W,T,C]`` with integer label and block ID.

    Arrays or paths to ``.npy`` arrays are accepted. Path-backed arrays are opened
    with memory mapping by default. Channel selection happens lazily, avoiding a
    copy of the complete dataset.
    """

    def __init__(
        self,
        data: str | Path | np.ndarray,
        labels: str | Path | np.ndarray,
        block_ids: str | Path | np.ndarray,
        channel_indices: Sequence[int] = (0,),
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
        mmap_mode: str | None = "r",
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        self.X = self._open(data, mmap_mode)
        self.y = self._open(labels, mmap_mode)
        self.block_ids = self._open(block_ids, mmap_mode)
        self.channel_indices = tuple(channel_indices)
        self.transform = transform
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        if self.X.ndim != 4:
            raise ValueError(f"data must have shape [N,W,T,C], got {self.X.shape}")
        if not (len(self.X) == len(self.y) == len(self.block_ids)):
            raise ValueError("data, labels, and block IDs must have equal length")
        if not self.channel_indices:
            raise ValueError("at least one channel must be selected")

    @staticmethod
    def _open(value: str | Path | np.ndarray, mmap_mode: str | None) -> np.ndarray:
        return np.load(Path(value), mmap_mode=mmap_mode) if isinstance(value, (str, Path)) else value

    def __len__(self) -> int:
        return len(self.y) if self.indices is None else len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = index if self.indices is None else int(self.indices[index])
        x = np.asarray(self.X[index][..., self.channel_indices])
        signal = self.transform(x) if self.transform else torch.from_numpy(x.copy()).float()
        label = torch.tensor(int(self.y[index]), dtype=torch.long)
        block_id = torch.tensor(int(self.block_ids[index]), dtype=torch.long)
        return signal, label, block_id
