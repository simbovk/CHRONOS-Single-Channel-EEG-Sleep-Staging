"""Notebook-equivalent EEG augmentation."""

from __future__ import annotations

import random

import numpy as np
import torch


class EEGAugment:
    """Apply the exact stochastic transforms used by the CV notebook to ``[W,T,C]``."""

    def __init__(self, p_scale=0.5, scale_range=(0.8, 1.2), p_noise=0.5,
                 noise_alpha=(0.01, 0.05), p_shift=0.3, max_shift_frac=0.1,
                 p_mask=0.3, n_masks=(1, 3), mask_frac=(0.01, 0.05),
                 p_chmix=0.15, chmix_alpha=(0.05, 0.15), p_eog_flip=0.05,
                 eog_index=1):
        self.p_scale, self.scale_range = p_scale, scale_range
        self.p_noise, self.noise_alpha = p_noise, noise_alpha
        self.p_shift, self.max_shift_frac = p_shift, max_shift_frac
        self.p_mask, self.n_masks, self.mask_frac = p_mask, n_masks, mask_frac
        self.p_chmix, self.chmix_alpha = p_chmix, chmix_alpha
        self.p_eog_flip, self.eog_index = p_eog_flip, eog_index

    @classmethod
    def from_config(cls, config: dict) -> "EEGAugment":
        values = {k: v for k, v in config.items() if k != "enabled"}
        return cls(**values)

    def __call__(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(x).copy()).float()
        _, time, channels = x.shape
        if random.random() < self.p_scale:
            x = x * torch.empty(channels).uniform_(*self.scale_range).view(1, 1, channels)
        if random.random() < self.p_noise:
            rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)
            x = x + torch.randn_like(x) * (random.uniform(*self.noise_alpha) * rms)
        if random.random() < self.p_shift:
            limit = int(self.max_shift_frac * time)
            shift = 0 if limit == 0 else random.randint(-limit, limit)
            if shift:
                x = torch.roll(x, shifts=shift, dims=1)
        if random.random() < self.p_mask:
            for _ in range(random.randint(*self.n_masks)):
                length = random.randint(int(self.mask_frac[0] * time), max(1, int(self.mask_frac[1] * time)))
                start = random.randint(0, max(0, time - length))
                x[:, start:start + length, :] = 0.0
        if channels >= 2 and random.random() < self.p_chmix:
            alpha = random.uniform(*self.chmix_alpha)
            x[..., 0] = (1 - alpha) * x[..., 0] + alpha * x[..., 1]
        if self.eog_index is not None and self.eog_index < channels and random.random() < self.p_eog_flip:
            x[..., self.eog_index] = -x[..., self.eog_index]
        return x
