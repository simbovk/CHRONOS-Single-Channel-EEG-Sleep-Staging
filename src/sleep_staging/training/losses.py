"""Notebook-equivalent class weighting and loss construction."""
import numpy as np
import torch
from torch import nn


def compute_class_weights(labels, num_classes=5, device="cpu"):
    """Inverse-frequency weights normalized exactly as in the final CV cell."""
    counts = np.bincount(np.asarray(labels).astype(int), minlength=num_classes).astype(np.float32)
    inverse = 1.0 / np.clip(counts, 1.0, None)
    return torch.tensor(inverse * (num_classes / inverse.sum()), dtype=torch.float32, device=device)


def make_loss(class_weights=None):
    """Return weighted cross entropy (without label smoothing)."""
    return nn.CrossEntropyLoss(weight=class_weights)
