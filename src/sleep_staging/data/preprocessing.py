"""Input validation; the authoritative notebook applies no signal preprocessing."""

import numpy as np


def validate_arrays(data: np.ndarray, labels: np.ndarray, block_ids: np.ndarray,
                    num_classes: int = 5) -> None:
    """Validate notebook array ranks, lengths, and label range."""
    if data.ndim != 4:
        raise ValueError(f"expected [N,W,T,C], got {data.shape}")
    if not (len(data) == len(labels) == len(block_ids)):
        raise ValueError("array lengths differ")
    if len(labels) and (np.min(labels) < 0 or np.max(labels) >= num_classes):
        raise ValueError("labels are outside the configured class range")
