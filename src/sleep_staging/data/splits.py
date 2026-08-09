"""Leakage-safe fold generation matching the notebook."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def generate_folds(labels: np.ndarray, block_ids: np.ndarray, n_splits: int = 5,
                   shuffle: bool = False, seed: int = 42):
    """Yield train/validation indices with disjoint block IDs.

    The notebook uses ``shuffle=False``. Scikit-learn rejects a random state in
    that mode, so the seed is supplied only when shuffling is explicitly enabled.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle,
                                    random_state=seed if shuffle else None)
    dummy = np.zeros(len(labels), dtype=np.uint8)
    for train, validation in splitter.split(dummy, labels, groups=block_ids):
        if np.intersect1d(block_ids[train], block_ids[validation]).size:
            raise RuntimeError("block leakage detected")
        yield train, validation
