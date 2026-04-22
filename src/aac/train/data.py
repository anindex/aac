"""Query pair dataset and train/val/test splits for compressor training.

Provides deterministic data generation with fixed seeds for reproducibility.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class QueryPairDataset(Dataset):
    """Random (source, target) pairs for training.

    Samples uniformly from V vertices with fixed seed for reproducibility.

    Args:
        num_vertices: Number of vertices V in the graph.
        num_pairs: Number of (source, target) pairs to generate.
        seed: Random seed for reproducibility.
    """

    def __init__(self, num_vertices: int, num_pairs: int, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        # Explicit int64 for CUDA index compatibility (numpy may produce int32 on some platforms)
        self.sources = torch.from_numpy(rng.randint(0, num_vertices, size=num_pairs).astype(np.int64))
        self.targets = torch.from_numpy(rng.randint(0, num_vertices, size=num_pairs).astype(np.int64))

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sources[idx], self.targets[idx]


def make_splits(
    num_vertices: int,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split vertex indices into train/val/test sets.

    Uses a fixed random seed for deterministic, reproducible splits.

    Args:
        num_vertices: Total number of vertices V.
        seed: Random seed for reproducibility.
        train_frac: Fraction of vertices for training.
        val_frac: Fraction of vertices for validation.

    Returns:
        (train_indices, val_indices, test_indices) as int64 tensors.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_vertices)
    n_train = int(num_vertices * train_frac)
    n_val = int(num_vertices * val_frac)
    train_idx = torch.from_numpy(perm[:n_train].copy()).long()
    val_idx = torch.from_numpy(perm[n_train : n_train + n_val].copy()).long()
    test_idx = torch.from_numpy(perm[n_train + n_val :].copy()).long()
    return train_idx, val_idx, test_idx
