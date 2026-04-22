"""Gap-closing loss for compressor training.

Minimizes E[d(s,t) - h_{A,T}(s,t)] + condition_regularization.
Admissibility is guaranteed by the row-stochastic structure of the compression
matrix: convex combinations cannot exceed the maximum, so
h_compressed <= h_teacher. The gap d - h is always >= 0 for admissible heuristics.
"""

from __future__ import annotations

import torch

from aac.compression.compressor import PositiveCompressor


def gap_closing_loss(
    d_true: torch.Tensor,
    h_smooth: torch.Tensor,
    compressor: PositiveCompressor,
    cond_lambda: float = 0.01,
) -> torch.Tensor:
    """Gap-closing loss: minimize E[d(s,t) - h_{A,T}(s,t)] + cond_reg.

    The gap (d_true - h_smooth) should be >= 0 by admissibility.
    Training minimizes this gap to make the heuristic as informative as possible.

    Args:
        d_true: (B,) true shortest-path distances.
        h_smooth: (B,) smoothed heuristic values (differentiable w.r.t. compressor).
        compressor: The PositiveCompressor being trained.
        cond_lambda: Weight for condition number regularization.

    Returns:
        Scalar loss tensor.
    """
    gap = d_true - h_smooth  # should be >= 0 by admissibility
    loss = gap.mean() + cond_lambda * compressor.condition_regularization()
    return loss
