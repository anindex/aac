"""Heuristic evaluation from embedding label vectors.

Directed graphs (tropical/Funk):
    h(u,t) = max_i(phi_i(u) - phi_i(t))

Undirected graphs (Hilbert variation norm):
    h(u,t) = max_i(delta_i) - min_i(delta_i)
    where delta = phi(u) - phi(t)

Both produce admissible (lower-bound) heuristics for A* search.

Sentinel masking: When an anchor is unreachable, the embedding coordinate
contains +/-SENTINEL (1e18). These must be masked out before computing
max/min to avoid massive overestimates that destroy A* admissibility.
"""

from __future__ import annotations

import torch

from aac.graphs.types import Embedding
from aac.utils.numerics import SENTINEL


def _is_sentinel_vec(x: torch.Tensor) -> torch.Tensor:
    """Check for sentinel values (abs(x) > 0.99 * SENTINEL)."""
    return torch.abs(x) > 0.99 * SENTINEL


def evaluate_heuristic(
    phi_source: torch.Tensor,
    phi_target: torch.Tensor,
    is_directed: bool,
) -> torch.Tensor:
    """Evaluate heuristic for a single (source, target) pair.

    Masks out coordinates where either source or target embedding contains
    a sentinel value to avoid corrupted heuristic estimates.

    Args:
        phi_source: (2K,) label vector for source vertex.
        phi_target: (2K,) label vector for target vertex.
        is_directed: If True, use Funk distance (max). If False, Hilbert variation norm.

    Returns:
        Scalar tensor with heuristic value h(source, target).
    """
    delta = phi_source - phi_target
    valid = ~(_is_sentinel_vec(phi_source) | _is_sentinel_vec(phi_target))

    if not valid.any():
        return torch.tensor(0.0, dtype=delta.dtype, device=delta.device)

    delta_valid = delta[valid]

    if is_directed:
        return torch.max(delta_valid)
    else:
        return torch.max(delta_valid) - torch.min(delta_valid)


def evaluate_heuristic_batch(
    phi_sources: torch.Tensor,
    phi_targets: torch.Tensor,
    is_directed: bool,
) -> torch.Tensor:
    """Evaluate heuristic for a batch of (source, target) pairs.

    Masks out coordinates where either source or target embedding contains
    a sentinel value to avoid corrupted heuristic estimates.

    Args:
        phi_sources: (B, 2K) label vectors for source vertices.
        phi_targets: (B, 2K) label vectors for target vertices.
        is_directed: If True, use Funk distance. If False, Hilbert variation norm.

    Returns:
        (B,) tensor of heuristic values.
    """
    delta = phi_sources - phi_targets  # (B, 2K)
    valid = ~(_is_sentinel_vec(phi_sources) | _is_sentinel_vec(phi_targets))  # (B, 2K)

    # Replace invalid entries with 0 (neutral for max-min)
    # For directed max: use -inf so invalid entries don't affect max
    # For undirected: use 0 so they don't affect max or min
    if is_directed:
        masked_delta = torch.where(valid, delta, torch.tensor(float('-inf'), dtype=delta.dtype, device=delta.device))
        result = torch.max(masked_delta, dim=1).values
        # If all entries are invalid for a sample, return 0 (autograd-safe)
        all_invalid = ~valid.any(dim=1)
        result = torch.where(all_invalid, torch.zeros_like(result), result)
        return result
    else:
        masked_delta_max = torch.where(valid, delta, torch.tensor(float('-inf'), dtype=delta.dtype, device=delta.device))
        masked_delta_min = torch.where(valid, delta, torch.tensor(float('inf'), dtype=delta.dtype, device=delta.device))
        result = torch.max(masked_delta_max, dim=1).values - torch.min(masked_delta_min, dim=1).values
        # If all entries are invalid for a sample, return 0 (autograd-safe)
        all_invalid = ~valid.any(dim=1)
        result = torch.where(all_invalid, torch.zeros_like(result), result)
        return result


def evaluate_from_embedding(
    embedding: Embedding,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    """Convenience: evaluate heuristic from embedding + vertex indices.

    Looks up phi[source_indices] and phi[target_indices], then calls
    evaluate_heuristic_batch.

    Args:
        embedding: Embedding with phi (V, 2K) and is_directed flag.
        source_indices: (B,) int64 source vertex indices.
        target_indices: (B,) int64 target vertex indices.

    Returns:
        (B,) tensor of heuristic values.
    """
    phi_sources = embedding.phi[source_indices]
    phi_targets = embedding.phi[target_indices]
    return evaluate_heuristic_batch(phi_sources, phi_targets, embedding.is_directed)
