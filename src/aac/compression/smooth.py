"""Differentiable smoothed heuristic for AAC.

Provides temperature-parameterized smooth approximations to the max-based
heuristics used in A* search. The smoothed heuristic:

    M_T(x) = (1/T) * logsumexp(T * x) - log(m) / T

satisfies M_T(x) <= max(x) for all x, with equality as T -> infinity.
This gives a differentiable lower bound suitable for gradient-based training
while preserving the admissibility guarantee.

For directed graphs:   M_T(delta)      approximates max(delta)
For undirected graphs: M_T(|delta|)    approximates max|delta|  (L-inf norm)

The L-inf norm is admissible because |d(k,u) - d(k,t)| <= d(u,t) by
triangle inequality, and row-stochastic compression preserves this bound.

Sentinel masking: When an anchor is unreachable, embedding coordinates contain
sentinel values (+/-1e18). These are masked out before logsumexp, so they
contribute exp(-1e30) ~ 0 to the sum.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from aac.utils.numerics import SENTINEL


def _is_sentinel_vec(x: torch.Tensor) -> torch.Tensor:
    """Check for sentinel values (abs(x) > 0.99 * SENTINEL)."""
    return torch.abs(x) > 0.99 * SENTINEL


def smoothed_heuristic_directed(
    y_source: torch.Tensor,
    y_target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Smoothed heuristic for directed graphs (paper Appendix A).

    M_T(y(u) - y(t)) <= max(y(u) - y(t)) = h_A(u,t)

    Formula: M_T(x) = (1/T) * logsumexp(T * x) - log(m) / T

    The subtraction of log(m)/T ensures the lower bound property,
    since logsumexp(T*x) >= T*max(x) and logsumexp(T*x) <= T*max(x) + log(m).

    Sentinel coordinates are masked by setting them to -1e30 before logsumexp,
    so they contribute exp(-1e30) ~ 0 to the sum.

    Args:
        y_source: (..., m) compressed labels for source vertices.
        y_target: (..., m) compressed labels for target vertices.
        temperature: Temperature parameter T > 0. Higher T -> closer to hard max.

    Returns:
        (...,) smoothed heuristic values.
    """
    delta = y_source - y_target
    m = delta.shape[-1]

    # Mask sentinel coordinates
    sentinel_mask = _is_sentinel_vec(y_source) | _is_sentinel_vec(y_target)
    if sentinel_mask.any():
        delta = torch.where(sentinel_mask, torch.tensor(-1e30, dtype=delta.dtype, device=delta.device), delta)

    return (
        torch.logsumexp(temperature * delta, dim=-1) / temperature
        - math.log(m) / temperature
    )


def smoothed_heuristic_undirected(
    y_source: torch.Tensor,
    y_target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Smoothed heuristic for undirected graphs (L-inf norm).

    M_T(|delta|) = (1/T) * logsumexp(T * |delta|) - log(m) / T

    Approximates max_i |y_source_i - y_target_i| which is admissible because
    |d(k,u) - d(k,t)| <= d(u,t) by triangle inequality.

    NOTE: The variation norm max(delta)-min(delta) is NOT admissible on raw
    distance labels (can reach 2*d(u,t)). L-inf is the correct admissible
    heuristic for compressed distance labels.

    Sentinel coordinates are masked by setting |delta| to 0 before logsumexp.

    Args:
        y_source: (..., m) compressed labels for source vertices.
        y_target: (..., m) compressed labels for target vertices.
        temperature: Temperature parameter T > 0.

    Returns:
        (...,) smoothed heuristic values.
    """
    delta = y_source - y_target
    m = delta.shape[-1]

    abs_delta = torch.abs(delta)

    # Mask sentinel coordinates (set |delta| to 0 so they don't contribute)
    sentinel_mask = _is_sentinel_vec(y_source) | _is_sentinel_vec(y_target)
    if sentinel_mask.any():
        abs_delta = torch.where(
            sentinel_mask,
            torch.zeros_like(abs_delta),
            abs_delta,
        )

    log_m_over_T = math.log(m) / temperature
    return (
        torch.logsumexp(temperature * abs_delta, dim=-1) / temperature
        - log_m_over_T
    )


def make_aac_heuristic(
    compressed_labels: torch.Tensor,
    is_directed: bool,
) -> Callable[[int, int], float]:
    """Create A*-compatible heuristic callable from compressed labels.

    Returns a function h(node, target) -> float that computes the hard
    (non-smoothed) heuristic from compressed label vectors.

    For directed graphs: h(u,t) = max(y[u] - y[t])
    For undirected graphs: h(u,t) = max_i |y[u,i] - y[t,i]|  (L-inf norm)

    NOTE: The variation norm max(delta)-min(delta) is NOT admissible on raw
    distance labels (can reach 2*d(u,t)). L-inf is admissible because
    |d(k,u) - d(k,t)| <= d(u,t) by triangle inequality.

    Sentinel coordinates are masked out before computing max.

    Args:
        compressed_labels: (V, m) compressed label vectors.
        is_directed: Whether the graph is directed.

    Returns:
        Callable h(node: int, target: int) -> float.
    """
    y = compressed_labels

    def h(node: int, target: int) -> float:
        y_node = y[node]
        y_target = y[target]
        delta = y_node - y_target
        valid = ~(_is_sentinel_vec(y_node) | _is_sentinel_vec(y_target))

        if not valid.any():
            return 0.0

        delta_valid = delta[valid]
        if is_directed:
            return max(0.0, torch.max(delta_valid).item())
        else:
            return max(0.0, torch.max(torch.abs(delta_valid)).item())

    return h
