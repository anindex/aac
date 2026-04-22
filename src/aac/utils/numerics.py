"""Log-domain arithmetic, shifted softmin, and sentinel handling.

All functions operate on PyTorch tensors and support autograd.
Uses fp64 for numerical stability with values spanning [0, 100000].
"""

from __future__ import annotations

import math

import torch

# Sentinel value for unreachable distances.
# Uses 1e18 (not inf) to avoid NaN from inf - inf in embedding differences.
# log(1e18) ~ 41.4, which stays safe in log-domain arithmetic.
SENTINEL: float = 1e18

# Pre-computed log of sentinel.
LOG_SENTINEL: float = math.log(1e18)


def shifted_softmin(
    v: torch.Tensor,
    beta: float = 1.0,
    dim: int = -1,
) -> torch.Tensor:
    """Numerically stable softmin that guarantees result <= min(v).

    Formula:
        softmin_beta(v) = v_min - (1/beta) * log(sum(exp(-beta * (v - v_min))))

    The shift by v_min ensures all exponentials are in [0, 1], preventing overflow.
    Uses torch.logsumexp for the log-sum-exp computation.

    Args:
        v: Input tensor of values (typically distances).
        beta: Inverse temperature. Higher beta -> closer to hard min.
            Safe for any beta with fp64. For fp32, use beta <= 50.
        dim: Dimension along which to compute softmin.

    Returns:
        Softmin values. Guaranteed to be <= min(v, dim) for all inputs.
    """
    v_min = torch.min(v, dim=dim, keepdim=True).values
    shifted = -beta * (v - v_min)
    log_sum = torch.logsumexp(shifted, dim=dim, keepdim=True)
    result = v_min - log_sum / beta
    return result.squeeze(dim)


def safe_log(x: torch.Tensor, eps: float = 1e-45) -> torch.Tensor:
    """Compute log(x) with clamping to avoid log(0) = -inf.

    Args:
        x: Input tensor (should be non-negative).
        eps: Minimum value to clamp x to before taking log.

    Returns:
        log(clamp(x, min=eps)). Returns large negative value for x near 0,
        never -inf or NaN.
    """
    return torch.log(torch.clamp(x, min=eps))


def safe_exp(x: torch.Tensor, max_val: float = LOG_SENTINEL) -> torch.Tensor:
    """Compute exp(x) with clamping to avoid overflow.

    Args:
        x: Input tensor.
        max_val: Maximum value to clamp x to before taking exp.
            Defaults to LOG_SENTINEL (~41.4), so result <= SENTINEL.

    Returns:
        exp(clamp(x, max=max_val)). Never returns inf.
    """
    return torch.exp(torch.clamp(x, max=max_val))


def is_sentinel(x: torch.Tensor, sentinel: float = SENTINEL) -> torch.Tensor:
    """Detect sentinel values in a tensor.

    Uses a 0.99 factor to catch values near the sentinel (e.g., after
    arithmetic operations that slightly change the value).

    Args:
        x: Input tensor.
        sentinel: The sentinel value to detect.

    Returns:
        Boolean tensor: True where x >= sentinel * 0.99.
    """
    return x >= sentinel * 0.99
