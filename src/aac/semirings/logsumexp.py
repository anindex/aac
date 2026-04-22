"""LogSumExp (smooth tropical) semiring with temperature parameter.

Uses shifted formulation for numerical stability (per PITFALLS.md Pitfall 1):
x_min shift ensures all exponentials in [0, 1], preventing overflow.
"""

from __future__ import annotations

import torch


class LogSumExpSemiring:
    """Smooth approximation to the tropical semiring, parameterized by temperature T.

    As T -> infinity, this approaches the tropical (min-plus) semiring.
    The sum operation is:
        sum_T(v) = v_min - (1/T) * log(sum(exp(-T * (v_i - v_min))))

    This is always a lower bound on the true minimum: sum_T(v) <= min(v).

    Note: This is an instance-based class (not pure static) because
    temperature T varies per use case. mul is still static since it
    does not depend on T.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def sum(self, xs: torch.Tensor, dim: int) -> torch.Tensor:
        """Smooth semiring addition: shifted log-sum-exp approximation to min.

        Uses the numerically stable formulation:
            result = x_min - (1/T) * logsumexp(-T * (x - x_min), dim)

        The shift by x_min ensures all arguments to exp() are in (-inf, 0],
        so exponentials are in [0, 1] -- no overflow possible.
        """
        T = self.temperature
        x_min = torch.min(xs, dim=dim, keepdim=True).values
        shifted = -T * (xs - x_min)
        return x_min.squeeze(dim) - torch.logsumexp(shifted, dim=dim) / T

    @staticmethod
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Semiring multiplication: standard addition (same as tropical)."""
        return a + b

    def zero(
        self,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Additive identity: +inf (same as tropical)."""
        return torch.tensor(float("inf"), dtype=dtype, device=device)

    def one(
        self,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Multiplicative identity: 0 (same as tropical)."""
        return torch.tensor(0.0, dtype=dtype, device=device)
