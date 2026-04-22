"""Tropical (min-plus) semiring: sum = min, mul = +, zero = +inf, one = 0."""

from __future__ import annotations

import torch

from aac.semirings.base import Semiring


class TropicalSemiring(Semiring):
    """Min-plus semiring: sum = min, mul = +, zero = +inf, one = 0.

    This is the algebraic foundation for shortest-path computation.
    The "addition" operation is min (finding shortest), and
    "multiplication" is + (extending a path by an edge weight).
    """

    @staticmethod
    def sum(xs: torch.Tensor, dim: int) -> torch.Tensor:
        """Semiring addition: min along the given dimension."""
        return torch.min(xs, dim=dim).values

    @staticmethod
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Semiring multiplication: standard addition (path extension)."""
        return a + b

    @staticmethod
    def zero(
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Additive identity: +inf (min(x, inf) = x for all x)."""
        return torch.tensor(float("inf"), dtype=dtype, device=device)

    @staticmethod
    def one(
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Multiplicative identity: 0 (x + 0 = x for all x)."""
        return torch.tensor(0.0, dtype=dtype, device=device)
