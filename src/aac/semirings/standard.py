"""Standard semiring: sum = +, mul = *, zero = 0, one = 1."""

from __future__ import annotations

import torch

from aac.semirings.base import Semiring


class StdSemiring(Semiring):
    """Standard (+, *) semiring over reals.

    This is the familiar arithmetic semiring. Provided for completeness
    and as a baseline comparison for tropical/LogSumExp semirings.
    """

    @staticmethod
    def sum(xs: torch.Tensor, dim: int) -> torch.Tensor:
        """Semiring addition: standard sum along dimension."""
        return torch.sum(xs, dim=dim)

    @staticmethod
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Semiring multiplication: standard element-wise product."""
        return a * b

    @staticmethod
    def zero(
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Additive identity: 0."""
        return torch.tensor(0.0, dtype=dtype, device=device)

    @staticmethod
    def one(
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Multiplicative identity: 1."""
        return torch.tensor(1.0, dtype=dtype, device=device)
