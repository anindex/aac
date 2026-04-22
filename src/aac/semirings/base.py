"""Abstract semiring base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Semiring(ABC):
    """Abstract semiring with additive (sum) and multiplicative (mul) operations.

    A semiring (S, +, *, 0, 1) satisfies:
    - (S, +, 0) is a commutative monoid (associative, commutative, identity 0)
    - (S, *, 1) is a monoid (associative, identity 1)
    - * distributes over +
    - 0 annihilates: a * 0 = 0 * a = 0
    """

    @staticmethod
    @abstractmethod
    def sum(xs: torch.Tensor, dim: int) -> torch.Tensor:
        """Semiring addition (aggregation) along dimension."""
        ...

    @staticmethod
    @abstractmethod
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Semiring multiplication (combination)."""
        ...

    @staticmethod
    @abstractmethod
    def zero(
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Additive identity element."""
        ...

    @staticmethod
    @abstractmethod
    def one(
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Multiplicative identity element."""
        ...
