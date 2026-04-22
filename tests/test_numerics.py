"""Tests for log-domain arithmetic utilities and numerical stability."""

import math

import torch

from aac.utils.numerics import (
    LOG_SENTINEL,
    SENTINEL,
    is_sentinel,
    safe_exp,
    safe_log,
    shifted_softmin,
)


class TestShiftedSoftmin:
    """Test shifted_softmin numerical properties."""

    def test_shifted_softmin_upper_bound(self) -> None:
        """shifted_softmin(v, beta=1.0) <= min(v) for random inputs in [0, 100000]."""
        torch.manual_seed(123)
        for _ in range(100):
            v = torch.rand(50, dtype=torch.float64) * 100000.0
            result = shifted_softmin(v, beta=1.0)
            v_min = torch.min(v)
            assert result.item() <= v_min.item() + 1e-10, (
                f"softmin {result.item()} > min {v_min.item()}"
            )

    def test_shifted_softmin_high_beta_stability(self) -> None:
        """shifted_softmin with beta=50.0 in fp64 does not produce NaN or Inf."""
        torch.manual_seed(456)
        v = torch.rand(100, dtype=torch.float64) * 100000.0
        result = shifted_softmin(v, beta=50.0)
        assert torch.isfinite(result).all(), f"Got non-finite result: {result}"

    def test_shifted_softmin_convergence(self) -> None:
        """shifted_softmin approaches min(v) as beta increases."""
        v = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float64)
        v_min = torch.min(v)
        result = shifted_softmin(v, beta=100.0)
        assert abs(result.item() - v_min.item()) < 0.01, (
            f"At beta=100, softmin {result.item()} not within 0.01 of min {v_min.item()}"
        )

    def test_shifted_softmin_gradcheck(self) -> None:
        """shifted_softmin is differentiable (torch.autograd.gradcheck passes)."""
        v = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)

        def fn(x: torch.Tensor) -> torch.Tensor:
            return shifted_softmin(x, beta=1.0)

        assert torch.autograd.gradcheck(fn, (v,), eps=1e-6, atol=1e-4)

    def test_softmin_with_sentinel(self) -> None:
        """softmin of [3.0, 5.0, SENTINEL] equals softmin of [3.0, 5.0] approximately."""
        v_with_sentinel = torch.tensor([3.0, 5.0, SENTINEL], dtype=torch.float64)
        v_without_sentinel = torch.tensor([3.0, 5.0], dtype=torch.float64)

        # Use high beta to approximate min
        result_with = shifted_softmin(v_with_sentinel, beta=50.0)
        result_without = shifted_softmin(v_without_sentinel, beta=50.0)

        # Both should be very close to 3.0, the sentinel should not affect the result
        assert abs(result_with.item() - result_without.item()) < 0.01, (
            f"With sentinel: {result_with.item()}, without: {result_without.item()}"
        )

    def test_shifted_softmin_batched(self) -> None:
        """shifted_softmin works on 2D tensors along specified dim."""
        v = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=torch.float64)
        result = shifted_softmin(v, beta=1.0, dim=-1)
        assert result.shape == (2,)
        # Each row's softmin should be <= its min
        for i in range(2):
            assert result[i].item() <= v[i].min().item() + 1e-10


class TestSentinel:
    """Test sentinel value constants and round-trip."""

    def test_sentinel_log_roundtrip(self) -> None:
        """log(SENTINEL) is finite, exp(log(SENTINEL)) approx equals SENTINEL."""
        log_val = math.log(SENTINEL)
        assert math.isfinite(log_val), f"log(SENTINEL) = {log_val} is not finite"
        assert abs(log_val - LOG_SENTINEL) < 1e-10
        roundtrip = math.exp(log_val)
        assert abs(roundtrip - SENTINEL) / SENTINEL < 1e-10, (
            f"exp(log(SENTINEL)) = {roundtrip} != SENTINEL = {SENTINEL}"
        )


class TestSafeLog:
    """Test safe_log edge cases."""

    def test_safe_log_zero(self) -> None:
        """safe_log(tensor([0.0])) does not produce -inf or NaN."""
        x = torch.tensor([0.0], dtype=torch.float64)
        result = safe_log(x)
        assert torch.isfinite(result).all(), f"safe_log(0) = {result}"
        # Should be a large negative number, not -inf
        assert result.item() < 0

    def test_safe_log_positive(self) -> None:
        """safe_log on positive values matches torch.log."""
        x = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float64)
        result = safe_log(x)
        expected = torch.log(x)
        assert torch.allclose(result, expected, atol=1e-10)


class TestSafeExp:
    """Test safe_exp edge cases."""

    def test_safe_exp_overflow(self) -> None:
        """safe_exp(tensor([1000.0])) returns value <= SENTINEL."""
        x = torch.tensor([1000.0], dtype=torch.float64)
        result = safe_exp(x)
        assert torch.isfinite(result).all(), f"safe_exp(1000) = {result}"
        assert result.item() <= SENTINEL + 1.0

    def test_safe_exp_normal(self) -> None:
        """safe_exp on normal values matches torch.exp."""
        x = torch.tensor([0.0, 1.0, 5.0], dtype=torch.float64)
        result = safe_exp(x)
        expected = torch.exp(x)
        assert torch.allclose(result, expected, atol=1e-10)


class TestIsSentinel:
    """Test sentinel detection."""

    def test_is_sentinel_detects_sentinel(self) -> None:
        """is_sentinel correctly identifies sentinel values."""
        x = torch.tensor([1.0, 5.0, SENTINEL, SENTINEL * 0.995, 100.0], dtype=torch.float64)
        mask = is_sentinel(x)
        assert mask[2].item() is True  # exact sentinel
        assert mask[3].item() is True  # near sentinel
        assert mask[0].item() is False  # normal value
        assert mask[4].item() is False  # normal value
