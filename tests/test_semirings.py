"""Tests for tropical, LogSumExp, and standard semiring abstractions.

Covers: TROP-01, TROP-02, TROP-04 requirements.
"""

from __future__ import annotations

import pytest
import torch


# ── Tropical Semiring Tests ──────────────────────────────────────────────────

class TestTropicalSemiring:
    """Tests for TropicalSemiring algebraic properties."""

    def test_tropical_sum_is_min(self) -> None:
        """TropicalSemiring.sum should return the minimum along the given dimension."""
        from aac.semirings import TropicalSemiring

        v = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0], dtype=torch.float64)
        result = TropicalSemiring.sum(v, dim=0)
        assert result.item() == pytest.approx(1.0)

        # 2D case: min along dim=1
        m = torch.tensor([[3.0, 1.0], [5.0, 2.0]], dtype=torch.float64)
        result = TropicalSemiring.sum(m, dim=1)
        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_tropical_mul_is_add(self) -> None:
        """TropicalSemiring.mul should return a + b."""
        from aac.semirings import TropicalSemiring

        a = torch.tensor(3.0, dtype=torch.float64)
        b = torch.tensor(4.0, dtype=torch.float64)
        result = TropicalSemiring.mul(a, b)
        assert result.item() == pytest.approx(7.0)

    def test_tropical_zero_is_inf(self) -> None:
        """TropicalSemiring.zero should return +inf (additive identity for min)."""
        from aac.semirings import TropicalSemiring

        z = TropicalSemiring.zero()
        assert z.item() == float("inf")
        assert z.dtype == torch.float64

    def test_tropical_one_is_zero(self) -> None:
        """TropicalSemiring.one should return 0 (multiplicative identity for +)."""
        from aac.semirings import TropicalSemiring

        o = TropicalSemiring.one()
        assert o.item() == 0.0
        assert o.dtype == torch.float64

    def test_tropical_associativity(self) -> None:
        """sum(sum(a,b), c) == sum(a, sum(b,c)) for random a, b, c."""
        from aac.semirings import TropicalSemiring

        torch.manual_seed(123)
        for _ in range(10):
            a, b, c = torch.rand(3, dtype=torch.float64) * 100

            # Stack and compute: sum(sum(a,b),c) vs sum(a, sum(b,c))
            ab = TropicalSemiring.sum(torch.stack([a, b]), dim=0)
            lhs = TropicalSemiring.sum(torch.stack([ab, c]), dim=0)

            bc = TropicalSemiring.sum(torch.stack([b, c]), dim=0)
            rhs = TropicalSemiring.sum(torch.stack([a, bc]), dim=0)

            assert lhs.item() == pytest.approx(rhs.item(), abs=1e-12)

    def test_tropical_identity(self) -> None:
        """mul(a, one) == a and sum(a, zero) == a."""
        from aac.semirings import TropicalSemiring

        torch.manual_seed(456)
        for _ in range(10):
            a = torch.rand(1, dtype=torch.float64).squeeze() * 100

            # Multiplicative identity: mul(a, one) == a
            one = TropicalSemiring.one()
            assert TropicalSemiring.mul(a, one).item() == pytest.approx(a.item(), abs=1e-12)

            # Additive identity: sum(a, zero) == a
            zero = TropicalSemiring.zero()
            result = TropicalSemiring.sum(torch.stack([a, zero]), dim=0)
            assert result.item() == pytest.approx(a.item(), abs=1e-12)

    def test_tropical_distributivity(self) -> None:
        """mul(a, sum(b,c)) == sum(mul(a,b), mul(a,c)) for scalars."""
        from aac.semirings import TropicalSemiring

        torch.manual_seed(789)
        for _ in range(10):
            a, b, c = torch.rand(3, dtype=torch.float64) * 100

            # Left side: a * min(b, c) = a + min(b, c)
            bc_sum = TropicalSemiring.sum(torch.stack([b, c]), dim=0)
            lhs = TropicalSemiring.mul(a, bc_sum)

            # Right side: min(a*b, a*c) = min(a+b, a+c)
            ab = TropicalSemiring.mul(a, b)
            ac = TropicalSemiring.mul(a, c)
            rhs = TropicalSemiring.sum(torch.stack([ab, ac]), dim=0)

            assert lhs.item() == pytest.approx(rhs.item(), abs=1e-12)


# ── LogSumExp Semiring Tests ─────────────────────────────────────────────────

class TestLogSumExpSemiring:
    """Tests for LogSumExpSemiring (smooth tropical approximation)."""

    def test_logsumexp_lower_bound(self) -> None:
        """LSE(T).sum(v) <= min(v) + eps for random v, various T."""
        from aac.semirings import LogSumExpSemiring, TropicalSemiring

        torch.manual_seed(111)
        for T in [1.0, 10.0, 50.0]:
            lse = LogSumExpSemiring(temperature=T)
            for _ in range(10):
                v = torch.rand(20, dtype=torch.float64) * 100
                lse_val = lse.sum(v, dim=0)
                trop_val = TropicalSemiring.sum(v, dim=0)
                # LSE should be a lower bound (or very close) to the true min
                assert lse_val.item() <= trop_val.item() + 1e-10, (
                    f"LSE({T}).sum > min + eps: {lse_val.item()} > {trop_val.item()}"
                )

    def test_logsumexp_convergence(self) -> None:
        """As T increases (1, 10, 100), LSE.sum approaches TropicalSemiring.sum."""
        from aac.semirings import LogSumExpSemiring, TropicalSemiring

        torch.manual_seed(222)
        v = torch.rand(20, dtype=torch.float64) * 100
        trop_val = TropicalSemiring.sum(v, dim=0).item()

        prev_gap = float("inf")
        for T in [1.0, 10.0, 100.0]:
            lse = LogSumExpSemiring(temperature=T)
            lse_val = lse.sum(v, dim=0).item()
            gap = abs(trop_val - lse_val)
            # Gap should decrease as T increases
            assert gap < prev_gap + 1e-12, (
                f"Gap not decreasing: T={T}, gap={gap}, prev_gap={prev_gap}"
            )
            prev_gap = gap

        # At T=100, gap should be very small
        assert prev_gap < 0.1, f"Gap at T=100 too large: {prev_gap}"


# ── Standard Semiring Tests ──────────────────────────────────────────────────

class TestStdSemiring:
    """Tests for StdSemiring (standard +,* semiring)."""

    def test_std_semiring_basic(self) -> None:
        """StdSemiring.sum is torch.sum, StdSemiring.mul is torch.mul."""
        from aac.semirings import StdSemiring

        v = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        assert StdSemiring.sum(v, dim=0).item() == pytest.approx(10.0)

        a = torch.tensor(3.0, dtype=torch.float64)
        b = torch.tensor(4.0, dtype=torch.float64)
        assert StdSemiring.mul(a, b).item() == pytest.approx(12.0)

        assert StdSemiring.zero().item() == 0.0
        assert StdSemiring.one().item() == 1.0


# ── Dtype Tests ──────────────────────────────────────────────────────────────

class TestSemiringDtype:
    """Tests that all semirings preserve fp64."""

    def test_semirings_fp64(self) -> None:
        """All operations produce fp64 output from fp64 input."""
        from aac.semirings import LogSumExpSemiring, StdSemiring, TropicalSemiring

        v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        a = torch.tensor(1.0, dtype=torch.float64)
        b = torch.tensor(2.0, dtype=torch.float64)

        for SR in [TropicalSemiring, StdSemiring]:
            assert SR.sum(v, dim=0).dtype == torch.float64
            assert SR.mul(a, b).dtype == torch.float64
            assert SR.zero().dtype == torch.float64
            assert SR.one().dtype == torch.float64

        lse = LogSumExpSemiring(temperature=1.0)
        assert lse.sum(v, dim=0).dtype == torch.float64
        assert lse.mul(a, b).dtype == torch.float64
        assert lse.zero().dtype == torch.float64
        assert lse.one().dtype == torch.float64


# ── Min-Plus SpMV/SpMM Tests ────────────────────────────────────────────────

class TestMinPlusSpMV:
    """Tests for min-plus sparse matrix-vector multiply."""

    def test_minplus_spmv_correctness(self) -> None:
        """minplus_spmv(A, x) matches brute-force: c[i] = min_j(A[i,j] + x[j])."""
        from aac.semirings import minplus_spmv

        INF = float("inf")
        A = torch.tensor(
            [[1.0, 3.0, INF], [2.0, INF, 4.0]], dtype=torch.float64
        )
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

        result = minplus_spmv(A, x)

        # Row 0: min(1+0, 3+1, inf+2) = min(1, 4, inf) = 1
        # Row 1: min(2+0, inf+1, 4+2) = min(2, inf, 6) = 2
        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_minplus_spmv_sentinel(self) -> None:
        """A row of all inf produces inf output. A column of inf is ignored."""
        from aac.semirings import minplus_spmv

        INF = float("inf")
        # Row 0: all inf -> result should be inf
        # Row 1: normal values
        A = torch.tensor(
            [[INF, INF], [1.0, 3.0]], dtype=torch.float64
        )
        x = torch.tensor([0.0, 1.0], dtype=torch.float64)

        result = minplus_spmv(A, x)
        assert result[0].item() == INF, f"Expected inf for all-inf row, got {result[0]}"
        assert result[1].item() == pytest.approx(1.0), f"Expected 1.0, got {result[1]}"

    def test_minplus_identity_matrix(self) -> None:
        """minplus_spmv with identity-like matrix (0 on diag, inf elsewhere) returns x."""
        from aac.semirings import minplus_spmv

        INF = float("inf")
        N = 5
        I_minplus = torch.full((N, N), INF, dtype=torch.float64)
        I_minplus.fill_diagonal_(0.0)

        torch.manual_seed(42)
        x = torch.rand(N, dtype=torch.float64) * 100

        result = minplus_spmv(I_minplus, x)
        assert torch.allclose(result, x), f"Expected x, got {result}"

    @pytest.mark.gpu
    def test_minplus_spmv_gpu(self) -> None:
        """minplus_spmv on GPU produces same result as CPU."""
        from aac.semirings import minplus_spmv

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(99)
        A = torch.rand(4, 5, dtype=torch.float64) * 10 + 1
        x = torch.rand(5, dtype=torch.float64) * 10 + 1

        cpu_result = minplus_spmv(A, x)
        gpu_result = minplus_spmv(A.cuda(), x.cuda())

        assert torch.allclose(cpu_result, gpu_result.cpu()), (
            f"GPU result differs from CPU: max diff = {(cpu_result - gpu_result.cpu()).abs().max()}"
        )


class TestMinPlusSpMM:
    """Tests for min-plus sparse matrix-matrix multiply."""

    def test_minplus_spmm_correctness(self) -> None:
        """minplus_spmm(A, B) matches naive triple-loop reference."""
        from aac.semirings import minplus_spmm

        torch.manual_seed(42)
        M, K, N = 3, 4, 5
        A = torch.rand(M, K, dtype=torch.float64) * 10 + 1  # in [1, 11]
        B = torch.rand(K, N, dtype=torch.float64) * 10 + 1

        result = minplus_spmm(A, B)

        # Brute-force reference
        expected = torch.full((M, N), float("inf"), dtype=torch.float64)
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    val = A[i, k] + B[k, j]
                    if val < expected[i, j]:
                        expected[i, j] = val

        assert torch.allclose(result, expected, atol=1e-12), (
            f"Max diff: {(result - expected).abs().max()}"
        )

    def test_minplus_spmm_identity(self) -> None:
        """minplus_spmm with identity-like matrix returns input unchanged."""
        from aac.semirings import minplus_spmm

        INF = float("inf")
        K = 4
        I_minplus = torch.full((K, K), INF, dtype=torch.float64)
        I_minplus.fill_diagonal_(0.0)

        torch.manual_seed(55)
        B = torch.rand(K, 3, dtype=torch.float64) * 100

        result = minplus_spmm(I_minplus, B)
        assert torch.allclose(result, B), f"Identity * B != B"


# ── Autograd / Gradcheck Tests ──────────────────────────────────────────────

class TestMinPlusAutograd:
    """Tests for custom autograd backward pass of min-plus operations."""

    def test_minplus_spmv_gradcheck(self) -> None:
        """MinPlusMatVec passes torch.autograd.gradcheck with fp64."""
        from aac.semirings._autograd import MinPlusMatVec

        torch.manual_seed(33)
        # Use values in [1, 10] to avoid ties and inf for stable finite differences
        A = torch.rand(3, 4, dtype=torch.float64) * 9 + 1
        x = torch.rand(4, dtype=torch.float64) * 9 + 1
        A.requires_grad_(True)
        x.requires_grad_(True)

        assert torch.autograd.gradcheck(MinPlusMatVec.apply, (A, x), eps=1e-6)

    def test_minplus_spmm_gradcheck(self) -> None:
        """MinPlusMatMat passes torch.autograd.gradcheck with fp64."""
        from aac.semirings._autograd import MinPlusMatMat

        torch.manual_seed(44)
        A = torch.rand(3, 4, dtype=torch.float64) * 9 + 1
        B = torch.rand(4, 5, dtype=torch.float64) * 9 + 1
        A.requires_grad_(True)
        B.requires_grad_(True)

        assert torch.autograd.gradcheck(MinPlusMatMat.apply, (A, B), eps=1e-6)

    def test_minplus_gradient_sparsity(self) -> None:
        """Gradient of A in SpMV has at most M nonzero entries (one per row)."""
        from aac.semirings._autograd import MinPlusMatVec

        torch.manual_seed(55)
        M, N = 5, 8
        A = torch.rand(M, N, dtype=torch.float64) * 9 + 1
        x = torch.rand(N, dtype=torch.float64) * 9 + 1
        A.requires_grad_(True)

        c = MinPlusMatVec.apply(A, x)
        c.sum().backward()

        grad_A = A.grad
        assert grad_A is not None

        # Each row should have exactly 1 non-zero gradient entry (the argmin)
        for i in range(M):
            nonzero_count = (grad_A[i] != 0).sum().item()
            assert nonzero_count == 1, (
                f"Row {i} has {nonzero_count} nonzero gradient entries, expected 1"
            )
