"""Min-plus sparse matrix operations using dense fallback.

Public API for min-plus SpMV and SpMM with autograd support.
These use custom autograd.Function because PyTorch's sparse CSR
backward is broken (issue #86963).

Note: O(M*K*N) memory for intermediate tensor in SpMM.
For graphs with V > 10K, use sparse edge-parallel Bellman-Ford instead.
"""

from __future__ import annotations

import torch

from aac.semirings._autograd import MinPlusMatMat, MinPlusMatVec


def minplus_spmv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Min-plus sparse matrix-vector multiply: c[i] = min_j(A[i,j] + x[j]).

    Uses dense fallback. A: (M, N), x: (N,) -> c: (M,).
    Supports autograd via custom backward pass.
    Handles sentinel values (inf): inf + x = inf, min(inf, y) = y.

    Args:
        A: (M, N) matrix of weights/distances.
        x: (N,) vector.

    Returns:
        (M,) result vector where c[i] = min_j(A[i,j] + x[j]).
    """
    return MinPlusMatVec.apply(A, x)


def minplus_spmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Min-plus matrix-matrix multiply: C[i,j] = min_k(A[i,k] + B[k,j]).

    Uses dense fallback. A: (M, K), B: (K, N) -> C: (M, N).
    Supports autograd via custom backward pass.

    Warning: O(M*K*N) memory for intermediate tensor. For graphs with
    V > 10K, use sparse edge-parallel Bellman-Ford instead.

    Args:
        A: (M, K) matrix.
        B: (K, N) matrix.

    Returns:
        (M, N) result matrix where C[i,j] = min_k(A[i,k] + B[k,j]).
    """
    return MinPlusMatMat.apply(A, B)
