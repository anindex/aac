"""Custom autograd.Function for differentiable min-plus operations.

PyTorch's sparse CSR backward is broken (issue #86963), so we implement
custom backward passes for min-plus matrix-vector and matrix-matrix multiply.
Gradient flows only through the argmin path for each output element.
"""

from __future__ import annotations

import torch
from torch.autograd import Function


class MinPlusMatVec(Function):
    """Min-plus matrix-vector: c[i] = min_j(A[i,j] + x[j]).

    Forward: broadcast A[i,j] + x[j], take min over j.
    Backward: gradient flows only through the argmin index.
      dA[i, j*] = dc[i] where j* = argmin_j(A[i,j] + x[j])
      dx[j] = sum_{i: argmin=j} dc[i]
    """

    # NOTE: clear_saved_tensors_on_access=True breaks gradcheck (which calls
    # backward multiple times). Disabled for correctness-testing compatibility.
    # Memory savings are negligible for the small tensors involved (argmin indices).

    @staticmethod
    def forward(ctx, A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # A: (M, N), x: (N,) -> c: (M,)
        sums = A + x.unsqueeze(0)  # (M, N) broadcast
        c, argmins = torch.min(sums, dim=1)  # (M,)
        ctx.save_for_backward(argmins)
        ctx.shape = A.shape
        return c

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmins,) = ctx.saved_tensors
        M, N = ctx.shape

        # dA: sparse -- only dA[i, argmin[i]] = grad_c[i]
        grad_A = torch.zeros(M, N, device=grad_c.device, dtype=grad_c.dtype)
        grad_A.scatter_(1, argmins.unsqueeze(1), grad_c.unsqueeze(1))

        # dx: accumulate grad_c at argmin positions
        grad_x = torch.zeros(N, device=grad_c.device, dtype=grad_c.dtype)
        grad_x.scatter_add_(0, argmins, grad_c)

        return grad_A, grad_x


class MinPlusMatMat(Function):
    """Min-plus matrix-matrix: C[i,j] = min_k(A[i,k] + B[k,j]).

    Forward: for each (i,j), compute min_k(A[i,k] + B[k,j]).
    Backward: gradient flows through the argmin k for each (i,j).
      dA[i, k*] += dC[i,j] where k* = argmin_k for (i,j)
      dB[k*, j] += dC[i,j]
    """

    # NOTE: clear_saved_tensors_on_access=True breaks gradcheck (which calls
    # backward multiple times). Disabled for correctness-testing compatibility.
    # Memory savings are negligible for the small tensors involved (argmin indices).

    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: (M, K), B: (K, N) -> C: (M, N)
        # Expand: A[i,k] + B[k,j] -> (M, K, N)
        sums = A.unsqueeze(2) + B.unsqueeze(0)  # (M, K, N)
        C, argmins = torch.min(sums, dim=1)  # (M, N), argmins (M, N)
        ctx.save_for_backward(argmins)
        ctx.shapes = (A.shape, B.shape)
        return C

    @staticmethod
    def backward(ctx, grad_C: torch.Tensor):
        (argmins,) = ctx.saved_tensors
        (M, K), (_, N) = ctx.shapes

        # grad_A[i, k*] = sum_j grad_C[i, j] where k* = argmins[i, j]
        grad_A = torch.zeros(M, K, device=grad_C.device, dtype=grad_C.dtype)
        grad_A.scatter_add_(1, argmins, grad_C)

        # grad_B[k*, j] = sum_i grad_C[i, j] where k* = argmins[i, j]
        # For each (i, j): grad_B[argmins[i,j], j] += grad_C[i, j]
        grad_B = torch.zeros(K, N, device=grad_C.device, dtype=grad_C.dtype)
        argmins_flat = argmins.reshape(-1)  # (M*N,)
        j_indices = torch.arange(N, device=grad_C.device).unsqueeze(0).expand(M, N).reshape(-1)
        flat_indices = argmins_flat * N + j_indices  # linear index into (K, N)
        # Use .view(-1) to guarantee a view (not a copy), so scatter_add_
        # writes into grad_B's storage. .reshape(-1) may return a copy for
        # non-contiguous tensors, silently producing zero gradients.
        grad_B.view(-1).scatter_add_(0, flat_indices, grad_C.reshape(-1))

        return grad_A, grad_B
