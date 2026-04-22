"""Compression modules for AAC.

Primary: LinearCompressor (Proposition "admissibility") -- row-stochastic
linear compression via Gumbel-softmax landmark selection.

Admissibility guarantee (Proposition "admissibility", paper Sec.3): For any
m x K row-stochastic matrix A (nonneg entries, rows sum to 1), the
compressed heuristic satisfies:
    h_A(u, t) <= h_teacher(u, t) <= d(u, t)
because each compressed dimension is a convex combination of teacher
differences, and a convex combination cannot exceed the maximum.

At inference, hard argmax selection produces one-hot rows (a special case
of row-stochastic), preserving the admissibility guarantee
(Corollary "inference").

During training, Gumbel-softmax with straight-through estimator provides
gradient flow while maintaining one-hot forward passes.

Also contains: PositiveCompressor (max-plus contraction, log-domain
variant) and DualCompressor (separate fwd/bwd PositiveCompressor pair).
These are retained for comparison but LinearCompressor is the production
compressor used in all AAC experiments.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCompressor(nn.Module):
    """Row-stochastic linear compression with Gumbel-softmax hard selection.

    Compresses K landmark distances to m dimensions via learned landmark
    selection. During training, uses Gumbel-softmax with straight-through
    estimator to learn which landmarks to keep. At inference, uses hard
    argmax selection: each compressed dimension selects exactly one landmark.

    Training mode (soft, with gradients):
        A = gumbel_softmax(W / tau, hard=True)  # straight-through one-hot
        y[v, :] = A @ d[v, :]

    Inference mode:
        y[v, i] = d[v, argmax(W[i,:])]  # exact landmark selection

    Admissibility guarantee: row-stochastic A (convex combination) satisfies
    max_i(A@delta) <= max(delta). Hard one-hot selection is a special case
    of row-stochastic, so the guarantee holds in both modes.

    Temperature annealing: tau starts high (smooth selection, good exploration)
    and decreases toward 0 (hard selection, committed decisions).

    For directed graphs: separate fwd/bwd selection matrices.
    Memory: m total values per vertex (m/2 fwd + m/2 bwd for directed).

    Args:
        K: Number of teacher landmarks.
        m: Total compressed dimensions.
        is_directed: Whether the graph is directed.
    """

    def __init__(self, K: int, m: int, is_directed: bool = True, m_fwd_ratio: float = 0.5) -> None:
        super().__init__()
        self.K = K
        self.m = m
        self.is_directed = is_directed
        self.m_fwd_ratio = m_fwd_ratio

        if is_directed:
            self.m_fwd = max(1, int(m * m_fwd_ratio))
            self.m_bwd = m - self.m_fwd
            if self.m_bwd < 1:
                self.m_bwd = 1
                self.m_fwd = m - 1
            self.W_fwd = nn.Parameter(self._init_block_sparse(self.m_fwd, K))
            self.W_bwd = nn.Parameter(self._init_block_sparse(self.m_bwd, K))
        else:
            self.W = nn.Parameter(self._init_block_sparse(m, K))

    @staticmethod
    def _init_block_sparse(m: int, K: int) -> torch.Tensor:
        """Block-sparse initialization: each row focuses on a block of inputs."""
        W = torch.full((m, K), -5.0)
        stride = max(1, K // m)
        for i in range(m):
            j_start = i * stride
            j_end = min(j_start + stride, K)
            W[i, j_start:j_end] = 0.0
        return W + 0.1 * torch.randn(m, K)

    def _get_A_soft(self, W: torch.Tensor, tau: float = 1.0, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Gumbel-softmax with straight-through hard selection (training).

        Returns one-hot rows in forward pass but uses soft gradients for backward.

        Args:
            W: Logits tensor (m, K).
            tau: Gumbel-softmax temperature.
            dtype: Output dtype (must match data being multiplied with).
        """
        return F.gumbel_softmax(W.to(dtype), tau=tau, hard=True, dim=-1)

    def _get_A_hard(self, W: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Hard argmax selection (inference). True one-hot rows.

        Args:
            W: Logits tensor (m, K).
            dtype: Output dtype (must match data being multiplied with).
        """
        idx = W.argmax(dim=-1)  # (m,)
        return F.one_hot(idx, W.shape[-1]).to(dtype)  # (m, K)

    def forward(
        self,
        d_out_t: torch.Tensor,
        d_in_t: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Compress distance labels.

        Args:
            d_out_t: (V, K) or (B, K) forward distances.
            d_in_t: (V, K) or (B, K) backward distances (None for undirected).
            tau: Gumbel-softmax temperature. Only used during training.

        Returns:
            For directed: (y_fwd, y_bwd) each (V, m_half).
            For undirected: y (V, m).
        """
        # Match selection matrix dtype to input data to avoid mixed-precision errors
        # (e.g., float32 teacher labels on Netherlands graph with float64 A matrix)
        data_dtype = d_out_t.dtype

        if self.training:
            get_A = lambda W: self._get_A_soft(W, tau, dtype=data_dtype)

            if self.is_directed:
                A_fwd = get_A(self.W_fwd)  # (m_fwd, K)
                A_bwd = get_A(self.W_bwd)  # (m_bwd, K)
                y_fwd = d_out_t @ A_fwd.t()
                y_bwd = d_in_t @ A_bwd.t()
                return y_fwd, y_bwd
            else:
                A = get_A(self.W)
                return d_out_t @ A.t()
        else:
            # Eval mode: use direct index selection instead of matmul with one-hot.
            # This avoids float32 rounding errors that can violate admissibility
            # (matmul with one-hot is mathematically equivalent to indexing, but
            # float32 matmul can introduce rounding > ULP of the true value).
            if self.is_directed:
                idx_fwd = self.W_fwd.argmax(dim=-1)  # (m_fwd,)
                idx_bwd = self.W_bwd.argmax(dim=-1)  # (m_bwd,)
                y_fwd = d_out_t[:, idx_fwd]  # (V, m_fwd) exact selection
                y_bwd = d_in_t[:, idx_bwd]   # (V, m_bwd) exact selection
                return y_fwd, y_bwd
            else:
                idx = self.W.argmax(dim=-1)  # (m,)
                return d_out_t[:, idx]  # (V, m) exact selection

    def selected_landmarks(self) -> dict:
        """Return the landmarks selected by hard argmax (for inspection)."""
        if self.is_directed:
            return {
                "fwd": self.W_fwd.argmax(dim=-1).tolist(),
                "bwd": self.W_bwd.argmax(dim=-1).tolist(),
            }
        return {"landmarks": self.W.argmax(dim=-1).tolist()}

    @torch.no_grad()
    def selection_stats(self) -> dict[str, int | float]:
        """Summarize hard selections for reporting and ablations.

        The effective budget should be interpreted per directional matrix:
        duplicate rows within the same forward/backward block waste capacity,
        while selecting the same landmark once in forward and once in backward
        is expected on directed graphs.
        """
        if self.is_directed:
            idx_fwd = self.W_fwd.argmax(dim=-1)
            idx_bwd = self.W_bwd.argmax(dim=-1)
            nominal_fwd = int(idx_fwd.numel())
            nominal_bwd = int(idx_bwd.numel())
            unique_fwd = int(torch.unique(idx_fwd).numel())
            unique_bwd = int(torch.unique(idx_bwd).numel())
            effective_unique_total = unique_fwd + unique_bwd
            nominal_total = nominal_fwd + nominal_bwd
            return {
                "nominal_fwd": nominal_fwd,
                "nominal_bwd": nominal_bwd,
                "unique_fwd": unique_fwd,
                "unique_bwd": unique_bwd,
                "duplicates_fwd": nominal_fwd - unique_fwd,
                "duplicates_bwd": nominal_bwd - unique_bwd,
                "effective_unique_total": effective_unique_total,
                "effective_unique_ratio": effective_unique_total / max(nominal_total, 1),
            }

        idx = self.W.argmax(dim=-1)
        nominal = int(idx.numel())
        unique = int(torch.unique(idx).numel())
        return {
            "nominal_fwd": nominal,
            "nominal_bwd": 0,
            "unique_fwd": unique,
            "unique_bwd": 0,
            "duplicates_fwd": nominal - unique,
            "duplicates_bwd": 0,
            "effective_unique_total": unique,
            "effective_unique_ratio": unique / max(nominal, 1),
        }

    def uniqueness_penalty(self) -> torch.Tensor:
        """Penalize duplicate landmark selections across rows.

        Encourages each row to pick a different landmark by penalizing
        the squared overlap in softmax distributions between rows.
        """
        penalty = torch.tensor(0.0, dtype=torch.float64)
        Ws = [self.W_fwd, self.W_bwd] if self.is_directed else [self.W]
        for W in Ws:
            A = torch.softmax(W.to(torch.float64), dim=-1)  # (m, K)
            # Pairwise cosine similarity of rows: high = duplicate selections
            gram = A @ A.t()  # (m, m)
            # Penalize off-diagonal entries (want rows to be orthogonal)
            m_dim = gram.shape[0]
            mask = ~torch.eye(m_dim, dtype=torch.bool, device=gram.device)
            penalty = penalty + gram[mask].mean()
        return penalty

    def condition_regularization(self) -> torch.Tensor:
        """Entropy regularization: encourage sharp rows."""
        if self.is_directed:
            A_fwd = torch.softmax(self.W_fwd.to(torch.float64), dim=-1)
            A_bwd = torch.softmax(self.W_bwd.to(torch.float64), dim=-1)
            ent_fwd = -(A_fwd * torch.log(A_fwd + 1e-10)).sum(dim=-1).mean()
            ent_bwd = -(A_bwd * torch.log(A_bwd + 1e-10)).sum(dim=-1).mean()
            return ent_fwd + ent_bwd
        else:
            A = torch.softmax(self.W.to(torch.float64), dim=-1)
            return -(A * torch.log(A + 1e-10)).sum(dim=-1).mean()

    def condition_number(self) -> float:
        """Monitor effective sparsity."""
        if self.is_directed:
            A = torch.softmax(self.W_fwd.to(torch.float64), dim=-1)
        else:
            A = torch.softmax(self.W.to(torch.float64), dim=-1)
        return (A.max() / (A.min() + 1e-10)).item()

    @torch.no_grad()
    def deduplicate_selections(self) -> int:
        """Post-training deduplication: resolve duplicate landmark selections.

        When multiple rows select the same landmark via argmax, reassign
        duplicates to their next-best unique choice. This is admissibility-
        preserving because each row still selects exactly one landmark
        (one-hot row-stochastic).

        Returns:
            Number of duplicates resolved.
        """
        total_resolved = 0
        Ws = (
            [("fwd", self.W_fwd), ("bwd", self.W_bwd)]
            if self.is_directed
            else [("all", self.W)]
        )
        for name, W in Ws:
            m, K = W.shape
            if m > K:
                continue  # More rows than landmarks; can't fully deduplicate
            selections = W.argmax(dim=-1).tolist()  # current hard selections
            used = set()
            for i in range(m):
                if selections[i] not in used:
                    used.add(selections[i])
                    continue
                # Duplicate: find best unused landmark for this row
                row_scores = W[i].clone()
                # Mask out already-used landmarks
                for j in used:
                    row_scores[j] = -1e9
                new_j = row_scores.argmax().item()
                # Boost W[i, new_j] so argmax picks it
                W[i, new_j] = W[i].max().item() + 1.0
                used.add(new_j)
                total_resolved += 1
        return total_resolved


def make_linear_heuristic(
    y_fwd: torch.Tensor,
    y_bwd: torch.Tensor,
    is_directed: bool,
) -> "Callable[[int, int], float]":
    """Create A*-compatible heuristic from linear-compressed labels.

    For directed:  h(u,t) = max(0, max(y_bwd[u]-y_bwd[t]), max(y_fwd[t]-y_fwd[u]))
    For undirected: h(u,t) = max_i |y[u,i] - y[t,i]|  (L-inf norm on delta)

    Sentinel masking: When a compressed dimension contains a sentinel value
    (|y| > 0.99 * SENTINEL), that dimension is excluded from the max to
    prevent unreachable-landmark distances from corrupting the heuristic.
    If ALL dimensions are sentinel for a pair, returns 0 (Dijkstra fallback).

    L-inf is admissible because |d(k,u) - d(k,t)| <= d(u,t) by triangle
    inequality, and each compressed coordinate is a convex combination of
    landmark distances (row-stochastic A), preserving this bound.

    NOTE: The variation norm max(delta)-min(delta) is NOT admissible on raw
    distance labels because it can reach 2*d(u,t). It is only admissible in
    the log-domain (Hilbert projective metric).

    Args:
        y_fwd: (V, m_fwd) compressed forward labels.
        y_bwd: (V, m_bwd) compressed backward labels.
        is_directed: Whether the graph is directed.

    Returns:
        Callable h(node, target) -> float.
    """

    import numpy as np

    from aac.utils.numerics import SENTINEL

    # Pre-convert to numpy for faster per-query evaluation in A* inner loop.
    # Avoids torch tensor indexing + .item() overhead (~10x speedup per call).
    y_fwd_np = y_fwd.detach().cpu().numpy()
    y_bwd_np = y_bwd.detach().cpu().numpy()
    sentinel_thresh = 0.99 * SENTINEL

    # One-time scan: if no sentinels anywhere, use fast path that skips
    # per-call masking (~2x speedup on well-connected graphs after SCC
    # restriction, where sentinels are absent).
    has_sentinel_fwd = bool(np.any(np.abs(y_fwd_np) >= sentinel_thresh))
    has_sentinel_bwd = bool(np.any(np.abs(y_bwd_np) >= sentinel_thresh))
    has_any_sentinel = has_sentinel_fwd or has_sentinel_bwd

    if is_directed:
        if has_any_sentinel:
            def h(node: int, target: int) -> float:
                # Backward bound: d(u,l_k) - d(t,l_k)
                bwd_n = y_bwd_np[node]
                bwd_t = y_bwd_np[target]
                bwd_valid = (np.abs(bwd_n) < sentinel_thresh) & (np.abs(bwd_t) < sentinel_thresh)

                # Forward bound: d(l_k,t) - d(l_k,u)
                fwd_n = y_fwd_np[node]
                fwd_t = y_fwd_np[target]
                fwd_valid = (np.abs(fwd_n) < sentinel_thresh) & (np.abs(fwd_t) < sentinel_thresh)

                result = 0.0
                if bwd_valid.any():
                    result = max(result, float(np.max((bwd_n - bwd_t)[bwd_valid])))
                if fwd_valid.any():
                    result = max(result, float(np.max((fwd_t - fwd_n)[fwd_valid])))
                return result
        else:
            # Fast path: no sentinels, skip masking
            def h(node: int, target: int) -> float:
                result = float(np.max(y_bwd_np[node] - y_bwd_np[target]))
                fwd_max = float(np.max(y_fwd_np[target] - y_fwd_np[node]))
                return max(0.0, result, fwd_max)
    else:
        if has_any_sentinel:
            def h(node: int, target: int) -> float:
                yn = y_fwd_np[node]
                yt = y_fwd_np[target]
                valid = (np.abs(yn) < sentinel_thresh) & (np.abs(yt) < sentinel_thresh)
                if not valid.any():
                    return 0.0
                delta = (yn - yt)[valid]
                return max(0.0, float(np.max(np.abs(delta))))
        else:
            # Fast path: no sentinels, skip masking
            def h(node: int, target: int) -> float:
                delta = y_fwd_np[node] - y_fwd_np[target]
                return max(0.0, float(np.max(np.abs(delta))))

    return h


class DualCompressor(nn.Module):
    """Separate max-plus compression for forward and backward distance labels.

    For directed graphs, the ALT heuristic uses both forward (d_out) and
    backward (d_in) distance bounds. Compressing them jointly via the tropical
    embedding [d_in, -d_out] fails because positive d_in entries always dominate
    negative -d_out entries in max/logsumexp operations.

    DualCompressor solves this by compressing forward and backward labels
    independently, then combining at heuristic evaluation time:

        h(u,t) = max(0,
            max_i(y_bwd_i(u) - y_bwd_i(t)),  // backward bound
            max_i(y_fwd_i(t) - y_fwd_i(u))   // forward bound
        )

    Memory: m_fwd + m_bwd = m total values per vertex.

    For undirected graphs: d_in = d_out, so only the backward compressor is
    used and h(u,t) = max_i |y_i(u) - y_i(t)|.

    Args:
        K: Number of teacher landmarks (input dimension per half).
        m: Total compressed dimensions (split equally between fwd and bwd).
        sigma: Scale parameter for smooth training. Auto-set from data if None.
    """

    def __init__(self, K: int, m: int, sigma: float | None = None) -> None:
        super().__init__()
        self.K = K
        self.m = m
        self.m_fwd = m // 2
        self.m_bwd = m - self.m_fwd

        self.comp_fwd = PositiveCompressor(K, self.m_fwd, sigma=sigma)
        self.comp_bwd = PositiveCompressor(K, self.m_bwd, sigma=sigma)

    @property
    def sigma(self) -> float:
        return self.comp_fwd.sigma

    def set_sigma_from_data(self, d_out: torch.Tensor, d_in: torch.Tensor) -> None:
        """Auto-set sigma from distance label ranges.

        Args:
            d_out: (K, V) forward distances from landmarks.
            d_in: (K, V) backward distances to landmarks.
        """
        d_out_t = d_out.t()  # (V, K)
        d_in_t = d_in.t()    # (V, K)
        self.comp_fwd.set_sigma_from_data(d_out_t)
        self.comp_bwd.set_sigma_from_data(d_in_t)

    def forward(
        self, d_out_t: torch.Tensor, d_in_t: torch.Tensor, hard: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress forward and backward labels separately.

        Args:
            d_out_t: (V, K) forward distances d(landmark, v).
            d_in_t: (V, K) backward distances d(v, landmark).
            hard: If True, use hard max (inference).

        Returns:
            (y_fwd, y_bwd): each (V, m_half) compressed labels.
        """
        y_fwd = self.comp_fwd(d_out_t, hard=hard)  # (V, m_fwd)
        y_bwd = self.comp_bwd(d_in_t, hard=hard)    # (V, m_bwd)
        return y_fwd, y_bwd

    def condition_regularization(self) -> torch.Tensor:
        return self.comp_fwd.condition_regularization() + self.comp_bwd.condition_regularization()

    def condition_number(self) -> float:
        return max(self.comp_fwd.condition_number(), self.comp_bwd.condition_number())


def softplus_inv(x: float) -> float:
    """Inverse of softplus: log(exp(x) - 1).

    For x > 20, softplus(x) ~ x, so softplus_inv(x) ~ x.
    This avoids numerical overflow in exp(x).

    Args:
        x: Positive float value to invert.

    Returns:
        y such that softplus(y) = x.
    """
    if x <= 0.0:
        raise ValueError(f"softplus_inv requires x > 0, got {x}")
    if x > 20.0:
        return x
    return math.log(math.exp(x) - 1.0)


class PositiveCompressor(nn.Module):
    """Max-plus compression for label vectors (log-domain contraction).

    Compresses embedding phi from (V, 2K) to (V, m) via:
        Training (smooth):  y_i(v) = sigma * logsumexp_j(alpha[i,j] + phi_j(v)/sigma)
        Inference (hard):   y_i(v) = max_j(alpha[i,j] + phi_j(v))

    where alpha is an unconstrained real parameter matrix.

    Max-plus contraction guarantee: for any real matrix alpha,
        max_i(y_i(u) - y_i(t)) <= max_j(phi_j(u) - phi_j(t)) = h_teacher(u,t)

    This means compressed heuristics are always admissible whenever the teacher
    heuristic is admissible -- no positivity constraint needed.

    During training, the smooth logsumexp approximation with sigma scaling
    provides gradient flow. At inference, the hard max gives the tightest
    admissible heuristic.

    Args:
        input_dim: Dimension of input label vectors (2K).
        compressed_dim: Target compression dimension (m).
        sigma: Scale parameter for smooth training. Auto-set from data if None.
    """

    def __init__(
        self,
        input_dim: int,
        compressed_dim: int,
        sigma: float | None = None,
    ) -> None:
        super().__init__()
        # Block-sparse initialization: each row specializes in a contiguous
        # block of input dimensions. Row i selects its assigned block
        # (alpha ≈ 0 for assigned dims, alpha << 0 for others).
        alpha_init = torch.full((compressed_dim, input_dim), -10.0)

        # Assign each row a block of input dimensions
        stride = max(1, input_dim // compressed_dim)
        for i in range(compressed_dim):
            j_start = i * stride
            j_end = min(j_start + stride, input_dim)
            alpha_init[i, j_start:j_end] = 0.0

        # Small noise for symmetry breaking within blocks
        alpha_init = alpha_init + 0.1 * torch.randn(compressed_dim, input_dim)
        self.alpha = nn.Parameter(alpha_init)
        # Scale parameter: fixed (not learned), set from data
        self._sigma: float | None = sigma

    @property
    def sigma(self) -> float:
        """Scale parameter for rescaled logsumexp."""
        if self._sigma is None:
            return 1.0
        return self._sigma

    def set_sigma_from_data(self, phi: torch.Tensor) -> None:
        """Auto-set sigma so phi/sigma has a reasonable range for logsumexp.

        Target: max |phi/sigma| ~ 10.

        Args:
            phi: (V, 2K) label vectors to calibrate from.
        """
        with torch.no_grad():
            phi_range = phi.max() - phi.min()
            if phi_range > 20.0:
                self._sigma = float(phi_range.item()) / 20.0

    @property
    def A(self) -> torch.Tensor:
        """Effective weight matrix exp(alpha) (for monitoring only)."""
        return torch.exp(self.alpha)

    @property
    def log_A(self) -> torch.Tensor:
        """Log of effective weight matrix (= alpha)."""
        return self.alpha

    def forward(self, phi: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Compress label vectors.

        Args:
            phi: (..., 2K) label vectors.
            hard: If True, use hard max (inference). If False, smooth logsumexp (training).

        Returns:
            (..., m) compressed label vectors.
        """
        # z[..., i, j] = alpha[i,j] + phi[..., j]
        z = phi.unsqueeze(-2) + self.alpha  # (..., m, 2K)

        if hard:
            return torch.max(z, dim=-1).values  # (..., m)
        else:
            s = self.sigma
            # Auto-set sigma if values would overflow fp64 in exp(z/s)
            if self._sigma is None and z.abs().max().item() > 500:
                self.set_sigma_from_data(phi)
                s = self.sigma
            return s * torch.logsumexp(z / s, dim=-1)  # (..., m)

    def condition_regularization(self) -> torch.Tensor:
        """Regularize alpha spread to prevent degenerate solutions."""
        return (self.alpha.max() - self.alpha.min()) ** 2

    def condition_number(self) -> float:
        """Monitor effective condition number exp(max(alpha) - min(alpha))."""
        return torch.exp(self.alpha.max() - self.alpha.min()).item()
