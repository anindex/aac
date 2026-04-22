"""1-Lipschitz neural compression for AAC.

Key idea: Landmark distances are 1-Lipschitz functions of the graph metric.
The space of 1-Lipschitz functions R^K -> R^m is much richer than the space
of landmark subsets (one-hot selection). A neural network constrained to be
1-Lipschitz (L_inf -> L_inf) can discover non-linear combinations that are
tighter than any single landmark selection.

Admissibility proof:
    If f: R^K -> R^m is 1-Lipschitz w.r.t. L_inf norm, then:
    h(u,t) = ||f(d(u)) - f(d(t))||_inf
           <= ||d(u) - d(t)||_inf               (1-Lipschitz)
           = max_k |d_k(u) - d_k(t)|            (definition of L_inf)
           = h_ALT(K)                            (ALT heuristic with K landmarks)
           <= d(u,t)                             (triangle inequality)

Architecture: GroupSort activations (Anil et al., 2019) with inf-norm spectral
normalization on weight matrices. GroupSort preserves the Lipschitz constant
while being a universal approximator in the 1-Lipschitz function class.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from aac.utils.numerics import SENTINEL


class GroupSort(nn.Module):
    """GroupSort activation: sort pairs of neurons.

    Splits input into groups of 2, sorts each group in descending order.
    This is a 1-Lipschitz activation w.r.t. any p-norm (Anil et al., 2019).

    For odd dimensions, the last element passes through unchanged.
    """

    def __init__(self, group_size: int = 2) -> None:
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch_shape, d_orig = x.shape
        g = self.group_size
        d = d_orig
        if d % g != 0:
            pad = g - (d % g)
            x = torch.nn.functional.pad(x, (0, pad), value=float("-inf"))
            d = d + pad
        x = x.view(*batch_shape, d // g, g)
        x = x.sort(dim=-1, descending=True).values
        x = x.view(*batch_shape, d)
        # Remove padding if added
        if d != d_orig:
            x = x[..., :d_orig]
        return x


def _spectral_norm_inf(W: torch.Tensor, n_iters: int = 3) -> torch.Tensor:
    """Compute the infinity-norm of a matrix: max row absolute sum.

    ||W||_inf = max_i sum_j |W_ij|

    This is the operator norm for L_inf -> L_inf mappings.

    Args:
        W: (out_features, in_features) weight matrix.
        n_iters: Unused (closed-form for inf-norm). Kept for API compat.

    Returns:
        Scalar tensor: the infinity-norm of W.
    """
    return W.abs().sum(dim=-1).max()


class LipschitzLinear(nn.Module):
    """Linear layer with L_inf spectral normalization.

    Ensures ||W||_inf <= 1 by dividing by max(1, ||W||_inf) at forward time.
    Combined with GroupSort activation, this creates a 1-Lipschitz block.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Initialize with small weights to start near identity-like
        nn.init.orthogonal_(self.weight)
        # Scale down so ||W||_inf <= 1 from the start
        with torch.no_grad():
            norm = _spectral_norm_inf(self.weight)
            if norm > 1.0:
                self.weight.div_(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight
        # Normalize: W_norm = W / max(1, ||W||_inf)
        norm = _spectral_norm_inf(W)
        W_norm = W / torch.clamp(norm, min=1.0)
        return torch.nn.functional.linear(x, W_norm, self.bias)

    def get_norm(self) -> float:
        """Return current infinity-norm of the weight matrix."""
        return _spectral_norm_inf(self.weight).item()


class LipschitzCompressor(nn.Module):
    """1-Lipschitz neural compression for landmark distances.

    Compresses K landmark distances to m dimensions using a neural network
    f: R^K -> R^m that is constrained to be 1-Lipschitz w.r.t. L_inf norm.

    For directed graphs:
        f_fwd: R^K -> R^{m_fwd}  compresses forward distances
        f_bwd: R^K -> R^{m_bwd}  compresses backward distances

    Heuristic evaluation (directed):
        h(u,t) = max(0,
            max_i(f_bwd(d_in[u])_i - f_bwd(d_in[t])_i),
            max_i(f_fwd(d_out[t])_i - f_fwd(d_out[u])_i)
        )

    This is admissible because:
        |f(d(u))_i - f(d(t))_i| <= ||f(d(u)) - f(d(t))||_inf
        <= ||d(u) - d(t)||_inf = max_k |d_k(u) - d_k(t)| <= d(u,t)

    Args:
        K: Number of teacher landmarks (input dimension).
        m: Total compressed dimensions.
        hidden_dim: Hidden layer width.
        n_layers: Number of LipschitzLinear + GroupSort blocks.
        is_directed: Whether the graph is directed.
    """

    def __init__(
        self,
        K: int,
        m: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        is_directed: bool = True,
    ) -> None:
        super().__init__()
        self.K = K
        self.m = m
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.is_directed = is_directed

        if is_directed:
            self.m_fwd = m // 2
            self.m_bwd = m - self.m_fwd
            self.net_fwd = self._build_net(K, self.m_fwd, hidden_dim, n_layers)
            self.net_bwd = self._build_net(K, self.m_bwd, hidden_dim, n_layers)
        else:
            self.net = self._build_net(K, m, hidden_dim, n_layers)

    @staticmethod
    def _build_net(in_dim: int, out_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
        """Build a 1-Lipschitz MLP: [LipschitzLinear -> GroupSort] x n -> LipschitzLinear."""
        layers: list[nn.Module] = []
        d_in = in_dim
        for _ in range(n_layers):
            # Ensure hidden dim is even for GroupSort
            h = hidden if hidden % 2 == 0 else hidden + 1
            layers.append(LipschitzLinear(d_in, h))
            layers.append(GroupSort(group_size=2))
            d_in = h
        layers.append(LipschitzLinear(d_in, out_dim))
        return nn.Sequential(*layers)

    def forward(
        self,
        d_out_t: torch.Tensor,
        d_in_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Compress distance labels via 1-Lipschitz network.

        Args:
            d_out_t: (V, K) or (B, K) forward distances.
            d_in_t: (V, K) or (B, K) backward distances (None for undirected).

        Returns:
            For directed: (y_fwd, y_bwd) each (V, m_half).
            For undirected: y (V, m).
        """
        if self.is_directed:
            y_fwd = self.net_fwd(d_out_t)   # (V, m_fwd)
            y_bwd = self.net_bwd(d_in_t)    # (V, m_bwd)
            return y_fwd, y_bwd
        else:
            return self.net(d_out_t)

    def condition_regularization(self) -> torch.Tensor:
        """Return 0 -- Lipschitz constraint replaces condition regularization."""
        return torch.tensor(0.0)

    def condition_number(self) -> float:
        """Return the product of layer norms (effective Lipschitz constant)."""
        total = 1.0
        nets = [self.net_fwd, self.net_bwd] if self.is_directed else [self.net]
        for net in nets:
            lip = 1.0
            for module in net:
                if isinstance(module, LipschitzLinear):
                    lip *= max(1.0, module.get_norm())
            total = max(total, lip)
        return total

    def uniqueness_penalty(self) -> torch.Tensor:
        """Return 0 -- not applicable for neural compression."""
        return torch.tensor(0.0)

    def lipschitz_penalty(self) -> torch.Tensor:
        """Penalize layers whose weight norm exceeds 1.

        This is a soft backup to the hard normalization in LipschitzLinear.forward().
        During training, gradients through the normalization can push weights
        above 1 between forward passes. This penalty provides an additional
        signal to keep norms small.
        """
        penalty = torch.tensor(0.0)
        nets = [self.net_fwd, self.net_bwd] if self.is_directed else [self.net]
        for net in nets:
            for module in net:
                if isinstance(module, LipschitzLinear):
                    norm = _spectral_norm_inf(module.weight)
                    # Penalize amount by which norm exceeds 1
                    penalty = penalty + torch.relu(norm - 1.0)
        return penalty


def make_lipschitz_heuristic(
    net_fwd: nn.Sequential,
    net_bwd: nn.Sequential | None,
    d_out_np: np.ndarray,
    d_in_np: np.ndarray | None,
    is_directed: bool,
) -> "Callable[[int, int], float]":
    """Create A*-compatible heuristic from 1-Lipschitz compressed labels.

    Precomputes all compressed labels y = f(d) for fast per-query evaluation.

    Args:
        net_fwd: Trained 1-Lipschitz network for forward distances.
        net_bwd: Trained 1-Lipschitz network for backward distances (directed only).
        d_out_np: (K, V) forward distances from landmarks (numpy).
        d_in_np: (K, V) backward distances to landmarks (numpy).
        is_directed: Whether the graph is directed.

    Returns:
        Callable h(node, target) -> float.
    """
    sentinel_thresh = 0.99 * SENTINEL

    # Mask sentinel distances before feeding to the neural network.
    # Unlike linear/ALT heuristics where sentinel coordinates can be masked
    # independently, the neural network mixes all K input dimensions via
    # weight matrices -- a single sentinel input (1e18) can contaminate ALL
    # output dimensions.  Replace sentinels with 0 (neutral) before the
    # forward pass, and track which vertices had sentinel inputs so the
    # heuristic closure can fall back to 0 (Dijkstra) for those vertices.
    sentinel_mask_out = d_out_np >= sentinel_thresh  # (K, V)
    d_out_safe = np.where(sentinel_mask_out, 0.0, d_out_np)
    any_sentinel_out = np.any(sentinel_mask_out, axis=0)  # (V,)

    if is_directed and d_in_np is not None:
        sentinel_mask_in = d_in_np >= sentinel_thresh  # (K, V)
        d_in_safe = np.where(sentinel_mask_in, 0.0, d_in_np)
        any_sentinel_in = np.any(sentinel_mask_in, axis=0)  # (V,)
    else:
        d_in_safe = d_out_safe
        any_sentinel_in = any_sentinel_out

    # Precompute compressed labels for all vertices
    with torch.no_grad():
        d_out_t = torch.from_numpy(d_out_safe.T).float()  # (V, K)
        y_fwd_t = net_fwd(d_out_t)
        y_fwd_np = y_fwd_t.cpu().numpy()  # (V, m_fwd)

        if is_directed and net_bwd is not None:
            d_in_t = torch.from_numpy(d_in_safe.T).float()    # (V, K)
            y_bwd_t = net_bwd(d_in_t)
            y_bwd_np = y_bwd_t.cpu().numpy()  # (V, m_bwd)
        else:
            y_bwd_np = y_fwd_np

    if is_directed:
        def h(node: int, target: int) -> float:
            if any_sentinel_out[node] or any_sentinel_out[target] or \
               any_sentinel_in[node] or any_sentinel_in[target]:
                return 0.0
            # Backward bound: y_bwd(u) - y_bwd(t)
            bwd_n = y_bwd_np[node]
            bwd_t = y_bwd_np[target]
            # Forward bound: y_fwd(t) - y_fwd(u)
            fwd_n = y_fwd_np[node]
            fwd_t = y_fwd_np[target]
            result = max(
                0.0,
                float(np.max(bwd_n - bwd_t)),
                float(np.max(fwd_t - fwd_n)),
            )
            return result
    else:
        def h(node: int, target: int) -> float:
            if any_sentinel_out[node] or any_sentinel_out[target]:
                return 0.0
            yn = y_fwd_np[node]
            yt = y_fwd_np[target]
            return max(0.0, float(np.max(np.abs(yn - yt))))

    return h
