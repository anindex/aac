"""Neural edge cost encoders for Contextual.

WarcraftCNN: 3-conv-layer baseline encoder for RGB terrain images.
WarcraftResNet: higher-capacity encoder (5x5 stem + BN + 3 residual blocks
                with 64 channels each) used for the Appendix B encoder-capacity
                ablation in the paper.
CabspottingMLP: Predicts per-edge travel times from edge features.

All encoders produce strictly positive outputs via softplus + epsilon,
matching the PositiveCompressor pattern.

Also provides cell_costs_to_edge_weights for converting CNN cell-cost
predictions to edge weights matching the warcraft averaging convention:
    weight = 0.5 * (cost[src] + cost[tgt]) * distance_factor
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


class WarcraftCNN(nn.Module):
    """Small CNN for predicting per-cell terrain costs from RGB patches.

    Input: (B, 3, img_size, img_size) terrain image (e.g., 96x96 for 12x12 grid)
    Output: (B, grid_size * grid_size) positive cost per cell

    Architecture: 4 conv layers with ReLU, AdaptiveAvgPool2d to grid resolution,
    1x1 conv to single channel, softplus + epsilon for positivity.

    Args:
        grid_size: Size of the grid graph (H = W = grid_size).
        img_size: Input image size (H_img = W_img = img_size).
    """

    def __init__(self, grid_size: int = 12, img_size: int = 96) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict positive terrain costs from RGB image.

        Args:
            x: (B, 3, H_img, W_img) input image tensor.

        Returns:
            (B, grid_size * grid_size) strictly positive cell costs.
        """
        raw = self.encoder(x).squeeze(1)  # (B, H, W)
        positive = F.softplus(raw) + 0.01  # strictly positive
        return positive.view(positive.shape[0], -1)  # (B, H*W)


class _ResidualBlock(nn.Module):
    """A standard ResNet basic block: two 3x3 convolutions with batch norm
    and a skip connection. All channels are equal (no down-sampling)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class WarcraftResNet(nn.Module):
    """Higher-capacity ResNet encoder for Warcraft terrain images.

    Architecture (matches the paper, Appendix B):
        - 5x5 conv stem with batch normalization (3 -> 64 channels).
        - Three residual blocks, 64 channels each.
        - AdaptiveAvgPool to grid resolution.
        - Two 1x1 convolutions (64 -> 64 -> 1) with ReLU between.
        - softplus + 0.01 for strict positivity.

    Input:  (B, 3, img_size, img_size).
    Output: (B, grid_size * grid_size) positive cell costs.

    Args:
        grid_size: Side length of the grid graph (H = W = grid_size).
        img_size:  Side length of the input image (H_img = W_img = img_size).
    """

    def __init__(self, grid_size: int = 12, img_size: int = 96) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            _ResidualBlock(64),
            _ResidualBlock(64),
            _ResidualBlock(64),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x).squeeze(1)
        positive = F.softplus(x) + 0.01
        return positive.view(positive.shape[0], -1)


class CabspottingMLP(nn.Module):
    """MLP for predicting edge travel times from edge features.

    Input: (B, E, input_dim) or (..., input_dim) edge feature tensor
    Output: (..., E) or (...) strictly positive edge costs

    Architecture: 3 hidden layers with ReLU, linear output,
    softplus + epsilon for positivity.

    Args:
        input_dim: Number of input features per edge.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict positive edge costs from features.

        Args:
            features: (..., input_dim) edge feature tensor.

        Returns:
            (...) strictly positive edge costs (last dim squeezed).
        """
        raw = self.mlp(features).squeeze(-1)
        return F.softplus(raw) + 0.01  # strictly positive


# 8-connected neighbor offsets: (drow, dcol) matching warcraft.py _NEIGHBORS
_NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

_SQRT2 = math.sqrt(2.0)


@lru_cache(maxsize=16)
def build_grid_edge_index(
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute edge indices for an 8-connected grid graph.

    Emits BOTH directions for each edge pair (directed graph).

    Args:
        grid_size: Grid dimension (H = W = grid_size).

    Returns:
        (sources, targets, dist_factors) where:
            sources: (num_edges,) int64 source vertex indices
            targets: (num_edges,) int64 target vertex indices
            dist_factors: (num_edges,) float64 distance factors
                          (1.0 for cardinal, sqrt(2) for diagonal)
    """
    sources: list[int] = []
    targets: list[int] = []
    dist_factors: list[float] = []

    for r in range(grid_size):
        for c in range(grid_size):
            src_id = r * grid_size + c
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    tgt_id = nr * grid_size + nc
                    is_diagonal = dr != 0 and dc != 0
                    factor = _SQRT2 if is_diagonal else 1.0
                    sources.append(src_id)
                    targets.append(tgt_id)
                    dist_factors.append(factor)

    return (
        torch.tensor(sources, dtype=torch.int64),
        torch.tensor(targets, dtype=torch.int64),
        torch.tensor(dist_factors, dtype=torch.float64),
    )


def cell_costs_to_edge_weights(
    cell_costs: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Convert per-cell costs to per-edge weights for the grid graph.

    Uses the warcraft averaging convention:
        weight = 0.5 * (cost[src_cell] + cost[tgt_cell]) * distance_factor

    Fully vectorized using precomputed neighbor index tensors (no Python loops
    over edges). Differentiable with respect to cell_costs.

    Args:
        cell_costs: (B, H, W) predicted cell costs from CNN.
        grid_size: Grid dimension (H = W = grid_size).

    Returns:
        (B, num_edges) edge weight tensor, strictly positive if cell_costs > 0.
    """
    sources, targets, dist_factors = build_grid_edge_index(grid_size)

    # Move to same device as cell_costs
    device = cell_costs.device
    sources = sources.to(device)
    targets = targets.to(device)
    dist_factors = dist_factors.to(device=device, dtype=cell_costs.dtype)

    # Flatten cell_costs from (B, H, W) to (B, H*W)
    B = cell_costs.shape[0]
    flat_costs = cell_costs.view(B, -1)  # (B, H*W)

    # Gather source and target costs
    src_costs = flat_costs[:, sources]  # (B, num_edges)
    tgt_costs = flat_costs[:, targets]  # (B, num_edges)

    # Compute edge weights: 0.5 * (src + tgt) * dist_factor
    edge_weights = 0.5 * (src_costs + tgt_costs) * dist_factors.unsqueeze(0)

    return edge_weights
