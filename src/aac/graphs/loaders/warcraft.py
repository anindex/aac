"""Warcraft terrain map loader for differentiable routing experiments.

Supports two data formats:

1. Synthetic .npz files: single "cost_map" key with 2D float array.
   Used for development/testing only.

2. Pogancic et al. (ICLR 2020) .npy dataset: real Warcraft terrain with
   pre-split train/val/test arrays. Each sample has:
   - RGB terrain image (96x96x3 uint8)
   - Vertex cost grid (HxW float16, 5 terrain types)
   - Binary shortest path mask (HxW uint8)

Edge weight from cell (r1,c1) to (r2,c2) = 0.5 * (cost[r1,c1] + cost[r2,c2]) * dist_factor.
For diagonal moves, dist_factor = sqrt(2).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from aac.graphs.convert import edges_to_graph
from aac.graphs.types import Graph

# 8-connected neighbor offsets: (drow, dcol)
_NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

_SQRT2 = math.sqrt(2)


def load_warcraft(
    npz_path: str | Path,
    cost_map_key: str = "cost_map",
) -> tuple[Graph, torch.Tensor]:
    """Load a Warcraft .npz cost map and build an 8-connected grid graph.

    Args:
        npz_path: Path to the .npz file containing the cost map.
        cost_map_key: Key for the cost map array in the .npz file.

    Returns:
        Tuple of (Graph, cost_map_tensor) where:
        - Graph is undirected with 8-connected edges and fp64 weights
        - cost_map_tensor is the raw (H, W) terrain cost tensor for visualization

    Raises:
        ValueError: If the expected key is not found in the .npz file.
    """
    data = np.load(npz_path)

    if cost_map_key not in data:
        available = list(data.keys())
        raise ValueError(
            f"Expected key '{cost_map_key}' not found in .npz file. "
            f"Available keys: {available}"
        )

    cost_map = data[cost_map_key].astype(np.float64)
    if cost_map.ndim != 2:
        raise ValueError(f"Cost map should be 2D, got shape {cost_map.shape}")

    H, W = cost_map.shape
    num_nodes = H * W

    # Build node coordinates: (col, row) for each cell
    coords: list[tuple[float, float]] = []
    for r in range(H):
        for c in range(W):
            coords.append((float(c), float(r)))

    # Build edges: 8-connected grid
    # Edge weight = cost_map[target_row, target_col] * distance_factor
    # For cardinal moves: distance_factor = 1.0
    # For diagonal moves: distance_factor = sqrt(2)
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []

    for r in range(H):
        for c in range(W):
            src_id = r * W + c
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    tgt_id = nr * W + nc
                    # Only emit one direction per pair for undirected
                    if src_id < tgt_id:
                        is_diagonal = dr != 0 and dc != 0
                        distance_factor = _SQRT2 if is_diagonal else 1.0
                        # Average cost of source and target cells times distance
                        weight = 0.5 * (cost_map[r, c] + cost_map[nr, nc]) * distance_factor
                        sources.append(src_id)
                        targets.append(tgt_id)
                        weights.append(weight)

    src_t = torch.tensor(sources, dtype=torch.int64)
    tgt_t = torch.tensor(targets, dtype=torch.int64)
    wgt_t = torch.tensor(weights, dtype=torch.float64)
    coord_t = torch.tensor(coords, dtype=torch.float64)
    cost_tensor = torch.tensor(cost_map, dtype=torch.float64)

    graph = edges_to_graph(
        sources=src_t,
        targets=tgt_t,
        weights=wgt_t,
        num_nodes=num_nodes,
        is_directed=False,
        coordinates=coord_t,
    )

    return graph, cost_tensor


def build_warcraft_graph(cost_map: np.ndarray) -> tuple[Graph, torch.Tensor]:
    """Build an 8-connected grid graph from a 2D vertex cost array.

    Same edge weight convention as load_warcraft:
        weight = 0.5 * (cost[src] + cost[tgt]) * distance_factor

    Args:
        cost_map: (H, W) float array of vertex costs.

    Returns:
        Tuple of (Graph, cost_map_tensor).
    """
    cost_map = cost_map.astype(np.float64)
    H, W = cost_map.shape
    num_nodes = H * W

    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []

    for r in range(H):
        for c in range(W):
            src_id = r * W + c
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    tgt_id = nr * W + nc
                    if src_id < tgt_id:
                        is_diagonal = dr != 0 and dc != 0
                        distance_factor = _SQRT2 if is_diagonal else 1.0
                        weight = 0.5 * (cost_map[r, c] + cost_map[nr, nc]) * distance_factor
                        sources.append(src_id)
                        targets.append(tgt_id)
                        weights.append(weight)

    graph = edges_to_graph(
        sources=torch.tensor(sources, dtype=torch.int64),
        targets=torch.tensor(targets, dtype=torch.int64),
        weights=torch.tensor(weights, dtype=torch.float64),
        num_nodes=num_nodes,
        is_directed=False,
    )
    return graph, torch.tensor(cost_map, dtype=torch.float64)


def load_warcraft_dataset(
    data_dir: str | Path,
    grid_size: int = 12,
) -> dict[str, dict[str, np.ndarray]]:
    """Load the Pogancic et al. (ICLR 2020) Warcraft shortest path dataset.

    Expects directory structure:
        data_dir/{grid_size}x{grid_size}/train_maps.npy
        data_dir/{grid_size}x{grid_size}/train_vertex_weights.npy
        data_dir/{grid_size}x{grid_size}/train_shortest_paths.npy
        (similarly for val_ and test_ prefixes)

    Args:
        data_dir: Path to dataset root (e.g., data/warcraft_real/warcraft_shortest_path_oneskin).
        grid_size: Grid dimension (12, 18, 24, or 30).

    Returns:
        Dict with keys 'train', 'val', 'test', each containing:
            'maps': (N, 96, 96, 3) uint8 RGB terrain images
            'weights': (N, grid_size, grid_size) float64 vertex costs
            'paths': (N, grid_size, grid_size) uint8 binary GT path masks

    Raises:
        FileNotFoundError: If required data files are missing.
    """
    base = Path(data_dir) / f"{grid_size}x{grid_size}"

    result = {}
    for split in ("train", "val", "test"):
        maps_path = base / f"{split}_maps.npy"
        weights_path = base / f"{split}_vertex_weights.npy"
        paths_path = base / f"{split}_shortest_paths.npy"

        if not maps_path.exists():
            raise FileNotFoundError(
                f"Missing {maps_path}. Download the Warcraft shortest path "
                "dataset from https://github.com/martius-lab/blackbox-backprop"
            )

        result[split] = {
            "maps": np.load(str(maps_path)),
            "weights": np.load(str(weights_path)).astype(np.float64),
            "paths": np.load(str(paths_path)),
        }

    return result
