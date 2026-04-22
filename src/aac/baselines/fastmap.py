"""FastMap baseline: iterative farthest-pair Euclidean embedding + L1 heuristic.

FastMap embeds graph vertices into m-dimensional Euclidean space by iteratively
selecting farthest pairs and projecting onto the connecting axis, subtracting
already-assigned coordinate contributions (residual distance).

The L1 (Manhattan) distance in the embedding is an admissible heuristic.
L2 (Euclidean) is NOT admissible per AAAI 2023 analysis.

References:
    Faloutsos & Lin (1995). FastMap: A Fast Algorithm for Indexing, Data-Mining
    and Visualization of Traditional and Multimedia Datasets. SIGMOD.
    Cohen et al. (2018). FastMap, FastEmbedding and Scalable Heuristic Search. IJCAI.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse.csgraph
import torch

from aac.graphs.convert import graph_to_scipy
from aac.graphs.types import Graph


def fastmap_preprocess(graph: Graph, num_dims: int) -> torch.Tensor:
    """Compute FastMap embedding coordinates for all vertices.

    Iteratively finds farthest pairs and projects vertices onto the
    connecting axis, subtracting already-assigned coordinate contributions.

    WARNING: FastMap assumes undirected (symmetric) distances. For directed
    graphs, the L1 heuristic is NOT guaranteed to be admissible.

    Args:
        graph: Input graph in CSR format.
        num_dims: Number of embedding dimensions (m).

    Returns:
        (V, num_dims) fp64 tensor of vertex coordinates.
    """
    import warnings

    if graph.is_directed:
        warnings.warn(
            "FastMap is designed for undirected graphs. "
            "The L1 heuristic may NOT be admissible for directed graphs.",
            stacklevel=2,
        )

    V = graph.num_nodes
    csr = graph_to_scipy(graph)
    coords = np.zeros((V, num_dims), dtype=np.float64)

    for k in range(num_dims):
        # Step 1: Pick arbitrary start vertex and find farthest vertex a
        start = 0
        d_start = scipy.sparse.csgraph.dijkstra(csr, indices=start)
        d_start = np.where(np.isinf(d_start), 1e18, d_start)
        a = int(np.argmax(d_start))

        # Step 2: Run dijkstra from a to find farthest vertex b
        d_a = scipy.sparse.csgraph.dijkstra(csr, indices=a)
        d_a = np.where(np.isinf(d_a), 1e18, d_a)
        b = int(np.argmax(d_a))

        # Step 3: Run dijkstra from b
        d_b = scipy.sparse.csgraph.dijkstra(csr, indices=b)
        d_b = np.where(np.isinf(d_b), 1e18, d_b)

        # Step 4: Compute d(a,b) for this dimension
        d_ab_sq = d_a[b] ** 2

        if k > 0:
            # Subtract already-assigned coordinate contributions (residual)
            coord_diff_a = coords[:, :k] - coords[a, :k]  # (V, k)
            coord_diff_b = coords[:, :k] - coords[b, :k]  # (V, k)
            coord_diff_ab = coords[a, :k] - coords[b, :k]  # (k,)

            d_a_res_sq = np.maximum(d_a ** 2 - np.sum(coord_diff_a ** 2, axis=1), 0.0)
            d_b_res_sq = np.maximum(d_b ** 2 - np.sum(coord_diff_b ** 2, axis=1), 0.0)
            d_ab_res_sq = max(d_ab_sq - np.sum(coord_diff_ab ** 2), 0.0)
        else:
            d_a_res_sq = d_a ** 2
            d_b_res_sq = d_b ** 2
            d_ab_res_sq = d_ab_sq

        # Clamp to avoid negative sqrt
        d_ab_res = max(np.sqrt(max(d_ab_res_sq, 0.0)), 1e-10)

        # Check if graph is fully embedded
        if d_a[b] < 1e-10:
            break

        # Project: coords[:, k] = (d_a_res_sq + d_ab_res_sq - d_b_res_sq) / (2 * d_ab_res)
        coords[:, k] = (d_a_res_sq + d_ab_res_sq - d_b_res_sq) / (2.0 * d_ab_res)

    return torch.tensor(coords, dtype=torch.float64)


def make_fastmap_heuristic(coords: torch.Tensor) -> Callable[[int, int], float]:
    """Create A*-compatible heuristic from FastMap coordinates.

    Uses L1 (Manhattan) distance, NOT L2 -- L2 is NOT admissible
    per AAAI 2023 analysis.

    Args:
        coords: (V, num_dims) fp64 tensor of vertex coordinates.

    Returns:
        Callable h(node: int, target: int) -> float.
    """

    def h(node: int, target: int) -> float:
        return torch.sum(torch.abs(coords[node] - coords[target])).item()

    return h


def fastmap_memory_bytes(num_dims: int, dtype_size: int = 4) -> int:
    """Compute per-vertex deployed memory for FastMap.

    FastMap stores m coordinate values per vertex.

    Args:
        num_dims: Number of embedding dimensions (m).
        dtype_size: Bytes per element. Default 4 (float32) per METR-05.

    Returns:
        Per-vertex memory in bytes: m * dtype_size.
    """
    return num_dims * dtype_size
