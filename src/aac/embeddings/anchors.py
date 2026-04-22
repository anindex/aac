"""Anchor selection strategies for landmark-based heuristics.

Provides four strategies for selecting K anchor vertices:
- Farthest-point sampling (FPS): greedy maximin coverage
- Random: uniform without replacement
- Boundary: coordinate extremes + FPS fill
- Planar partition: grid-based spatial partitioning
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.csgraph
import torch

from aac.graphs.convert import graph_to_scipy
from aac.graphs.types import Graph


def farthest_point_sampling(
    graph: Graph,
    num_anchors: int,
    seed_vertex: int | None = None,
    rng: torch.Generator | None = None,
    valid_vertices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy farthest-point sampling for anchor selection.

    Start with a seed vertex, iteratively add the vertex that is farthest
    (maximum min-distance) from the current anchor set. Uses SciPy Dijkstra
    for accuracy.

    Args:
        graph: Input graph in CSR format.
        num_anchors: Number of anchors to select (K).
        seed_vertex: Starting vertex index. Random if None.
        rng: Optional torch generator for reproducible seed selection.
        valid_vertices: Optional (N,) int64 tensor restricting anchor selection
            to a subset of vertices (e.g., largest connected component).
            Vertices outside this set are excluded from selection.

    Returns:
        (num_anchors,) int64 tensor of anchor vertex indices.
    """
    V = graph.num_nodes

    # Build mask for valid vertices
    if valid_vertices is not None:
        valid_mask = np.zeros(V, dtype=bool)
        valid_mask[valid_vertices.numpy()] = True
        num_valid = int(valid_mask.sum())
        assert num_anchors <= num_valid, (
            f"Cannot select {num_anchors} anchors from {num_valid} valid nodes"
        )
    else:
        valid_mask = np.ones(V, dtype=bool)
        assert num_anchors <= V, f"Cannot select {num_anchors} anchors from {V} nodes"

    if seed_vertex is None:
        if valid_vertices is not None:
            idx = torch.randint(valid_vertices.shape[0], (1,), generator=rng).item()
            seed_vertex = int(valid_vertices[idx].item())
        else:
            seed_vertex = torch.randint(V, (1,), generator=rng).item()

    scipy_csr = graph_to_scipy(graph)
    anchors = [seed_vertex]

    # Initial distances from seed vertex
    min_dist = scipy.sparse.csgraph.dijkstra(scipy_csr, indices=seed_vertex)
    # Replace np.inf with large value to avoid issues with argmax
    min_dist = np.where(np.isinf(min_dist), 1e18, min_dist)
    # Mask out invalid vertices so they are never selected
    min_dist[~valid_mask] = -1.0

    for _ in range(num_anchors - 1):
        # Mask already-selected anchors so they can't be re-selected
        for a in anchors:
            min_dist[a] = -1.0

        # Select vertex with maximum min-distance to existing anchors
        next_anchor = int(np.argmax(min_dist))
        anchors.append(next_anchor)

        # Update min distances with new anchor
        new_dist = scipy.sparse.csgraph.dijkstra(scipy_csr, indices=next_anchor)
        new_dist = np.where(np.isinf(new_dist), 1e18, new_dist)
        min_dist = np.minimum(min_dist, new_dist)
        # Maintain mask on invalid vertices
        min_dist[~valid_mask] = -1.0

    return torch.tensor(anchors, dtype=torch.int64)


def random_anchors(
    num_nodes: int,
    num_anchors: int,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniform random anchor selection without replacement.

    Args:
        num_nodes: Total number of vertices V.
        num_anchors: Number of anchors to select (K).
        rng: Optional torch generator for reproducibility.

    Returns:
        (num_anchors,) int64 tensor of anchor vertex indices.
    """
    assert num_anchors <= num_nodes, f"Cannot select {num_anchors} anchors from {num_nodes} nodes"

    # Use torch.randperm for sampling without replacement
    perm = torch.randperm(num_nodes, generator=rng)
    return perm[:num_anchors].to(torch.int64)


def boundary_anchors(
    graph: Graph,
    num_anchors: int,
) -> torch.Tensor:
    """Select anchors at coordinate extremes, then fill with FPS.

    Selects vertices at min-x, max-x, min-y, max-y coordinate positions.
    If num_anchors > 4, fills remaining anchors using farthest-point sampling
    from the boundary set.

    Args:
        graph: Graph with coordinates (graph.coordinates must not be None).
        num_anchors: Number of anchors to select (K >= 4).

    Returns:
        (num_anchors,) int64 tensor of anchor vertex indices.

    Raises:
        AssertionError: If graph has no coordinates.
    """
    assert graph.coordinates is not None, "boundary_anchors requires graph.coordinates"
    coords = graph.coordinates  # (V, 2)

    # Find extremes
    min_x_idx = int(torch.argmin(coords[:, 0]).item())
    max_x_idx = int(torch.argmax(coords[:, 0]).item())
    min_y_idx = int(torch.argmin(coords[:, 1]).item())
    max_y_idx = int(torch.argmax(coords[:, 1]).item())

    # Collect unique boundary vertices
    boundary = []
    seen = set()
    for idx in [min_x_idx, max_x_idx, min_y_idx, max_y_idx]:
        if idx not in seen:
            boundary.append(idx)
            seen.add(idx)

    if len(boundary) >= num_anchors:
        return torch.tensor(boundary[:num_anchors], dtype=torch.int64)

    # Fill remaining with FPS starting from boundary set
    scipy_csr = graph_to_scipy(graph)
    V = graph.num_nodes

    # Initialize min_dist from all boundary vertices
    min_dist = np.full(V, 1e18)
    for b_idx in boundary:
        d = scipy.sparse.csgraph.dijkstra(scipy_csr, indices=b_idx)
        d = np.where(np.isinf(d), 1e18, d)
        min_dist = np.minimum(min_dist, d)

    # Greedily add farthest vertices
    while len(boundary) < num_anchors:
        # Mask already-selected anchors
        for b_idx in boundary:
            min_dist[b_idx] = -1.0
        next_anchor = int(np.argmax(min_dist))
        boundary.append(next_anchor)

        new_dist = scipy.sparse.csgraph.dijkstra(scipy_csr, indices=next_anchor)
        new_dist = np.where(np.isinf(new_dist), 1e18, new_dist)
        min_dist = np.minimum(min_dist, new_dist)
        # Re-mask selected anchors after np.minimum (new_dist may have overwritten them)
        for b_idx in boundary:
            min_dist[b_idx] = -1.0

    return torch.tensor(boundary, dtype=torch.int64)


def planar_partition_anchors(
    graph: Graph,
    num_anchors: int,
) -> torch.Tensor:
    """Select anchors by spatial grid partitioning.

    Divides the coordinate space into sqrt(num_anchors) x sqrt(num_anchors)
    sectors and selects the vertex closest to each sector center.

    Args:
        graph: Graph with coordinates (graph.coordinates must not be None).
        num_anchors: Number of anchors to select (K).

    Returns:
        (num_anchors,) int64 tensor of anchor vertex indices. May return
        fewer than num_anchors if some sectors are empty.

    Raises:
        AssertionError: If graph has no coordinates.
    """
    assert graph.coordinates is not None, "planar_partition_anchors requires graph.coordinates"
    coords = graph.coordinates.cpu().numpy()  # (V, 2)

    # Grid dimensions
    grid_side = max(1, int(np.sqrt(num_anchors)))
    # Ensure we have at least num_anchors cells
    grid_x = grid_side
    grid_y = (num_anchors + grid_x - 1) // grid_x

    # Bounding box with small epsilon to avoid edge cases
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    eps = 1e-10
    x_range = max(x_max - x_min, eps)
    y_range = max(y_max - y_min, eps)

    # Compute sector centers
    anchors = []
    seen = set()
    for ix in range(grid_x):
        for iy in range(grid_y):
            if len(anchors) >= num_anchors:
                break
            cx = x_min + (ix + 0.5) * x_range / grid_x
            cy = y_min + (iy + 0.5) * y_range / grid_y
            # Find closest vertex to sector center
            dists = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2
            closest = int(np.argmin(dists))
            if closest not in seen:
                anchors.append(closest)
                seen.add(closest)

    return torch.tensor(anchors, dtype=torch.int64)
