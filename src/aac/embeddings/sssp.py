"""Batched single-source shortest path computation.

Provides GPU-capable Bellman-Ford (edge-parallel with scatter_reduce),
SciPy Dijkstra reference implementation, and optional NetworKit Dijkstra
backend for large-scale graphs. Supports chunked processing and float32
dtype to reduce peak memory for graphs with 5M+ nodes.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import torch

from aac.graphs.convert import edges_to_graph, graph_to_scipy
from aac.graphs.types import Graph, TeacherLabels
from aac.utils.memory import memory_guard

logger = logging.getLogger(__name__)


def _bf_relax_step(
    dist: torch.Tensor,
    row_idx: torch.Tensor,
    weights: torch.Tensor,
    col_expanded: torch.Tensor,
) -> torch.Tensor:
    """Single Bellman-Ford relaxation step (compiled for performance)."""
    source_dist = dist[:, row_idx]
    candidate = source_dist + weights.unsqueeze(0)
    new_dist = dist.clone()
    new_dist.scatter_reduce_(1, col_expanded, candidate, reduce="amin", include_self=True)
    return new_dist


def bellman_ford_batched(
    graph: Graph,
    source_indices: torch.Tensor,
    sentinel: float = 1e18,
    max_iter: int | None = None,
) -> torch.Tensor:
    """Memory-efficient batched Bellman-Ford using edge-parallel relaxation.

    Uses COO edge list and scatter_reduce_ for O(K*V + E) memory
    instead of O(K*V^2) for dense matmul approach.

    Args:
        graph: Input graph in CSR format.
        source_indices: (K,) int64 tensor of source vertex indices.
        sentinel: Value for unreachable nodes. Default 1e18.
        max_iter: Maximum iterations. Default V-1.

    Returns:
        (K, V) fp64 distance tensor. sentinel for unreachable nodes.
    """
    V = graph.num_nodes
    K = source_indices.shape[0]
    device = graph.values.device

    if max_iter is None:
        max_iter = V - 1

    # Get COO representation
    row_idx, col_idx, weights = graph.to_coo()
    E = row_idx.shape[0]

    # Initialize distances: 0 at sources, sentinel elsewhere
    dist = torch.full((K, V), sentinel, dtype=torch.float64, device=device)
    dist[torch.arange(K, device=device), source_indices.to(device)] = 0.0

    col_expanded = col_idx.unsqueeze(0).expand(K, E)

    for _iteration in range(max_iter):
        new_dist = _bf_relax_step(dist, row_idx, weights, col_expanded)

        # Check convergence
        if torch.equal(new_dist, dist):
            break
        dist = new_dist

    # Negative-weight cycle detection: one additional relaxation pass
    check_dist = _bf_relax_step(dist, row_idx, weights, col_expanded)
    if not torch.equal(check_dist, dist):
        import warnings
        warnings.warn(
            "Bellman-Ford detected a negative-weight cycle: distances may be incorrect",
            RuntimeWarning,
            stacklevel=2,
        )

    return dist


def scipy_dijkstra_batched(
    graph: Graph,
    source_indices: torch.Tensor,
    sentinel: float = 1e18,
) -> torch.Tensor:
    """Reference batched SSSP using SciPy Dijkstra.

    Args:
        graph: Input graph in CSR format.
        source_indices: (K,) int64 tensor of source vertex indices.
        sentinel: Value for unreachable nodes. Default 1e18.

    Returns:
        (K, V) fp64 distance tensor. sentinel for unreachable nodes.
    """
    scipy_csr = graph_to_scipy(graph)
    K = source_indices.shape[0]
    V = graph.num_nodes

    results = np.zeros((K, V), dtype=np.float64)
    for i, src in enumerate(source_indices.tolist()):
        d = scipy.sparse.csgraph.dijkstra(scipy_csr, indices=src)
        d = np.where(np.isinf(d), sentinel, d)
        results[i] = d

    return torch.tensor(results, dtype=torch.float64)


def networkit_dijkstra_batched(
    graph: Graph,
    source_indices: torch.Tensor,
    sentinel: float = 1e18,
) -> torch.Tensor:
    """Batched SSSP using NetworKit Dijkstra (C++ backend).

    Provides 3-10x speedup over SciPy Dijkstra on large road networks.
    Requires optional dependency: ``pip install aac[scalability]``.

    Args:
        graph: Input graph in CSR format.
        source_indices: (K,) int64 tensor of source vertex indices.
        sentinel: Value for unreachable nodes. Default 1e18.

    Returns:
        (K, V) fp64 distance tensor. sentinel for unreachable nodes.

    Raises:
        ImportError: If networkit is not installed.
    """
    try:
        import networkit as nk
    except ImportError:
        raise ImportError(
            "networkit is required for backend='networkit'. "
            "Install with: pip install aac[scalability]"
        )

    # Convert Graph to NetworKit via COO
    row, col, val = graph.to_coo()
    row_np = row.cpu().numpy().astype(np.uint64)
    col_np = col.cpu().numpy().astype(np.uint64)
    val_np = val.cpu().numpy().astype(np.float64)

    # Assert non-negative indices before uint64 cast (T-17-05)
    assert row.min() >= 0 and col.min() >= 0, (
        "Negative COO indices detected; cannot safely cast to uint64"
    )

    # Build NetworKit graph from COO arrays
    nk_graph = nk.GraphFromCoo(
        (val_np, (row_np, col_np)),
        n=graph.num_nodes,
        weighted=True,
        directed=graph.is_directed,
    )

    K = source_indices.shape[0]
    V = graph.num_nodes
    results = np.zeros((K, V), dtype=np.float64)

    for i, src in enumerate(source_indices.tolist()):
        dijkstra = nk.distance.Dijkstra(nk_graph, src, storePaths=False)
        dijkstra.run()
        dists = np.array(dijkstra.getDistances())
        # Remap NetworKit's double_max (~1.8e308) to project sentinel (T-17-05)
        dists[dists >= 1e300] = sentinel
        results[i] = dists

    return torch.tensor(results, dtype=torch.float64)


def _chunked_sssp(
    sssp_fn,
    graph: Graph,
    source_indices: torch.Tensor,
    chunk_size: int,
    sentinel: float = 1e18,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Process SSSP in chunks to reduce peak memory.

    Pre-allocates the full result tensor in float64, processes anchors
    in batches of ``chunk_size``, then casts to the requested dtype.

    Args:
        sssp_fn: SSSP function (bellman_ford_batched, scipy_dijkstra_batched,
            or networkit_dijkstra_batched).
        graph: Input graph.
        source_indices: (K,) int64 anchor indices.
        chunk_size: Number of anchors per chunk.
        sentinel: Value for unreachable nodes.
        dtype: Output dtype (cast applied after all chunks computed).

    Returns:
        (K, V) distance tensor in the requested dtype.
    """
    K = source_indices.shape[0]
    V = graph.num_nodes
    num_chunks = (K + chunk_size - 1) // chunk_size

    # Pre-allocate full result in float64
    result = torch.empty(K, V, dtype=torch.float64)

    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        chunk_sources = source_indices[start:end]
        chunk_dist = sssp_fn(graph, chunk_sources, sentinel=sentinel)
        result[start:end] = chunk_dist
        logger.info(
            "SSSP chunk %d/%d (%d/%d anchors)",
            start // chunk_size + 1,
            num_chunks,
            end,
            K,
        )

    # Cast to requested dtype
    if dtype != torch.float64:
        result = result.to(dtype)

    return result


def compute_teacher_labels(
    graph: Graph,
    anchor_indices: torch.Tensor,
    use_gpu: bool = True,
    sentinel: float = 1e18,
    chunk_size: int | None = None,
    dtype: torch.dtype = torch.float64,
    backend: str = "auto",
) -> TeacherLabels:
    """Compute forward and reverse SSSP from each anchor.

    Forward: d_out[k, v] = d(anchor_k, v) -- distance FROM anchor to vertex.
    Reverse: d_in[k, v] = d(v, anchor_k) -- distance FROM vertex to anchor.

    For undirected graphs, d_out == d_in (symmetric distances).
    For directed graphs, reverse SSSP is computed on the transposed graph.

    Args:
        graph: Input graph.
        anchor_indices: (K,) int64 anchor vertex indices.
        use_gpu: If True and backend='auto' without NetworKit, use
            bellman_ford_batched. Otherwise scipy_dijkstra_batched.
        sentinel: Value for unreachable nodes.
        chunk_size: Number of anchors per chunk. None means no chunking
            (all anchors processed at once). Set to e.g. 8 to reduce
            peak memory from O(K*V) to O(chunk_size*V).
        dtype: Output tensor dtype. Default torch.float64. Use
            torch.float32 to halve memory at ~1e-7 relative error.
        backend: SSSP backend selection. One of:
            - 'auto': Try NetworKit first, fall back to SciPy/Bellman-Ford.
            - 'scipy': Force SciPy Dijkstra backend.
            - 'networkit': Force NetworKit Dijkstra (raises if not installed).

    Returns:
        TeacherLabels with d_out, d_in, anchor_indices, and is_directed flag.

    Raises:
        ValueError: If backend is not one of 'auto', 'scipy', 'networkit'.
        ImportError: If backend='networkit' and networkit is not installed.
    """
    # Validate backend parameter (T-17-07)
    valid_backends = {"auto", "scipy", "networkit"}
    if backend not in valid_backends:
        raise ValueError(
            f"Unknown backend {backend!r}. Must be one of {sorted(valid_backends)}."
        )

    # Memory guard: auto-reduce chunk_size if needed (T-17-04)
    chunk_size = memory_guard(
        num_anchors=anchor_indices.shape[0],
        num_nodes=graph.num_nodes,
        is_directed=graph.is_directed,
        chunk_size=chunk_size,
        dtype=dtype,
    )

    # Backend selection
    if backend == "networkit":
        sssp_fn = networkit_dijkstra_batched
    elif backend == "scipy":
        sssp_fn = scipy_dijkstra_batched
    else:
        # backend == "auto"
        try:
            import networkit  # noqa: F401
            sssp_fn = networkit_dijkstra_batched
        except ImportError:
            # Fall back to existing behavior.
            # Guard: only use BF if CUDA is actually available, otherwise
            # BF on CPU is ~1000x slower than SciPy Dijkstra.
            if use_gpu and torch.cuda.is_available():
                sssp_fn = bellman_ford_batched
            else:
                sssp_fn = scipy_dijkstra_batched

    # Forward SSSP: distances from each anchor
    if chunk_size is not None:
        d_out = _chunked_sssp(sssp_fn, graph, anchor_indices, chunk_size, sentinel, dtype)
    else:
        d_out = sssp_fn(graph, anchor_indices, sentinel=sentinel)
        if dtype != torch.float64:
            d_out = d_out.to(dtype)

    if not graph.is_directed:
        # Undirected: distances are symmetric
        d_in = d_out
    else:
        # Directed: compute reverse SSSP on transposed graph
        # Transpose = swap source/target in edges
        row_idx, col_idx, weights = graph.to_coo()
        transposed = edges_to_graph(
            sources=col_idx,
            targets=row_idx,
            weights=weights,
            num_nodes=graph.num_nodes,
            is_directed=True,
        )
        if chunk_size is not None:
            d_in = _chunked_sssp(sssp_fn, transposed, anchor_indices, chunk_size, sentinel, dtype)
        else:
            d_in = sssp_fn(transposed, anchor_indices, sentinel=sentinel)
            if dtype != torch.float64:
                d_in = d_in.to(dtype)

    return TeacherLabels(
        d_out=d_out,
        d_in=d_in,
        anchor_indices=anchor_indices,
        is_directed=graph.is_directed,
    )
