"""Conversion between CSR, COO, scipy.sparse, and dense formats."""

from __future__ import annotations

from typing import Optional

import scipy.sparse
import torch

from aac.graphs.types import Graph


def edges_to_graph(
    sources: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    num_nodes: int,
    is_directed: bool,
    coordinates: Optional[torch.Tensor] = None,
) -> Graph:
    """Build a Graph in CSR format from edge lists.

    Args:
        sources: (E,) int64 source vertex indices (0-indexed).
        targets: (E,) int64 target vertex indices (0-indexed).
        weights: (E,) fp64 edge weights.
        num_nodes: Number of vertices V.
        is_directed: Whether the graph is directed.
        coordinates: Optional (V, 2) coordinates for spatial graphs.

    Returns:
        Graph in CSR format. For undirected graphs, both (u,v) and (v,u) edges
        are stored, so num_edges = 2 * len(sources).
    """
    src = sources.to(torch.int64)
    tgt = targets.to(torch.int64)
    wgt = weights.to(torch.float64)

    if not is_directed:
        # Add reverse edges for undirected graphs
        src = torch.cat([src, tgt])
        tgt = torch.cat([targets.to(torch.int64), sources.to(torch.int64)])
        wgt = torch.cat([wgt, weights.to(torch.float64)])

    num_edges = src.shape[0]

    # Sort by (source, target) for CSR construction
    if num_nodes < 3_000_000_000:  # int64 safe: src * num_nodes + tgt won't overflow
        sort_keys = src * num_nodes + tgt
        order = torch.argsort(sort_keys)
    else:
        # Fall back to numpy lexsort to avoid int64 overflow
        import numpy as np
        order_np = np.lexsort((tgt.cpu().numpy(), src.cpu().numpy()))
        order = torch.tensor(order_np, dtype=torch.int64, device=src.device)
    src_sorted = src[order]
    tgt_sorted = tgt[order]
    wgt_sorted = wgt[order]

    # Build crow_indices using bincount
    counts = torch.bincount(src_sorted, minlength=num_nodes)
    crow_indices = torch.zeros(num_nodes + 1, dtype=torch.int64)
    crow_indices[1:] = torch.cumsum(counts, dim=0)

    return Graph(
        crow_indices=crow_indices,
        col_indices=tgt_sorted,
        values=wgt_sorted,
        num_nodes=num_nodes,
        num_edges=num_edges,
        is_directed=is_directed,
        coordinates=coordinates,
    )


def graph_to_scipy(graph: Graph) -> scipy.sparse.csr_matrix:
    """Convert a Graph to a SciPy CSR matrix.

    Args:
        graph: Graph in CSR format.

    Returns:
        scipy.sparse.csr_matrix with shape (V, V).
    """
    return scipy.sparse.csr_matrix(
        (
            graph.values.cpu().numpy(),
            graph.col_indices.cpu().numpy(),
            graph.crow_indices.cpu().numpy(),
        ),
        shape=(graph.num_nodes, graph.num_nodes),
    )


def transpose_graph(graph: Graph) -> Graph:
    """Transpose a directed graph (reverse all edge directions).

    For each edge (u, v, w) in the input, the output contains (v, u, w).
    Useful for computing reverse SSSP (d_in distances) on directed graphs.

    Args:
        graph: Directed graph in CSR format.

    Returns:
        New Graph with reversed edges, also in CSR format.
    """
    rows, cols, weights = graph.to_coo()
    # Swap sources and targets
    return edges_to_graph(
        sources=cols,
        targets=rows,
        weights=weights,
        num_nodes=graph.num_nodes,
        is_directed=graph.is_directed,
        coordinates=graph.coordinates,
    )


def scipy_to_graph(mat: scipy.sparse.csr_matrix, is_directed: bool) -> Graph:
    """Convert a SciPy CSR matrix to a Graph.

    The scipy matrix must already contain both directions for each undirected
    edge (i.e., full symmetric matrix, not upper/lower triangle only).
    num_edges is set to mat.nnz, the number of stored entries.

    Args:
        mat: scipy.sparse.csr_matrix with edge weights. For undirected graphs,
            must be symmetric (both (u,v) and (v,u) stored).
        is_directed: Whether the graph is directed.

    Returns:
        Graph in CSR format with fp64 values.
    """
    # Ensure CSR format
    mat = mat.tocsr()
    mat.sort_indices()

    num_nodes = mat.shape[0]
    num_edges = mat.nnz

    crow_indices = torch.tensor(mat.indptr, dtype=torch.int64)
    col_indices = torch.tensor(mat.indices, dtype=torch.int64)
    values = torch.tensor(mat.data, dtype=torch.float64)

    return Graph(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=values,
        num_nodes=num_nodes,
        num_edges=num_edges,
        is_directed=is_directed,
    )


def graph_to_dense(graph: Graph, sentinel: float = 1e18) -> torch.Tensor:
    """Convert a Graph to a (V, V) dense adjacency matrix.

    Non-edges are filled with the sentinel value. Diagonal entries are 0.

    Args:
        graph: Graph in CSR format.
        sentinel: Value for non-edges. Default 1e18.

    Returns:
        (V, V) dense adjacency tensor.
    """
    return graph.to_dense(sentinel=sentinel)
