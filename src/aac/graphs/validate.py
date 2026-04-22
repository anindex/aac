"""Graph validation checks."""

from __future__ import annotations

import torch

from aac.graphs.types import Graph


def validate_graph(graph: Graph) -> None:
    """Validate graph structure and weights.

    Raises ValueError with a descriptive message if:
    - Any edge weight <= 0
    - is_directed=False but adjacency is not symmetric
    - crow_indices shape != (num_nodes + 1,)
    - col_indices or values shape != (num_edges,)
    - Any col_index >= num_nodes or < 0

    Args:
        graph: Graph to validate.

    Raises:
        ValueError: If validation fails.
    """
    # Check crow_indices shape
    expected_crow_shape = (graph.num_nodes + 1,)
    if graph.crow_indices.shape != expected_crow_shape:
        raise ValueError(
            f"crow_indices shape mismatch: expected {expected_crow_shape}, "
            f"got {graph.crow_indices.shape}"
        )

    # Check col_indices and values shapes
    expected_edge_shape = (graph.num_edges,)
    if graph.col_indices.shape != expected_edge_shape:
        raise ValueError(
            f"col_indices/values shape mismatch: expected {expected_edge_shape}, "
            f"got col_indices {graph.col_indices.shape}"
        )
    if graph.values.shape != expected_edge_shape:
        raise ValueError(
            f"col_indices/values shape mismatch: expected {expected_edge_shape}, "
            f"got values {graph.values.shape}"
        )

    # Check col_indices bounds
    if graph.num_edges > 0:
        if torch.any(graph.col_indices < 0):
            raise ValueError("col_indices out of bounds: found negative indices")
        if torch.any(graph.col_indices >= graph.num_nodes):
            raise ValueError(
                f"col_indices out of bounds: found indices >= {graph.num_nodes}"
            )

    # Check for negative edge weights (zero weights are valid for Dijkstra/A*)
    if graph.num_edges > 0 and torch.any(graph.values < 0):
        raise ValueError("Graph has negative edge weights")

    # Check symmetry for undirected graphs
    if not graph.is_directed and graph.num_edges > 0:
        # Build dense adjacency and check symmetry
        V = graph.num_nodes
        dense = torch.zeros(V, V, dtype=graph.values.dtype, device=graph.values.device)
        row_indices, col_indices, values = graph.to_coo()
        dense[row_indices, col_indices] = values

        # Check if A == A^T (ignoring diagonal)
        if not torch.allclose(dense, dense.T):
            raise ValueError("Undirected graph has asymmetric adjacency")
