"""Differentiable smooth Bellman-Ford with logsumexp scatter-min.

Replaces the hard `scatter_reduce_("amin")` in the standard Bellman-Ford
with a smooth-min operator based on the negated logsumexp trick:

    smooth_min(x_1, ..., x_n) = -1/beta * logsumexp(-beta * x_1, ..., -beta * x_n)

This makes the shortest-path computation differentiable with respect to
edge weights, enabling gradient-based learning of edge cost predictors
(Contextual of AAC).

All computation uses fp64 for numerical stability.
"""

from __future__ import annotations

import torch

from aac.graphs.types import Graph


def graph_with_weights(graph: Graph, new_weights: torch.Tensor) -> Graph:
    """Create a new Graph with the same topology but different edge weights.

    Since Graph is a frozen dataclass, this constructs a new instance
    with the same CSR structure (crow_indices, col_indices) but updated values.

    Args:
        graph: Original graph providing topology.
        new_weights: (E,) tensor of new edge weights. Must match graph.num_edges.

    Returns:
        New Graph with the given weights.
    """
    assert new_weights.shape[0] == graph.num_edges, (
        f"new_weights has {new_weights.shape[0]} elements but graph has {graph.num_edges} edges"
    )
    return Graph(
        crow_indices=graph.crow_indices,
        col_indices=graph.col_indices,
        values=new_weights,
        num_nodes=graph.num_nodes,
        num_edges=graph.num_edges,
        is_directed=graph.is_directed,
        coordinates=graph.coordinates,
    )


def smooth_bellman_ford_batched(
    graph: Graph,
    source_indices: torch.Tensor,
    beta: float = 10.0,
    sentinel: float = 1e18,
    max_iter: int | None = None,
) -> torch.Tensor:
    """Differentiable batched Bellman-Ford using smooth-min relaxation.

    Replaces hard min with smooth-min via negated logsumexp:
        smooth_min(a, b) = -logsumexp([-beta*a, -beta*b]) / beta

    For each Bellman-Ford iteration, for each target vertex v, we compute
    the smooth-min over the current distance and all incoming edge candidates.

    Implementation uses scatter-based logsumexp for vectorized aggregation:
        1. Compute per-vertex max via scatter_reduce("amax") for stability
        2. Compute exp(neg_beta_cand - max_per_vertex) and scatter_add by target
        3. Add exp(neg_beta_current - max_per_vertex) for self-contribution
        4. new_dist = -(max_per_vertex + log(sum_exps)) / beta

    Args:
        graph: Input graph. values may have requires_grad=True for backprop.
        source_indices: (K,) int64 tensor of source vertex indices.
        beta: Inverse temperature for smooth-min. Higher = closer to hard min.
        sentinel: Value for unreachable nodes.
        max_iter: Maximum BF iterations. Default V-1.

    Returns:
        (K, V) fp64 distance tensor.
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

    # Precompute col_idx expansion for scatter operations
    col_expanded = col_idx.unsqueeze(0).expand(K, E)  # (K, E)

    for _iteration in range(max_iter):
        # Compute candidate distances via each edge
        # candidate[k, e] = dist[k, row_idx[e]] + weights[e]
        source_dist = dist[:, row_idx]  # (K, E)
        candidate = source_dist + weights.unsqueeze(0)  # (K, E)

        # Smooth scatter-min using logsumexp aggregation
        new_dist = _smooth_scatter_min(
            dist, col_idx, col_expanded, candidate, beta, V, K, E
        )

        # Preserve source distances (always 0) -- use torch.where to avoid
        # in-place mutation on a tensor with grad_fn
        source_mask = torch.zeros((K, V), dtype=torch.bool, device=device)
        source_mask[torch.arange(K, device=device), source_indices.to(device)] = True
        new_dist = torch.where(source_mask, torch.zeros_like(new_dist), new_dist)

        # Check convergence (mask out sentinel values to allow early termination
        # on graphs with unreachable nodes)
        reachable = new_dist.abs() < 0.99 * sentinel
        if reachable.any() and torch.allclose(
            new_dist[reachable], dist[reachable], atol=1e-8
        ):
            break
        dist = new_dist

    return dist


def _smooth_scatter_min(
    current_dist: torch.Tensor,  # (K, V)
    col_idx: torch.Tensor,  # (E,)
    col_expanded: torch.Tensor,  # (K, E)
    candidates: torch.Tensor,  # (K, E)
    beta: float,
    V: int,
    K: int,
    E: int,
) -> torch.Tensor:
    """Vectorized smooth scatter-min using logsumexp aggregation.

    For each target vertex v, computes:
        new_dist[k, v] = smooth_min(current_dist[k, v], {candidate[k, e] : col_idx[e] == v})

    Using the identity:
        smooth_min(x_1, ..., x_n) = -logsumexp(-beta * x_1, ..., -beta * x_n) / beta

    Numerically stable via shift-by-max:
        logsumexp(z_1, ..., z_n) = max_z + log(sum(exp(z_i - max_z)))
    """
    # Compute -beta * values for logsumexp
    neg_beta_cand = -beta * candidates  # (K, E)
    neg_beta_dist = -beta * current_dist  # (K, V)

    # Step 1: Find per-vertex max of -beta * candidates for numerical stability
    # We need the max over all incoming candidates AND the current distance
    # Start with current distances as the initial max
    max_per_vertex = neg_beta_dist.clone()  # (K, V)

    # Scatter amax from candidates into target vertices
    max_per_vertex.scatter_reduce_(
        1, col_expanded, neg_beta_cand, reduce="amax", include_self=True
    )

    # Step 2: Compute shifted exponentials and scatter-add
    # exp(neg_beta_cand - max_per_vertex[col_idx]) for each edge
    max_at_targets = max_per_vertex.gather(1, col_expanded)  # (K, E)
    shifted_cand_exp = torch.exp(neg_beta_cand - max_at_targets)  # (K, E)

    # Self-contribution: exp(neg_beta_dist - max_per_vertex) for each vertex
    self_exp = torch.exp(neg_beta_dist - max_per_vertex)  # (K, V)

    # Scatter-add shifted candidate exponentials into vertices
    sum_exps = self_exp.clone()  # (K, V), starts with self-contribution
    sum_exps.scatter_add_(1, col_expanded, shifted_cand_exp)

    # Step 3: Combine: logsumexp = max + log(sum_exps)
    log_sum = max_per_vertex + torch.log(sum_exps)

    # Final: new_dist = -logsumexp / beta
    new_dist = -log_sum / beta

    return new_dist
