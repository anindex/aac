"""Bidirectional A* search with mu-based stopping on CSR graphs."""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Callable

from aac.graphs.types import Graph
from aac.search.types import SearchResult


def bidirectional_astar(
    graph: Graph,
    source: int,
    target: int,
    h_forward: Callable[[int, int], float],
    h_backward: Callable[[int, int], float],
) -> SearchResult:
    """Find shortest path using bidirectional A* with mu-based stopping.

    Args:
        graph: Graph in CSR format with non-negative edge weights.
        source: Source vertex index.
        target: Target vertex index.
        h_forward: Consistent (monotone) heuristic h(node, target) for forward search.
            Must satisfy h(u,t) <= w(u,v) + h(v,t) for all edges (u,v).
            Merely admissible heuristics are NOT sufficient for the
            bidirectional stopping condition to guarantee optimality.
        h_backward: Consistent heuristic h(node, source) for backward search.
            For undirected graphs, h_backward can be the same as h_forward
            with swapped arguments.

    Returns:
        SearchResult with optimal path, cost, and combined expansion count.
    """
    if source == target:
        return SearchResult(
            path=[source],
            cost=0.0,
            expansions=0,
            optimal=True,
            h_source=0.0,
        )

    V = graph.num_nodes

    # Build forward adjacency from CSR
    crow = graph.crow_indices
    col = graph.col_indices
    vals = graph.values

    # Build reverse adjacency dict from COO
    rev_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    rows, cols, weights = graph.to_coo()
    for i in range(rows.shape[0]):
        u = rows[i].item()
        v = cols[i].item()
        w = weights[i].item()
        rev_adj[v].append((u, w))

    # Forward search state
    g_fwd = [float("inf")] * V
    g_fwd[source] = 0.0
    parent_fwd = [-1] * V
    closed_fwd = [False] * V

    # Backward search state
    g_bwd = [float("inf")] * V
    g_bwd[target] = 0.0
    parent_bwd = [-1] * V
    closed_bwd = [False] * V

    h_s = h_forward(source, target)

    # Open lists: (f, counter, node)
    counter = 0
    open_fwd: list[tuple[float, int, int]] = []
    open_bwd: list[tuple[float, int, int]] = []

    heapq.heappush(open_fwd, (g_fwd[source] + h_forward(source, target), counter, source))
    counter += 1
    heapq.heappush(open_bwd, (g_bwd[target] + h_backward(target, source), counter, target))
    counter += 1

    # Best path cost found so far
    mu = float("inf")
    meeting_node = -1

    expansions = 0

    while open_fwd or open_bwd:
        # Get minimum f-values from both open lists
        f_top_fwd = open_fwd[0][0] if open_fwd else float("inf")
        f_top_bwd = open_bwd[0][0] if open_bwd else float("inf")

        # Stopping condition for bidirectional A* with consistent heuristics:
        # We can stop when the minimum of the two top f-values exceeds mu.
        # This guarantees that no undiscovered path can beat the best known
        # path cost mu, since any such path must pass through a node with
        # f >= min(f_top_fwd, f_top_bwd) in at least one direction.
        if min(f_top_fwd, f_top_bwd) >= mu:
            break

        # Expand from the direction with smaller f-top
        if f_top_fwd <= f_top_bwd:
            # Expand forward
            _, _, u = heapq.heappop(open_fwd)
            if closed_fwd[u]:
                continue
            closed_fwd[u] = True
            expansions += 1

            # Check if this node was already closed in backward direction
            if closed_bwd[u]:
                candidate = g_fwd[u] + g_bwd[u]
                if candidate < mu:
                    mu = candidate
                    meeting_node = u

            # Expand forward neighbors via CSR
            start = crow[u].item()
            end = crow[u + 1].item()
            for idx in range(start, end):
                v = col[idx].item()
                w = vals[idx].item()
                new_g = g_fwd[u] + w
                if new_g < g_fwd[v]:
                    g_fwd[v] = new_g
                    parent_fwd[v] = u
                    f_v = new_g + h_forward(v, target)
                    heapq.heappush(open_fwd, (f_v, counter, v))
                    counter += 1
                    # Check if v has finite g-value in backward direction
                    if g_bwd[v] < float("inf"):
                        candidate = new_g + g_bwd[v]
                        if candidate < mu:
                            mu = candidate
                            meeting_node = v
        else:
            # Expand backward
            _, _, u = heapq.heappop(open_bwd)
            if closed_bwd[u]:
                continue
            closed_bwd[u] = True
            expansions += 1

            # Check if this node was already closed in forward direction
            if closed_fwd[u]:
                candidate = g_fwd[u] + g_bwd[u]
                if candidate < mu:
                    mu = candidate
                    meeting_node = u

            # Expand backward neighbors (reverse adjacency)
            for v, w in rev_adj.get(u, []):
                new_g = g_bwd[u] + w
                if new_g < g_bwd[v]:
                    g_bwd[v] = new_g
                    parent_bwd[v] = u
                    f_v = new_g + h_backward(v, source)
                    heapq.heappush(open_bwd, (f_v, counter, v))
                    counter += 1
                    # Check if v has finite g-value in forward direction
                    if g_fwd[v] < float("inf"):
                        candidate = g_fwd[v] + new_g
                        if candidate < mu:
                            mu = candidate
                            meeting_node = v

    if meeting_node == -1 or mu == float("inf"):
        return SearchResult(
            path=[],
            cost=float("inf"),
            expansions=expansions,
            optimal=False,
            h_source=h_s,
        )

    # Reconstruct path through meeting node
    path = _reconstruct_bidirectional_path(
        parent_fwd, parent_bwd, source, target, meeting_node
    )

    return SearchResult(
        path=path,
        cost=mu,
        expansions=expansions,
        optimal=True,
        h_source=h_s,
    )


def _reconstruct_bidirectional_path(
    parent_fwd: list[int],
    parent_bwd: list[int],
    source: int,
    target: int,
    meeting_node: int,
) -> list[int]:
    """Reconstruct path from source to target through meeting node.

    Walks parent_fwd from meeting_node back to source, then
    walks parent_bwd from meeting_node forward to target.
    """
    max_steps = len(parent_fwd)

    # Forward part: meeting_node -> source (reversed)
    fwd_part = [meeting_node]
    current = meeting_node
    for _ in range(max_steps):
        if current == source:
            break
        current = parent_fwd[current]
        if current == -1:
            return []
        fwd_part.append(current)
    else:
        raise RuntimeError("Bidirectional path reconstruction exceeded V steps -- corrupted parent array")
    fwd_part.reverse()

    # Backward part: meeting_node -> target
    bwd_part = []
    current = meeting_node
    for _ in range(max_steps):
        if current == target:
            break
        current = parent_bwd[current]
        if current == -1:
            return []
        bwd_part.append(current)
    else:
        raise RuntimeError("Bidirectional path reconstruction exceeded V steps -- corrupted parent array")

    return fwd_part + bwd_part
