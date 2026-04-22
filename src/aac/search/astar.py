"""A* search with pluggable heuristic callable on CSR graphs."""

from __future__ import annotations

import heapq
from collections.abc import Callable

from aac.graphs.types import Graph
from aac.search.types import SearchResult


def astar(
    graph: Graph,
    source: int,
    target: int,
    heuristic: Callable[[int, int], float],
    *,
    track_expansions: bool = False,
    use_bpmx: bool = False,
) -> SearchResult:
    """Find shortest path from source to target using A* search.

    Args:
        graph: Graph in CSR format with non-negative edge weights.
        source: Source vertex index.
        target: Target vertex index.
        heuristic: Admissible heuristic function h(node, target) -> float.
            Must satisfy h(node, target) <= d(node, target) for all nodes.
        track_expansions: If True, populate ``expanded_nodes`` and
            ``g_values`` in the returned :class:`SearchResult` for
            visualization.  Disabled by default to avoid overhead.
        use_bpmx: If True, apply Felner-style one-step Bidirectional Pathmax
            propagation at expansion time. For each expanded node ``u`` we
            evaluate ``h(v, target)`` on every successor ``v`` and tighten
            ``h(u, target) := max(h(u, target), max_v h(v, target) - w(u,v))``;
            successors are then enqueued with ``f(v) = g(v) + max(h(v),
            h_BPMX(u) - w(u,v))``. BPMX preserves admissibility under
            closed-set A* without node reopenings, because it only ever
            replaces an admissible value with the max of admissible values.
            Default False (matches the rest of the paper's protocol).

    Returns:
        SearchResult with optimal path, cost, and expansion count.
        When *track_expansions* is True, ``expanded_nodes`` contains
        node IDs in expansion order and ``g_values`` maps each expanded
        node to its best-known distance from *source*.
    """
    if source == target:
        return SearchResult(
            path=[source],
            cost=0.0,
            expansions=0,
            optimal=True,
            h_source=0.0,
            expanded_nodes=[] if track_expansions else None,
            g_values={source: 0.0} if track_expansions else None,
        )

    V = graph.num_nodes
    # Pre-convert CSR tensors to Python lists for fast inner-loop access.
    # Avoids Tensor.item() overhead (~10x speedup per neighbor expansion).
    crow = graph.crow_indices.tolist()
    col = graph.col_indices.tolist()
    vals = graph.values.tolist()

    # g[v] = best known distance from source to v
    g = [float("inf")] * V
    g[source] = 0.0

    # Parent array for path reconstruction
    parent = [-1] * V

    # Closed set: True if node has been expanded
    closed = [False] * V

    h_s = heuristic(source, target)

    # Open list: (f, counter, node)
    # counter for FIFO tie-breaking among equal f-values
    counter = 0
    open_list: list[tuple[float, int, int]] = []
    heapq.heappush(open_list, (g[source] + h_s, counter, source))
    counter += 1

    expansions = 0

    # Optional expansion tracking for visualization
    expanded_order: list[int] | None = [] if track_expansions else None

    while open_list:
        f_current, _, u = heapq.heappop(open_list)

        # Lazy deletion: skip already-closed nodes
        if closed[u]:
            continue

        closed[u] = True
        expansions += 1
        if expanded_order is not None:
            expanded_order.append(u)

        # Found target
        if u == target:
            path = _reconstruct_path(parent, source, target)
            return SearchResult(
                path=path,
                cost=g[target],
                expansions=expansions,
                optimal=True,
                h_source=h_s,
                expanded_nodes=expanded_order,
                g_values={v: g[v] for v in expanded_order} if expanded_order is not None else None,
            )

        # Expand neighbors via CSR (pre-converted lists -- no .item() overhead)
        start = crow[u]
        end = crow[u + 1]

        if use_bpmx:
            # Felner-style one-step Bidirectional Pathmax. We pre-compute
            # h(v, target) for every successor v, then tighten h(u) and use
            # the max-propagated value to bound h(v) admissibly. Both steps
            # only ever take a max of values that are already admissible
            # lower bounds, so admissibility is preserved.
            neighbor_h: list[float] = []
            h_u = heuristic(u, target)
            for idx in range(start, end):
                v = col[idx]
                w = vals[idx]
                hv = heuristic(v, target)
                neighbor_h.append(hv)
                cand = hv - w
                if cand > h_u:
                    h_u = cand
            for offset, idx in enumerate(range(start, end)):
                v = col[idx]
                w = vals[idx]
                hv = neighbor_h[offset]
                propagated = h_u - w
                if propagated > hv:
                    hv = propagated
                new_g = g[u] + w
                if new_g < g[v]:
                    g[v] = new_g
                    parent[v] = u
                    f_v = new_g + hv
                    heapq.heappush(open_list, (f_v, counter, v))
                    counter += 1
        else:
            for idx in range(start, end):
                v = col[idx]
                w = vals[idx]

                new_g = g[u] + w
                if new_g < g[v]:
                    g[v] = new_g
                    parent[v] = u
                    f_v = new_g + heuristic(v, target)
                    heapq.heappush(open_list, (f_v, counter, v))
                    counter += 1

    # Target unreachable
    return SearchResult(
        path=[],
        cost=float("inf"),
        expansions=expansions,
        optimal=False,
        h_source=h_s,
        expanded_nodes=expanded_order,
        g_values={v: g[v] for v in expanded_order} if expanded_order is not None else None,
    )


def _reconstruct_path(parent: list[int], source: int, target: int) -> list[int]:
    """Walk parent pointers from target back to source."""
    path = [target]
    current = target
    max_steps = len(parent)
    for _ in range(max_steps):
        current = parent[current]
        if current == -1:
            return []  # Should not happen if target was reached
        path.append(current)
        if current == source:
            break
    else:
        raise RuntimeError("Path reconstruction exceeded V steps -- corrupted parent array")
    path.reverse()
    return path
