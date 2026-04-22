"""Dijkstra's algorithm as A* with zero heuristic."""

from __future__ import annotations

from aac.graphs.types import Graph
from aac.search.astar import astar
from aac.search.types import SearchResult


def _zero_heuristic(node: int, target: int) -> float:
    """Zero heuristic: h(node, target) = 0.0 for all nodes."""
    return 0.0


def dijkstra(
    graph: Graph, source: int, target: int, *, track_expansions: bool = False
) -> SearchResult:
    """Find shortest path from source to target using Dijkstra's algorithm.

    Implemented as A* with a zero heuristic, guaranteeing optimality.

    Args:
        graph: Graph in CSR format with non-negative edge weights.
        source: Source vertex index.
        target: Target vertex index.
        track_expansions: If True, populate ``expanded_nodes`` and
            ``g_values`` in the returned :class:`SearchResult` for
            visualization.  Disabled by default to avoid overhead.

    Returns:
        SearchResult with optimal path, cost, and expansion count.
    """
    return astar(
        graph,
        source,
        target,
        heuristic=_zero_heuristic,
        track_expansions=track_expansions,
    )
