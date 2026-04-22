"""Batched query execution for multiple source-target pairs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from aac.graphs.types import Graph
from aac.search.astar import astar
from aac.search.types import SearchResult


def batch_search(
    graph: Graph,
    queries: list[tuple[int, int]],
    heuristic: Callable[[int, int], float],
    search_fn: Optional[Callable] = None,
) -> list[SearchResult]:
    """Run search for multiple source-target pairs.

    Args:
        graph: Graph in CSR format with non-negative edge weights.
        queries: List of (source, target) pairs.
        heuristic: Admissible heuristic function h(node, target) -> float.
        search_fn: Search function to use. Defaults to astar.
            Must accept (graph, source, target, heuristic=...) signature.

    Returns:
        List of SearchResult, one per query, in the same order as queries.
    """
    if search_fn is None:
        search_fn = astar

    results: list[SearchResult] = []
    for source, target in queries:
        result = search_fn(graph, source, target, heuristic=heuristic)
        results.append(result)

    return results
