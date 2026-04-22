"""Search result types for single-pair shortest path queries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchResult:
    """Result of a single-pair shortest path query.

    Attributes:
        path: Vertex sequence from source to target (empty if unreachable).
        cost: Total path cost (inf if unreachable).
        expansions: Number of nodes popped from the open list.
        optimal: True if the search completed normally (path is provably optimal).
        h_source: Heuristic value h(source, target) for diagnostics.
        expanded_nodes: Ordered list of expanded node IDs (expansion order preserved).
            ``None`` when expansion tracking is disabled (default).
            The list preserves insertion order so heatmaps can color by
            expansion step.
        g_values: Mapping from node ID to best-known g-value at termination.
            ``None`` when expansion tracking is disabled (default).
            Together with an admissible heuristic, f-values can be
            recomputed as ``g_values[v] + h(v, target)``.
    """

    path: list[int]
    cost: float
    expansions: int
    optimal: bool
    h_source: float = 0.0
    expanded_nodes: list[int] | None = field(default=None, repr=False)
    g_values: dict[int, float] | None = field(default=None, repr=False)
