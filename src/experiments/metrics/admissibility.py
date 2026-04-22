"""Admissibility (path optimality) checker for heuristic search results."""

from __future__ import annotations

import math
from dataclasses import dataclass

from aac.search.types import SearchResult


@dataclass(frozen=True)
class AdmissibilityResult:
    """Result of admissibility (optimality) verification.

    Attributes:
        num_queries: Total number of queries checked.
        num_violations: Number of queries where search cost exceeds Dijkstra cost.
        violation_indices: Indices of queries with cost violations.
        max_cost_diff: Maximum (search_cost - dijkstra_cost) across all queries.
    """

    num_queries: int
    num_violations: int
    violation_indices: list[int]
    max_cost_diff: float


def check_admissibility(
    search_results: list[SearchResult],
    dijkstra_costs: list[float],
    atol: float = 1e-6,
) -> AdmissibilityResult:
    """Check whether search results are admissible (optimal) against Dijkstra reference.

    A violation occurs when search_result.cost > dijkstra_cost + atol,
    meaning the heuristic-guided search returned a suboptimal path.

    Args:
        search_results: List of SearchResult from heuristic search.
        dijkstra_costs: List of reference optimal costs from Dijkstra.
        atol: Absolute tolerance for cost comparison.

    Returns:
        AdmissibilityResult with violation count and details.
    """
    if len(search_results) != len(dijkstra_costs):
        raise ValueError(
            f"Length mismatch: {len(search_results)} results vs "
            f"{len(dijkstra_costs)} reference costs"
        )

    violations: list[int] = []
    max_diff = 0.0

    for i, (result, ref_cost) in enumerate(zip(search_results, dijkstra_costs)):
        # Skip pairs where both costs are inf (unreachable)
        if math.isinf(result.cost) and math.isinf(ref_cost):
            continue
        diff = result.cost - ref_cost
        if diff > atol:
            violations.append(i)
        if not math.isnan(diff):
            max_diff = max(max_diff, diff)

    return AdmissibilityResult(
        num_queries=len(search_results),
        num_violations=len(violations),
        violation_indices=violations,
        max_cost_diff=max_diff,
    )
