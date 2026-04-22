"""Heuristic combiners for admissible A* search."""

from __future__ import annotations

from collections.abc import Callable


def make_hybrid_heuristic(
    *heuristics: Callable[[int, int], float],
) -> Callable[[int, int], float]:
    """Combine multiple admissible heuristics by taking the pointwise max.

    Since each h_i(u,t) <= d(u,t), max_i h_i(u,t) <= d(u,t), so the
    combined heuristic is also admissible. The max is at least as
    informative as any individual heuristic.

    Args:
        *heuristics: Two or more heuristic functions h(node, target) -> float.

    Returns:
        Callable h(node, target) -> float that returns max of all inputs.
    """
    if len(heuristics) < 2:
        raise ValueError("Need at least 2 heuristics to combine")

    def h(node: int, target: int) -> float:
        return max(hi(node, target) for hi in heuristics)

    return h
