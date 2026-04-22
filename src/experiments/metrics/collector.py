"""Metrics collector for per-query experiment statistics."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch

from aac.graphs.types import Graph
from aac.search.batch import batch_search
from aac.search.types import SearchResult


@dataclass
class QueryMetrics:
    """Per-query measurement record.

    Attributes:
        query_idx: Index of the query in the query list.
        source: Source vertex.
        target: Target vertex.
        cost: Path cost returned by search.
        expansions: Number of node expansions.
        latency_ms: Query latency in milliseconds.
        optimal: Whether the search returned an optimal path.
        h_source: Heuristic value at source.
        ref_cost: Reference (Dijkstra) optimal cost, if available. Used to
            detect actual suboptimality violations independently of the
            ``optimal`` flag (which is also False for unreachable queries).
    """

    query_idx: int
    source: int
    target: int
    cost: float
    expansions: int
    latency_ms: float
    optimal: bool
    h_source: float
    ref_cost: float | None = None


@dataclass
class PreprocessingMetrics:
    """Preprocessing stage timing breakdown.

    Attributes:
        anchor_selection_sec: Time for anchor/landmark selection.
        sssp_sec: Time for SSSP computations from anchors.
        training_sec: Time for neural network training (AAC only).
        total_sec: Total preprocessing wall time.
    """

    anchor_selection_sec: float
    sssp_sec: float
    training_sec: float
    total_sec: float


@dataclass
class MetricsCollector:
    """Accumulates per-query metrics and exports summaries.

    Usage::

        collector = MetricsCollector()
        for i, (s, t) in enumerate(queries):
            result = astar(graph, s, t, heuristic=h)
            collector.add_query(i, s, t, result, latency_ms=elapsed)
        print(collector.summary())
        collector.to_csv("results.csv")
    """

    _queries: list[QueryMetrics] = field(default_factory=list)

    def add_query(
        self,
        query_idx: int,
        source: int,
        target: int,
        result: SearchResult,
        latency_ms: float,
        ref_cost: float | None = None,
    ) -> None:
        """Record metrics for a single query.

        Args:
            query_idx: Index of this query in the query list.
            source: Source vertex.
            target: Target vertex.
            result: SearchResult from search function.
            latency_ms: Measured latency in milliseconds.
            ref_cost: Optional Dijkstra reference cost for suboptimality detection.
        """
        self._queries.append(
            QueryMetrics(
                query_idx=query_idx,
                source=source,
                target=target,
                cost=result.cost,
                expansions=result.expansions,
                latency_ms=latency_ms,
                optimal=result.optimal,
                h_source=result.h_source,
                ref_cost=ref_cost,
            )
        )

    def summary(self, suboptimality_atol: float = 1e-6) -> dict:
        """Compute aggregate statistics over all recorded queries.

        Args:
            suboptimality_atol: Absolute tolerance for cost comparison when
                detecting suboptimality violations via ``ref_cost``.

        Returns:
            Dictionary with keys: expansions_mean, expansions_median,
            p50_ms, p95_ms, cost_mean, num_violations, num_queries,
            total_expansions.

            ``num_violations`` counts actual suboptimality (cost > ref_cost +
            atol) when reference costs are available, otherwise falls back to
            counting queries where ``optimal`` is False.
        """
        if not self._queries:
            return {
                "expansions_mean": 0.0,
                "expansions_median": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "cost_mean": 0.0,
                "num_violations": 0,
                "num_queries": 0,
                "total_expansions": 0,
            }

        expansions = np.array([q.expansions for q in self._queries])
        latencies = np.array([q.latency_ms for q in self._queries])
        costs = np.array([q.cost for q in self._queries])

        # Count actual suboptimality violations when reference costs are
        # available.  This avoids counting unreachable queries (where
        # optimal=False merely indicates no path was found) as violations.
        has_ref = any(q.ref_cost is not None for q in self._queries)
        if has_ref:
            num_violations = 0
            for q in self._queries:
                if q.ref_cost is None:
                    continue
                # Both unreachable -> not a violation
                if math.isinf(q.cost) and math.isinf(q.ref_cost):
                    continue
                if q.cost - q.ref_cost > suboptimality_atol:
                    num_violations += 1
        else:
            # Fallback: use the optimal flag (legacy behaviour)
            num_violations = sum(1 for q in self._queries if not q.optimal)

        return {
            "expansions_mean": float(np.mean(expansions)),
            "expansions_median": float(np.median(expansions)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "cost_mean": float(np.mean(costs)),
            "num_violations": num_violations,
            "num_queries": len(self._queries),
            "total_expansions": int(np.sum(expansions)),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export all query metrics as a pandas DataFrame.

        Returns:
            DataFrame with one row per query.
        """
        return pd.DataFrame([asdict(q) for q in self._queries])

    def to_csv(self, path: str) -> None:
        """Write all query metrics to a CSV file.

        A ``# git_hash=..., timestamp=...`` comment line is prepended so
        that every result file records which code version produced it.
        Readers should use ``pd.read_csv(path, comment='#')`` to skip it.

        Args:
            path: File path for the CSV output.
        """
        from experiments.reporting.csv_writer import write_csv_metadata

        with open(path, "w", newline="") as f:
            write_csv_metadata(f)
            self.to_dataframe().to_csv(f, index=False)


def batch_throughput(
    graph: Graph,
    heuristic: Callable[[int, int], float],
    queries: list[tuple[int, int]],
    batch_sizes: list[int] | None = None,
    seed: int = 42,
) -> dict[int, float]:
    """Measure query throughput (queries/second) at various batch sizes.

    Implements METR-06: batch throughput measurement. For each batch size,
    runs batch_search and computes queries per second.

    Args:
        graph: Graph in CSR format.
        heuristic: Admissible heuristic function h(node, target) -> float.
        queries: Full list of (source, target) query pairs.
        batch_sizes: List of batch sizes to test. Default [1, 8, 32, 128, 1024].
        seed: RNG seed (unused, reserved for future stochastic methods).

    Returns:
        Dict mapping batch_size -> throughput in queries per second.
        Example: {1: 500.0, 8: 2000.0, 32: 5000.0}
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 128, 1024]

    results: dict[int, float] = {}
    for bs in batch_sizes:
        batch_queries = queries[:bs] if len(queries) >= bs else queries
        actual_size = len(batch_queries)

        # Warmup call to avoid measuring JIT/cache-cold overhead
        batch_search(graph, batch_queries, heuristic)

        # Synchronize CUDA if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        batch_search(graph, batch_queries, heuristic)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        throughput = actual_size / elapsed if elapsed > 0 else float("inf")
        results[bs] = throughput

    return results
