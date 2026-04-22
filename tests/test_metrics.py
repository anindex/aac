"""Tests for experiment metrics: collector, timing, admissibility, memory."""

from __future__ import annotations

import numpy as np

from experiments.metrics.admissibility import AdmissibilityResult, check_admissibility
from experiments.metrics.collector import (
    MetricsCollector,
    PreprocessingMetrics,
    QueryMetrics,
    batch_throughput,
)
from experiments.metrics.timing import TimingResult, time_query
from experiments.utils import memory_bytes_per_vertex
from aac.search.types import SearchResult


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    """Tests for MetricsCollector summary and export."""

    def _make_result(
        self, cost: float = 10.0, expansions: int = 50, optimal: bool = True
    ) -> SearchResult:
        return SearchResult(
            path=[0, 1, 2], cost=cost, expansions=expansions, optimal=optimal
        )

    def test_summary_empty(self) -> None:
        mc = MetricsCollector()
        s = mc.summary()
        assert s["num_queries"] == 0
        assert s["total_expansions"] == 0

    def test_summary_single_query(self) -> None:
        mc = MetricsCollector()
        mc.add_query(0, 0, 5, self._make_result(cost=10.0, expansions=50), 1.5)
        s = mc.summary()
        assert s["num_queries"] == 1
        assert s["expansions_mean"] == 50.0
        assert s["expansions_median"] == 50.0
        assert s["cost_mean"] == 10.0
        assert s["total_expansions"] == 50
        assert s["num_violations"] == 0

    def test_summary_with_violations(self) -> None:
        mc = MetricsCollector()
        mc.add_query(0, 0, 1, self._make_result(optimal=True), 1.0)
        mc.add_query(1, 2, 3, self._make_result(optimal=False), 2.0)
        mc.add_query(2, 4, 5, self._make_result(optimal=False), 3.0)
        s = mc.summary()
        assert s["num_violations"] == 2
        assert s["num_queries"] == 3

    def test_summary_percentiles(self) -> None:
        mc = MetricsCollector()
        for i in range(100):
            mc.add_query(i, 0, 1, self._make_result(), float(i))
        s = mc.summary()
        # p50 should be around 49.5, p95 around 94.05
        assert 49.0 <= s["p50_ms"] <= 50.0
        assert 93.0 <= s["p95_ms"] <= 95.5

    def test_to_dataframe(self) -> None:
        mc = MetricsCollector()
        mc.add_query(0, 0, 5, self._make_result(), 1.5)
        mc.add_query(1, 2, 3, self._make_result(), 2.5)
        df = mc.to_dataframe()
        assert len(df) == 2
        assert "query_idx" in df.columns
        assert "cost" in df.columns
        assert "latency_ms" in df.columns

    def test_to_csv(self, tmp_path) -> None:
        mc = MetricsCollector()
        mc.add_query(0, 0, 5, self._make_result(), 1.5)
        csv_path = tmp_path / "test.csv"
        mc.to_csv(str(csv_path))
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "query_idx" in content
        assert "cost" in content


# ---------------------------------------------------------------------------
# QueryMetrics / PreprocessingMetrics
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Tests for metric dataclasses."""

    def test_query_metrics_fields(self) -> None:
        qm = QueryMetrics(
            query_idx=0,
            source=1,
            target=2,
            cost=10.0,
            expansions=50,
            latency_ms=1.5,
            optimal=True,
            h_source=3.0,
        )
        assert qm.query_idx == 0
        assert qm.cost == 10.0
        assert qm.optimal is True

    def test_preprocessing_metrics_fields(self) -> None:
        pm = PreprocessingMetrics(
            anchor_selection_sec=1.0,
            sssp_sec=5.0,
            training_sec=10.0,
            total_sec=16.0,
        )
        assert pm.total_sec == 16.0
        assert pm.anchor_selection_sec == 1.0


# ---------------------------------------------------------------------------
# TimingResult / time_query
# ---------------------------------------------------------------------------


class TestTiming:
    """Tests for timing utility."""

    def test_timing_result_fields(self) -> None:
        tr = TimingResult(
            median_ms=1.0, p50_ms=1.0, p95_ms=2.0, min_ms=0.5, max_ms=3.0, num_runs=100
        )
        assert tr.p50_ms == 1.0
        assert tr.p95_ms == 2.0
        assert tr.num_runs == 100

    def test_time_query_returns_result(self) -> None:
        counter = [0]

        def dummy():
            counter[0] += 1

        result = time_query(dummy, warmup_runs=2, num_runs=10)
        assert isinstance(result, TimingResult)
        assert result.num_runs == 10
        assert result.p50_ms >= 0
        assert result.p95_ms >= result.p50_ms
        assert result.min_ms <= result.max_ms
        # warmup(2) + timed(10) = 12 calls
        assert counter[0] == 12

    def test_time_query_percentiles_order(self) -> None:
        result = time_query(lambda: None, warmup_runs=1, num_runs=20)
        assert result.min_ms <= result.p50_ms
        assert result.p50_ms <= result.p95_ms
        assert result.p95_ms <= result.max_ms


# ---------------------------------------------------------------------------
# Admissibility
# ---------------------------------------------------------------------------


class TestAdmissibility:
    """Tests for admissibility checker."""

    def test_no_violations(self) -> None:
        results = [
            SearchResult(path=[0, 1], cost=5.0, expansions=10, optimal=True),
            SearchResult(path=[0, 2], cost=8.0, expansions=15, optimal=True),
        ]
        dijkstra_costs = [5.0, 8.0]
        ar = check_admissibility(results, dijkstra_costs)
        assert ar.num_violations == 0
        assert ar.violation_indices == []
        assert ar.num_queries == 2
        assert ar.max_cost_diff <= 1e-6

    def test_with_violations(self) -> None:
        results = [
            SearchResult(path=[0, 1], cost=5.0, expansions=10, optimal=True),
            SearchResult(path=[0, 2], cost=9.0, expansions=15, optimal=False),
        ]
        dijkstra_costs = [5.0, 8.0]
        ar = check_admissibility(results, dijkstra_costs)
        assert ar.num_violations == 1
        assert ar.violation_indices == [1]
        assert abs(ar.max_cost_diff - 1.0) < 1e-9

    def test_within_tolerance(self) -> None:
        results = [
            SearchResult(path=[0, 1], cost=5.0 + 1e-8, expansions=10, optimal=True),
        ]
        dijkstra_costs = [5.0]
        ar = check_admissibility(results, dijkstra_costs)
        assert ar.num_violations == 0

    def test_length_mismatch_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Length mismatch"):
            check_admissibility([], [1.0])


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class TestMemory:
    """Tests for memory_bytes_per_vertex."""

    def test_default_float32(self) -> None:
        assert memory_bytes_per_vertex(16) == 16 * 4  # float32 default

    def test_custom_dtype_size(self) -> None:
        assert memory_bytes_per_vertex(16, dtype_size=8) == 16 * 8  # float64

    def test_alt_2k(self) -> None:
        # ALT stores 2*K values per vertex (forward + backward)
        assert memory_bytes_per_vertex(2 * 16) == 2 * 16 * 4


# ---------------------------------------------------------------------------
# batch_throughput
# ---------------------------------------------------------------------------


class TestBatchThroughput:
    """Tests for batch_throughput function."""

    def test_batch_throughput_returns_dict(self) -> None:
        import torch

        from aac.graphs.convert import edges_to_graph

        # Build a small triangle graph: 0-1-2-0
        sources = torch.tensor([0, 1, 2], dtype=torch.int64)
        targets = torch.tensor([1, 2, 0], dtype=torch.int64)
        weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        graph = edges_to_graph(sources, targets, weights, num_nodes=3, is_directed=False)

        queries = [(0, 1), (0, 2), (1, 2)]

        def zero_h(node: int, target: int) -> float:
            return 0.0

        result = batch_throughput(
            graph, zero_h, queries, batch_sizes=[1, 2, 3]
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {1, 2, 3}
        for bs, qps in result.items():
            assert qps > 0, f"Throughput for batch_size={bs} should be positive"

    def test_batch_throughput_default_sizes(self) -> None:
        import torch

        from aac.graphs.convert import edges_to_graph

        sources = torch.tensor([0, 1], dtype=torch.int64)
        targets = torch.tensor([1, 0], dtype=torch.int64)
        weights = torch.tensor([1.0, 1.0], dtype=torch.float64)
        graph = edges_to_graph(sources, targets, weights, num_nodes=2, is_directed=False)

        queries = [(0, 1)] * 1024

        result = batch_throughput(graph, lambda n, t: 0.0, queries)
        # Default batch sizes
        assert 1 in result
        assert 1024 in result
