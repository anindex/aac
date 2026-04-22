"""Tests for reproducibility utilities: seeding and query generation."""

from __future__ import annotations

import torch

from experiments.utils import generate_queries, seed_everything


class TestSeedEverything:
    """Tests for seed_everything determinism."""

    def test_same_seed_same_output(self) -> None:
        seed_everything(42)
        a = torch.randn(10)
        seed_everything(42)
        b = torch.randn(10)
        assert torch.allclose(a, b), "Same seed should produce identical torch.randn output"

    def test_different_seed_different_output(self) -> None:
        seed_everything(42)
        a = torch.randn(10)
        seed_everything(99)
        b = torch.randn(10)
        assert not torch.allclose(a, b), "Different seeds should produce different output"


class TestGenerateQueries:
    """Tests for generate_queries reproducibility and correctness."""

    def _make_small_graph(self):
        """Create a small connected graph for testing."""
        from aac.graphs.convert import edges_to_graph

        # Simple chain: 0-1-2-3-4
        sources = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        targets = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        return edges_to_graph(
            sources, targets, weights, num_nodes=5, is_directed=False
        )

    def test_same_seed_same_queries(self) -> None:
        graph = self._make_small_graph()
        q1 = generate_queries(graph, 20, seed=42)
        q2 = generate_queries(graph, 20, seed=42)
        assert q1 == q2, "Same seed should produce identical query lists"

    def test_different_seed_different_queries(self) -> None:
        graph = self._make_small_graph()
        q1 = generate_queries(graph, 20, seed=42)
        q2 = generate_queries(graph, 20, seed=99)
        assert q1 != q2, "Different seeds should produce different query lists"

    def test_no_self_loops(self) -> None:
        graph = self._make_small_graph()
        queries = generate_queries(graph, 50, seed=42)
        for s, t in queries:
            assert s != t, f"Query ({s}, {t}) has source == target"

    def test_correct_count(self) -> None:
        graph = self._make_small_graph()
        queries = generate_queries(graph, 15, seed=42)
        assert len(queries) == 15

    def test_queries_in_largest_component(self) -> None:
        """Ensure queries are from the largest connected component."""
        import numpy as np
        import scipy.sparse.csgraph

        from aac.graphs.convert import edges_to_graph, graph_to_scipy

        # Create graph with two components: {0,1,2} and {3,4}
        sources = torch.tensor([0, 1, 3], dtype=torch.int64)
        targets = torch.tensor([1, 2, 4], dtype=torch.int64)
        weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        graph = edges_to_graph(
            sources, targets, weights, num_nodes=5, is_directed=False
        )

        sp = graph_to_scipy(graph)
        _, labels = scipy.sparse.csgraph.connected_components(sp, directed=False)
        sizes = np.bincount(labels)
        largest = int(np.argmax(sizes))
        largest_nodes = set(np.where(labels == largest)[0])

        queries = generate_queries(graph, 20, seed=42)
        for s, t in queries:
            assert s in largest_nodes, f"Source {s} not in largest component"
            assert t in largest_nodes, f"Target {t} not in largest component"

    def test_directed_queries_in_strong_lcc(self) -> None:
        """Regression: directed graphs must restrict to strong LCC.

        A graph with SCC gap (weak LCC > strong LCC) must only sample
        queries from the largest *strong* component, matching the
        landmark preprocessing in BaseRunner.preprocess_aac().
        """
        import numpy as np
        import scipy.sparse.csgraph

        from aac.graphs.convert import edges_to_graph, graph_to_scipy

        # Graph with 2 strong CCs: {0,1} (bidirectional) and {2} (one-way to 0)
        # Node 2 has edge 2->0 only; no reverse. Hence weak CC = {0,1,2} but
        # strong SCCs = {0,1} and {2}. Landmark preprocessing uses strong LCC
        # = {0,1}, so queries must also stay within {0,1}.
        sources = torch.tensor([0, 1, 2], dtype=torch.int64)
        targets = torch.tensor([1, 0, 0], dtype=torch.int64)
        weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        graph = edges_to_graph(
            sources, targets, weights, num_nodes=3, is_directed=True
        )

        sp = graph_to_scipy(graph)
        _, labels = scipy.sparse.csgraph.connected_components(
            sp, directed=True, connection="strong"
        )
        sizes = np.bincount(labels)
        largest = int(np.argmax(sizes))
        strong_nodes = set(np.where(labels == largest)[0])

        queries = generate_queries(graph, 100, seed=42)
        for s, t in queries:
            assert s in strong_nodes, (
                f"Source {s} not in strong LCC {strong_nodes} -- "
                f"generate_queries must use strong CC for directed graphs"
            )
            assert t in strong_nodes, (
                f"Target {t} not in strong LCC {strong_nodes} -- "
                f"generate_queries must use strong CC for directed graphs"
            )
