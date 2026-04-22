"""Query generation correctness audit.

Verifies that generate_queries() and compute_strong_lcc() correctly restrict
queries to the largest strongly connected component for directed graphs,
produce uniform sampling, and are deterministic given a seed.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.csgraph
from scipy.stats import chisquare

from experiments.utils import compute_strong_lcc, generate_queries, seed_everything
from tests.fixtures.adversarial_graphs import (
    disconnected_directed,
    scc_boundary_graph,
    strongly_connected_directed_10,
    weakly_not_strongly_connected,
)
from aac.graphs.convert import edges_to_graph, graph_to_scipy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_scc_labels(graph):
    """Compute strong-CC labels via scipy for independent verification."""
    sp = graph_to_scipy(graph)
    n_comp, labels = scipy.sparse.csgraph.connected_components(
        sp, directed=True, connection="strong",
    )
    return n_comp, labels


def _node_scc_map(labels: np.ndarray) -> dict[int, int]:
    """Map node -> SCC id."""
    return {int(i): int(labels[i]) for i in range(len(labels))}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQueriesInStrongLCC:
    """Verify all generated queries land within a single SCC."""

    def test_queries_in_strong_lcc_directed(self):
        """On scc_boundary_graph (directed), all 100 queries must be
        within the same SCC.  The graph has two 4-node SCCs joined by a
        one-way bridge, so queries must NOT span the two SCCs."""
        graph = scc_boundary_graph()
        queries = generate_queries(graph, num_queries=100, seed=42)

        n_comp, labels = _get_scc_labels(graph)
        scc_map = _node_scc_map(labels)

        for i, (s, t) in enumerate(queries):
            assert scc_map[s] == scc_map[t], (
                f"Query {i}: s={s} (SCC {scc_map[s]}) and "
                f"t={t} (SCC {scc_map[t]}) are in different SCCs"
            )

    def test_queries_in_strong_lcc_strongly_connected(self):
        """On strongly_connected_directed_10 (one SCC), all 50 queries
        should be valid with s != t."""
        graph = strongly_connected_directed_10()
        queries = generate_queries(graph, num_queries=50, seed=42)

        assert len(queries) == 50
        for i, (s, t) in enumerate(queries):
            assert s != t, f"Query {i}: s == t == {s}"
            assert 0 <= s < 10, f"Query {i}: s={s} out of range"
            assert 0 <= t < 10, f"Query {i}: t={t} out of range"


class TestUniformSampling:
    """Verify query generation samples nodes approximately uniformly."""

    def test_uniform_sampling(self):
        """On strongly_connected_directed_10 (10 nodes, all in one SCC),
        generate 5000 queries and check that the distribution of source
        and target nodes is approximately uniform via chi-squared test."""
        graph = strongly_connected_directed_10()
        queries = generate_queries(graph, num_queries=5000, seed=42)

        source_counts = np.zeros(10, dtype=int)
        target_counts = np.zeros(10, dtype=int)
        for s, t in queries:
            source_counts[s] += 1
            target_counts[t] += 1

        # Each node should appear ~500 times as source and ~500 as target.
        # Chi-squared test for uniformity (p > 0.01 to avoid flaky failures).
        _, p_source = chisquare(source_counts)
        _, p_target = chisquare(target_counts)

        assert p_source > 0.01, (
            f"Source distribution not uniform: counts={source_counts}, p={p_source:.6f}"
        )
        assert p_target > 0.01, (
            f"Target distribution not uniform: counts={target_counts}, p={p_target:.6f}"
        )


class TestDeterminism:
    """Verify deterministic query generation."""

    def test_queries_deterministic(self):
        """Same seed must produce identical queries."""
        graph = strongly_connected_directed_10()
        q1 = generate_queries(graph, num_queries=100, seed=42)
        q2 = generate_queries(graph, num_queries=100, seed=42)
        assert q1 == q2, "Queries with same seed differ"

    def test_queries_different_seeds(self):
        """Different seeds must produce different queries."""
        graph = strongly_connected_directed_10()
        q1 = generate_queries(graph, num_queries=100, seed=42)
        q2 = generate_queries(graph, num_queries=100, seed=123)
        assert q1 != q2, "Queries with different seeds are identical"


class TestComputeStrongLCC:
    """Verify compute_strong_lcc returns the correct component."""

    def test_compute_strong_lcc_directed_scc_boundary(self):
        """On scc_boundary_graph, the largest SCC should have exactly 4 nodes.
        Both SCCs have 4 nodes; compute_strong_lcc picks the one with the
        smallest index (argmax of bincount), which is deterministic."""
        graph = scc_boundary_graph()
        lcc_nodes, seed_vertex = compute_strong_lcc(graph)

        # Each SCC has 4 nodes; the returned LCC should have 4
        assert len(lcc_nodes) == 4, (
            f"Expected 4-node SCC, got {len(lcc_nodes)} nodes: {lcc_nodes}"
        )

        # Verify the returned nodes form a valid SCC
        n_comp, labels = _get_scc_labels(graph)
        lcc_label = labels[lcc_nodes[0]]
        for node in lcc_nodes:
            assert labels[node] == lcc_label, (
                f"Node {node} has SCC label {labels[node]}, expected {lcc_label}"
            )

        # seed_vertex should be the first node in the LCC
        assert seed_vertex == int(lcc_nodes[0])

    def test_compute_strong_lcc_directed_no_scc(self):
        """On weakly_not_strongly_connected (one-way chain), the largest SCC
        is a single node (each node is its own SCC of size 1)."""
        graph = weakly_not_strongly_connected()
        lcc_nodes, seed_vertex = compute_strong_lcc(graph)

        # Each node is its own SCC, so largest SCC has 1 node
        assert len(lcc_nodes) == 1, (
            f"Expected single-node SCC, got {len(lcc_nodes)} nodes: {lcc_nodes}"
        )
        assert seed_vertex == int(lcc_nodes[0])

    def test_compute_strong_lcc_undirected(self):
        """On an undirected connected graph, compute_strong_lcc should return
        all nodes (since weak CC = the entire connected graph)."""
        import torch

        # Build a simple 5-node undirected connected path graph: 0-1-2-3-4
        s = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        t = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        w = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        graph = edges_to_graph(s, t, w, 5, is_directed=False)

        lcc_nodes, seed_vertex = compute_strong_lcc(graph)

        assert len(lcc_nodes) == 5, (
            f"Expected all 5 nodes in LCC, got {len(lcc_nodes)}: {lcc_nodes}"
        )
        assert set(lcc_nodes.tolist()) == {0, 1, 2, 3, 4}
        assert seed_vertex == int(lcc_nodes[0])
