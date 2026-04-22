"""Property-based tests for admissibility invariants using hypothesis.

Tests three classes of properties:
1. Row-stochastic invariants: Gumbel-softmax produces row-stochastic A,
   hard argmax produces one-hot rows.
2. Admissibility on random graphs: ALT, LinearCompressor, and AAC heuristics
   never exceed true shortest-path distances.
3. Substochastic admissibility: row-substochastic matrices preserve the
   max(A @ delta) <= max(delta) bound.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.sparse.csgraph import shortest_path

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.compression.smooth import make_aac_heuristic
from aac.graphs.convert import edges_to_graph, graph_to_scipy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_random_connected_graph(num_nodes, num_extra_edges, seed, is_directed=False):
    """Build a connected graph: spanning tree + random edges.

    Constructs a random spanning tree to guarantee connectivity, then adds
    extra random edges for density. All weights are in [1.0, 10.0].

    Args:
        num_nodes: Number of vertices (V).
        num_extra_edges: Number of additional random edges beyond spanning tree.
        seed: Random seed for reproducibility.
        is_directed: Whether the graph is directed.

    Returns:
        Graph in CSR format.
    """
    rng = np.random.RandomState(seed)
    sources, targets, weights = [], [], []

    # Spanning tree (ensures connectivity)
    perm = rng.permutation(num_nodes)
    for i in range(num_nodes - 1):
        w = rng.uniform(1.0, 10.0)
        u, v = int(perm[i]), int(perm[i + 1])
        sources.append(u)
        targets.append(v)
        weights.append(w)
        if is_directed:
            # Add reverse edge for strong connectivity in directed case
            sources.append(v)
            targets.append(u)
            weights.append(rng.uniform(1.0, 10.0))

    # Extra edges
    for _ in range(num_extra_edges):
        u = rng.randint(0, num_nodes)
        v = rng.randint(0, num_nodes)
        if u == v:
            continue
        w = rng.uniform(1.0, 10.0)
        sources.append(u)
        targets.append(v)
        weights.append(w)
        if is_directed:
            # Add reverse for extra edges too (ensures strong connectivity)
            sources.append(v)
            targets.append(u)
            weights.append(rng.uniform(1.0, 10.0))

    src_t = torch.tensor(sources, dtype=torch.int64)
    tgt_t = torch.tensor(targets, dtype=torch.int64)
    wgt_t = torch.tensor(weights, dtype=torch.float64)

    return edges_to_graph(src_t, tgt_t, wgt_t, num_nodes, is_directed=is_directed)


# ---------------------------------------------------------------------------
# 1. Row-Stochastic Invariants (no hypothesis needed)
# ---------------------------------------------------------------------------


class TestRowStochasticInvariants:
    """Verify that LinearCompressor produces proper stochastic matrices."""

    @pytest.mark.parametrize("K,m", [(6, 3), (10, 4), (8, 8)])
    def test_gumbel_softmax_row_sums_one(self, K, m):
        """Training mode produces row-stochastic A (rows sum to 1)."""
        comp = LinearCompressor(K=K, m=m, is_directed=False)
        comp.train()
        A_soft = comp._get_A_soft(comp.W, tau=1.0, dtype=torch.float64)
        # Row sums should be 1.0
        row_sums = A_soft.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
        # All entries non-negative
        assert (A_soft >= -1e-7).all()

    @pytest.mark.parametrize("K,m", [(6, 3), (10, 4), (8, 8)])
    def test_hard_argmax_one_hot(self, K, m):
        """Inference mode produces one-hot rows."""
        comp = LinearCompressor(K=K, m=m, is_directed=False)
        comp.eval()
        A_hard = comp._get_A_hard(comp.W, dtype=torch.float64)
        # Each row should have exactly one 1.0, rest 0.0
        assert torch.allclose(
            A_hard.sum(dim=-1), torch.ones(m, dtype=torch.float64)
        )
        assert ((A_hard == 0.0) | (A_hard == 1.0)).all()

    @pytest.mark.parametrize("K,m", [(6, 3), (10, 4), (8, 8)])
    def test_soft_A_shape(self, K, m):
        """Gumbel-softmax A has shape (m, K)."""
        comp = LinearCompressor(K=K, m=m, is_directed=False)
        comp.train()
        A_soft = comp._get_A_soft(comp.W, tau=1.0, dtype=torch.float64)
        assert A_soft.shape == (m, K)

    @pytest.mark.parametrize("K,m", [(6, 3), (10, 4), (8, 8)])
    def test_hard_A_shape(self, K, m):
        """Hard argmax A has shape (m, K)."""
        comp = LinearCompressor(K=K, m=m, is_directed=False)
        comp.eval()
        A_hard = comp._get_A_hard(comp.W, dtype=torch.float64)
        assert A_hard.shape == (m, K)


# ---------------------------------------------------------------------------
# 2. Admissibility on Random Graphs (hypothesis-driven)
# ---------------------------------------------------------------------------


class TestAdmissibilityOnRandomGraphs:
    """Property-based tests: heuristics never exceed true shortest-path distance."""

    @given(
        seed=st.integers(0, 10000),
        n=st.integers(6, 12),
        extra=st.integers(3, 10),
    )
    @settings(max_examples=25, deadline=30000)
    def test_alt_admissible_undirected(self, seed, n, extra):
        """ALT heuristic never exceeds true shortest-path distance."""
        graph = make_random_connected_graph(n, extra, seed, is_directed=False)
        K = min(3, n - 1)

        rng = torch.Generator()
        rng.manual_seed(seed % 1000)
        teacher = alt_preprocess(graph, K, rng=rng)
        h = make_alt_heuristic(teacher)

        # Compute APSP
        sp_matrix = graph_to_scipy(graph)
        dist = shortest_path(sp_matrix, directed=False)

        for u in range(n):
            for v in range(n):
                if u != v and np.isfinite(dist[u, v]):
                    assert h(u, v) <= dist[u, v] + 1e-6, (
                        f"ALT inadmissible: h({u},{v})={h(u, v)} > d={dist[u, v]}"
                    )

    @given(
        seed=st.integers(0, 10000),
        n=st.integers(6, 12),
        extra=st.integers(3, 10),
    )
    @settings(max_examples=20, deadline=60000)
    def test_linear_admissible_undirected(self, seed, n, extra):
        """LinearCompressor heuristic never exceeds true shortest-path distance."""
        torch.manual_seed(seed)
        graph = make_random_connected_graph(n, extra, seed, is_directed=False)
        K = min(4, n - 1)
        m = min(3, K)

        rng = torch.Generator()
        rng.manual_seed(seed % 1000)
        teacher = alt_preprocess(graph, K, rng=rng)

        comp = LinearCompressor(K=K, m=m, is_directed=False)
        comp.eval()

        d_out_t = teacher.d_out.t()  # (V, K)
        with torch.no_grad():
            y = comp(d_out_t)

        h = make_linear_heuristic(y, y, is_directed=False)

        # APSP
        dist = shortest_path(graph_to_scipy(graph), directed=False)

        for u in range(n):
            for v in range(n):
                if u != v and np.isfinite(dist[u, v]):
                    assert h(u, v) <= dist[u, v] + 1e-6, (
                        f"Linear inadmissible: h({u},{v})={h(u, v)} > d={dist[u, v]}"
                    )

    @given(
        seed=st.integers(0, 10000),
        n=st.integers(6, 12),
        extra=st.integers(3, 10),
    )
    @settings(max_examples=20, deadline=60000)
    def test_aac_admissible_undirected(self, seed, n, extra):
        """make_aac_heuristic never exceeds true shortest-path distance."""
        torch.manual_seed(seed)
        graph = make_random_connected_graph(n, extra, seed, is_directed=False)
        K = min(4, n - 1)
        m = min(3, K)

        rng = torch.Generator()
        rng.manual_seed(seed % 1000)
        teacher = alt_preprocess(graph, K, rng=rng)

        comp = LinearCompressor(K=K, m=m, is_directed=False)
        comp.eval()

        d_out_t = teacher.d_out.t()  # (V, K)
        with torch.no_grad():
            y = comp(d_out_t)

        h = make_aac_heuristic(y, is_directed=False)

        # APSP
        dist = shortest_path(graph_to_scipy(graph), directed=False)

        for u in range(n):
            for v in range(n):
                if u != v and np.isfinite(dist[u, v]):
                    assert h(u, v) <= dist[u, v] + 1e-6, (
                        f"AAC inadmissible: h({u},{v})={h(u, v)} > d={dist[u, v]}"
                    )

    @given(
        seed=st.integers(0, 10000),
        n=st.integers(6, 12),
        extra=st.integers(3, 10),
    )
    @settings(max_examples=15, deadline=60000)
    def test_alt_admissible_directed(self, seed, n, extra):
        """ALT heuristic on directed graphs never exceeds true shortest-path distance."""
        graph = make_random_connected_graph(n, extra, seed, is_directed=True)
        K = min(3, n - 1)

        rng = torch.Generator()
        rng.manual_seed(seed % 1000)
        teacher = alt_preprocess(graph, K, rng=rng)
        h = make_alt_heuristic(teacher)

        # Compute APSP
        dist = shortest_path(graph_to_scipy(graph), directed=True)

        for u in range(n):
            for v in range(n):
                if u != v and np.isfinite(dist[u, v]):
                    assert h(u, v) <= dist[u, v] + 1e-6, (
                        f"ALT directed inadmissible: h({u},{v})={h(u, v)} > d={dist[u, v]}"
                    )


# ---------------------------------------------------------------------------
# 3. Substochastic Admissibility
# ---------------------------------------------------------------------------


class TestSubstochasticAdmissibility:
    """Row-substochastic A (rows sum <= 1) preserves max(A @ delta) <= max(delta)."""

    @given(
        K=st.integers(3, 8),
        m=st.integers(2, 5),
        seed=st.integers(0, 10000),
    )
    @settings(max_examples=30, deadline=10000)
    def test_substochastic_remains_admissible(self, K, m, seed):
        """Row-substochastic A preserves max(A @ delta) <= max(delta)."""
        rng = np.random.RandomState(seed)

        # Random substochastic matrix: start row-stochastic, scale rows down
        A = rng.dirichlet(np.ones(K), size=m)  # row-stochastic (m, K)
        A *= rng.uniform(0.5, 1.0, size=(m, 1))  # scale rows to substochastic

        # Verify substochastic property
        row_sums = A.sum(axis=1)
        assert np.all(row_sums <= 1.0 + 1e-10), f"Not substochastic: {row_sums}"

        # Random non-negative delta vector
        delta = rng.uniform(0, 100, size=K)

        result = A @ delta
        assert np.max(result) <= np.max(delta) + 1e-10, (
            f"Substochastic violated: max(A @ delta)={np.max(result)} > max(delta)={np.max(delta)}"
        )

    @given(
        K=st.integers(3, 8),
        m=st.integers(2, 5),
        seed=st.integers(0, 10000),
    )
    @settings(max_examples=30, deadline=10000)
    def test_stochastic_convex_combination_bound(self, K, m, seed):
        """Row-stochastic A: max(A @ delta) <= max(delta) for non-negative delta."""
        rng = np.random.RandomState(seed)

        # Row-stochastic matrix
        A = rng.dirichlet(np.ones(K), size=m)  # (m, K), rows sum to 1

        # Random non-negative delta vector
        delta = rng.uniform(0, 100, size=K)

        result = A @ delta
        assert np.max(result) <= np.max(delta) + 1e-10, (
            f"Stochastic violated: max(A @ delta)={np.max(result)} > max(delta)={np.max(delta)}"
        )

    @given(
        K=st.integers(3, 8),
        m=st.integers(2, 5),
        seed=st.integers(0, 10000),
    )
    @settings(max_examples=30, deadline=10000)
    def test_one_hot_selection_preserves_max(self, K, m, seed):
        """One-hot rows (hard selection): max(A @ delta) = delta[selected] <= max(delta)."""
        rng = np.random.RandomState(seed)

        # One-hot matrix: each row selects one column
        A = np.zeros((m, K))
        for i in range(m):
            j = rng.randint(0, K)
            A[i, j] = 1.0

        # Random delta (can be negative for directed heuristic differences)
        delta = rng.uniform(-50, 100, size=K)

        result = A @ delta
        # Each result[i] = delta[selected_j], so max(result) <= max(delta)
        assert np.max(result) <= np.max(delta) + 1e-10, (
            f"One-hot violated: max(A @ delta)={np.max(result)} > max(delta)={np.max(delta)}"
        )
