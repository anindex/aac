"""Property-based admissibility tests using hypothesis.

Asserts that ALT, LinearCompressor, and AAC heuristics never exceed the true
shortest-path distance on random connected graphs (undirected and directed).
The property is checked across many random graphs via hypothesis to catch
implementations that drift inadmissible on edge cases.
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.sparse.csgraph import shortest_path

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.compression.smooth import make_aac_heuristic
from aac.graphs.convert import edges_to_graph, graph_to_scipy


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
            # Reverse edge for strong connectivity in the directed case.
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
            # Reverse for the extra edges too, preserving strong connectivity.
            sources.append(v)
            targets.append(u)
            weights.append(rng.uniform(1.0, 10.0))

    src_t = torch.tensor(sources, dtype=torch.int64)
    tgt_t = torch.tensor(targets, dtype=torch.int64)
    wgt_t = torch.tensor(weights, dtype=torch.float64)

    return edges_to_graph(src_t, tgt_t, wgt_t, num_nodes, is_directed=is_directed)


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
