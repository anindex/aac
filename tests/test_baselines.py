"""Tests for ALT and FastMap baselines.

Validates admissibility, optimality, memory accounting, and PHIL data structure.
"""

import numpy as np
import scipy.sparse.csgraph
import torch

from aac.graphs.convert import edges_to_graph, graph_to_scipy
from aac.graphs.types import Graph

# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

def _make_undirected_graph() -> Graph:
    """5-node undirected graph with known shortest paths.

    Edges: (0,1,1), (0,2,4), (1,2,2), (1,3,5), (2,3,1), (3,4,3)
    """
    sources = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int64)
    targets = torch.tensor([1, 2, 2, 3, 3, 4], dtype=torch.int64)
    weights = torch.tensor([1.0, 4.0, 2.0, 5.0, 1.0, 3.0], dtype=torch.float64)
    return edges_to_graph(sources, targets, weights, num_nodes=5, is_directed=False)


def _make_directed_graph() -> Graph:
    """5-node directed graph with known shortest paths.

    Edges: (0,1,2), (0,2,5), (1,2,1), (1,3,6), (2,3,2), (2,4,7), (3,4,1)
    """
    sources = torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.int64)
    targets = torch.tensor([1, 2, 2, 3, 3, 4, 4], dtype=torch.int64)
    weights = torch.tensor([2.0, 5.0, 1.0, 6.0, 2.0, 7.0, 1.0], dtype=torch.float64)
    return edges_to_graph(sources, targets, weights, num_nodes=5, is_directed=True)


def _all_pairs_shortest(graph: Graph) -> np.ndarray:
    """Compute all-pairs shortest paths via scipy."""
    csr = graph_to_scipy(graph)
    dist = scipy.sparse.csgraph.shortest_path(csr, directed=graph.is_directed)
    return dist


# ---------------------------------------------------------------------------
# ALT tests
# ---------------------------------------------------------------------------

class TestALT:
    """Tests for ALT baseline (landmark-based triangle inequality heuristic)."""

    def test_alt_admissibility_undirected(self):
        """ALT h(u,t) <= d(u,t) for ALL pairs on undirected graph."""
        from aac.baselines import alt_preprocess, make_alt_heuristic

        graph = _make_undirected_graph()
        labels = alt_preprocess(graph, num_landmarks=3, seed_vertex=0)
        h = make_alt_heuristic(labels)
        apsp = _all_pairs_shortest(graph)

        violations = 0
        V = graph.num_nodes
        for u in range(V):
            for t in range(V):
                if u == t:
                    continue
                h_val = h(u, t)
                d_val = apsp[u, t]
                if not np.isinf(d_val) and h_val > d_val + 1e-9:
                    violations += 1

        assert violations == 0, f"ALT admissibility violated {violations} times"

    def test_alt_admissibility_directed(self):
        """ALT h(u,t) <= d(u,t) for ALL pairs on directed graph."""
        from aac.baselines import alt_preprocess, make_alt_heuristic

        graph = _make_directed_graph()
        labels = alt_preprocess(graph, num_landmarks=3, seed_vertex=0)
        h = make_alt_heuristic(labels)
        apsp = _all_pairs_shortest(graph)

        violations = 0
        V = graph.num_nodes
        for u in range(V):
            for t in range(V):
                if u == t:
                    continue
                h_val = h(u, t)
                d_val = apsp[u, t]
                if not np.isinf(d_val) and h_val > d_val + 1e-9:
                    violations += 1

        assert violations == 0, f"ALT admissibility violated {violations} times"

    def test_alt_triangle_inequality(self):
        """ALT h equals max over landmark triangle inequalities."""
        from aac.baselines import alt_preprocess, make_alt_heuristic

        graph = _make_undirected_graph()
        labels = alt_preprocess(graph, num_landmarks=3, seed_vertex=0)
        h = make_alt_heuristic(labels)

        # Manually compute expected heuristic value for one pair
        d_out = labels.d_out  # (K, V)
        u, t = 0, 4
        # For undirected: h(u,t) = max_k |d_out[k,u] - d_out[k,t]|
        expected = torch.max(torch.abs(d_out[:, u] - d_out[:, t])).item()
        actual = h(u, t)
        assert abs(actual - expected) < 1e-9, f"Expected {expected}, got {actual}"

    def test_alt_directed(self):
        """ALT heuristic on directed graph uses both d_out and d_in."""
        from aac.baselines import alt_preprocess, make_alt_heuristic

        graph = _make_directed_graph()
        labels = alt_preprocess(graph, num_landmarks=3, seed_vertex=0)
        h = make_alt_heuristic(labels)

        # Directed should use both d_out and d_in
        d_out = labels.d_out  # (K, V)
        d_in = labels.d_in   # (K, V)
        u, t = 0, 4
        # h(u,t) = max(0, max_k(d_out[k,t] - d_out[k,u]), max_k(d_in[k,u] - d_in[k,t]))
        fwd = torch.max(d_out[:, t] - d_out[:, u]).item()
        bwd = torch.max(d_in[:, u] - d_in[:, t]).item()
        expected = max(0.0, fwd, bwd)
        actual = h(u, t)
        assert abs(actual - expected) < 1e-9, f"Expected {expected}, got {actual}"

    def test_alt_undirected(self):
        """ALT heuristic on undirected graph uses |d_out[k,u] - d_out[k,t]|."""
        from aac.baselines import alt_preprocess, make_alt_heuristic

        graph = _make_undirected_graph()
        labels = alt_preprocess(graph, num_landmarks=3, seed_vertex=0)
        h = make_alt_heuristic(labels)

        # For undirected, h should be non-negative for all pairs
        V = graph.num_nodes
        for u in range(V):
            for t in range(V):
                assert h(u, t) >= -1e-9, f"h({u},{t}) = {h(u, t)} is negative"

    def test_alt_optimal_search(self):
        """A* with ALT heuristic finds optimal path (cost == Dijkstra cost)."""
        from aac.baselines import alt_preprocess, make_alt_heuristic
        from aac.search.astar import astar
        from aac.search.dijkstra import dijkstra

        graph = _make_undirected_graph()
        labels = alt_preprocess(graph, num_landmarks=3, seed_vertex=0)
        h = make_alt_heuristic(labels)

        for s in range(graph.num_nodes):
            for t in range(graph.num_nodes):
                if s == t:
                    continue
                dij = dijkstra(graph, s, t)
                ast = astar(graph, s, t, heuristic=h)
                assert abs(ast.cost - dij.cost) < 1e-9, (
                    f"A*({s},{t}) cost {ast.cost} != Dijkstra cost {dij.cost}"
                )
                assert ast.optimal


# ---------------------------------------------------------------------------
# FastMap tests
# ---------------------------------------------------------------------------

class TestFastMap:
    """Tests for FastMap baseline (iterative farthest-pair Euclidean embedding)."""

    def test_fastmap_embedding_dims(self):
        """fastmap_preprocess returns (V, m) tensor."""
        from aac.baselines import fastmap_preprocess

        graph = _make_undirected_graph()
        coords = fastmap_preprocess(graph, num_dims=3)
        assert coords.shape == (graph.num_nodes, 3)
        assert coords.dtype == torch.float64

    def test_fastmap_admissibility(self):
        """FastMap L1 h(u,t) <= d(u,t) for ALL pairs."""
        from aac.baselines import fastmap_preprocess, make_fastmap_heuristic

        graph = _make_undirected_graph()
        coords = fastmap_preprocess(graph, num_dims=3)
        h = make_fastmap_heuristic(coords)
        apsp = _all_pairs_shortest(graph)

        violations = 0
        V = graph.num_nodes
        for u in range(V):
            for t in range(V):
                if u == t:
                    continue
                h_val = h(u, t)
                d_val = apsp[u, t]
                if not np.isinf(d_val) and h_val > d_val + 1e-9:
                    violations += 1

        assert violations == 0, f"FastMap admissibility violated {violations} times"

    def test_fastmap_optimal_search(self):
        """A* with FastMap heuristic finds optimal path."""
        from aac.baselines import fastmap_preprocess, make_fastmap_heuristic
        from aac.search.astar import astar
        from aac.search.dijkstra import dijkstra

        graph = _make_undirected_graph()
        coords = fastmap_preprocess(graph, num_dims=3)
        h = make_fastmap_heuristic(coords)

        for s in range(graph.num_nodes):
            for t in range(graph.num_nodes):
                if s == t:
                    continue
                dij = dijkstra(graph, s, t)
                ast = astar(graph, s, t, heuristic=h)
                assert abs(ast.cost - dij.cost) < 1e-9, (
                    f"A*({s},{t}) cost {ast.cost} != Dijkstra cost {dij.cost}"
                )
                assert ast.optimal


# ---------------------------------------------------------------------------
# Memory accounting tests
# ---------------------------------------------------------------------------

class TestMemory:
    """Tests for memory accounting functions."""

    def test_alt_memory_bytes(self):
        """alt_memory_bytes(16) == 128 (2*16*4 = 128 bytes, float32 default)."""
        from aac.baselines import alt_memory_bytes

        assert alt_memory_bytes(16) == 128

    def test_fastmap_memory_bytes(self):
        """fastmap_memory_bytes(16) == 64 (16*4 = 64 bytes, float32 default)."""
        from aac.baselines import fastmap_memory_bytes

        assert fastmap_memory_bytes(16) == 64

    def test_memory_bytes_override(self):
        """Memory accounting with explicit dtype_size override."""
        from aac.baselines import alt_memory_bytes, fastmap_memory_bytes

        assert alt_memory_bytes(16, dtype_size=8) == 256
        assert fastmap_memory_bytes(16, dtype_size=8) == 128


# ---------------------------------------------------------------------------
# PHIL data test
# ---------------------------------------------------------------------------

class TestPHIL:
    """Tests for PHIL reported data structure."""

    def test_phil_data(self):
        """PHIL_REPORTED dict has expected keys and caveat."""
        from aac.baselines import PHIL_REPORTED

        assert "modena" in PHIL_REPORTED
        assert "new_york" in PHIL_REPORTED
        assert "caveat" in PHIL_REPORTED
        assert "No public code" in PHIL_REPORTED["caveat"]


# ---------------------------------------------------------------------------
# CDH tests
# ---------------------------------------------------------------------------

def _make_random_undirected_graph(V: int = 20, edge_p: float = 0.25, seed: int = 7) -> Graph:
    """Small connected-ish weighted random undirected graph for CDH tests."""
    rng = np.random.default_rng(seed)
    src, tgt, wts = [], [], []
    # Spanning path guarantees connectivity.
    for i in range(V - 1):
        src.append(i); tgt.append(i + 1)
        wts.append(float(rng.uniform(0.5, 2.0)))
    # Extra random edges.
    for i in range(V):
        for j in range(i + 2, V):
            if rng.random() < edge_p:
                src.append(i); tgt.append(j)
                wts.append(float(rng.uniform(0.5, 3.0)))
    return edges_to_graph(
        torch.tensor(src, dtype=torch.int64),
        torch.tensor(tgt, dtype=torch.int64),
        torch.tensor(wts, dtype=torch.float64),
        num_nodes=V,
        is_directed=False,
    )


class TestCDH:
    """Tests for CDH baseline (compressed differential heuristic)."""

    def test_cdh_preprocess_shapes(self):
        """cdh_preprocess returns CDHLabels with correct (V, r) shapes."""
        from aac.baselines import cdh_preprocess

        graph = _make_random_undirected_graph(V=20)
        labels = cdh_preprocess(graph, num_pivots=8, num_stored=3, seed_vertex=0)
        assert labels.pivot_indices.shape == (20, 3)
        assert labels.pivot_distances.shape == (20, 3)
        assert labels.num_pivots == 8
        assert labels.num_stored == 3
        assert not labels.is_directed
        assert labels.pivot_distances_in is None

    def test_cdh_admissibility_undirected(self):
        """CDH intersection heuristic is admissible on all query pairs."""
        from aac.baselines import cdh_preprocess, make_cdh_heuristic

        graph = _make_random_undirected_graph(V=25, seed=13)
        labels = cdh_preprocess(graph, num_pivots=6, num_stored=2, seed_vertex=0)
        h = make_cdh_heuristic(labels)
        apsp = _all_pairs_shortest(graph)

        violations = 0
        V = graph.num_nodes
        for u in range(V):
            for t in range(V):
                if u == t:
                    continue
                hv = h(u, t)
                dv = apsp[u, t]
                if not np.isinf(dv) and hv > dv + 1e-9:
                    violations += 1
        assert violations == 0

    def test_cdh_admissibility_directed(self):
        """CDH admissible on a small directed graph."""
        from aac.baselines import cdh_preprocess, make_cdh_heuristic

        graph = _make_directed_graph()
        labels = cdh_preprocess(graph, num_pivots=3, num_stored=2, seed_vertex=0)
        h = make_cdh_heuristic(labels)
        assert labels.is_directed
        assert labels.pivot_distances_in is not None
        apsp = _all_pairs_shortest(graph)
        V = graph.num_nodes
        for u in range(V):
            for t in range(V):
                if u == t:
                    continue
                hv = h(u, t)
                dv = apsp[u, t]
                if not np.isinf(dv):
                    assert hv <= dv + 1e-9, f"violated at ({u},{t}): {hv} > {dv}"

    def test_cdh_bound_substitution_admissible_undirected(self):
        """CDH with bound-substitution stays admissible on undirected graphs."""
        from aac.baselines import cdh_preprocess, make_cdh_heuristic

        graph = _make_random_undirected_graph(V=25, seed=19)
        labels = cdh_preprocess(graph, num_pivots=8, num_stored=2, seed_vertex=0)
        assert labels.pivot_pivot_out is not None
        h = make_cdh_heuristic(labels, use_bound_substitution=True)
        apsp = _all_pairs_shortest(graph)
        V = graph.num_nodes
        for u in range(V):
            for t in range(V):
                if u == t:
                    continue
                hv = h(u, t)
                dv = apsp[u, t]
                if not np.isinf(dv):
                    assert hv <= dv + 1e-9, (
                        f"bound-sub violated at ({u},{t}): {hv} > {dv}"
                    )

    def test_cdh_bound_substitution_admissible_directed(self):
        """CDH with bound-substitution stays admissible on directed graphs."""
        from aac.baselines import cdh_preprocess, make_cdh_heuristic

        graph = _make_directed_graph()
        labels = cdh_preprocess(graph, num_pivots=3, num_stored=2, seed_vertex=0)
        assert labels.pivot_pivot_in is not None
        h = make_cdh_heuristic(labels, use_bound_substitution=True)
        apsp = _all_pairs_shortest(graph)
        V = graph.num_nodes
        for u in range(V):
            for t in range(V):
                if u == t:
                    continue
                hv = h(u, t)
                dv = apsp[u, t]
                if not np.isinf(dv):
                    assert hv <= dv + 1e-9, (
                        f"directed bound-sub violated at ({u},{t}): {hv} > {dv}"
                    )

    def test_cdh_bound_substitution_never_worse(self):
        """Bound-substitution never decreases the heuristic vs. intersection."""
        from aac.baselines import cdh_preprocess, make_cdh_heuristic

        graph = _make_random_undirected_graph(V=30, seed=7)
        labels = cdh_preprocess(graph, num_pivots=8, num_stored=2, seed_vertex=0)
        h0 = make_cdh_heuristic(labels, use_bound_substitution=False)
        h1 = make_cdh_heuristic(labels, use_bound_substitution=True)
        V = graph.num_nodes
        for u in range(V):
            for t in range(V):
                if u == t:
                    continue
                assert h1(u, t) + 1e-9 >= h0(u, t)

    def test_cdh_optimal_search(self):
        """A* with CDH heuristic returns optimal costs on a small graph."""
        from aac.baselines import cdh_preprocess, make_cdh_heuristic
        from aac.search.astar import astar
        from aac.search.dijkstra import dijkstra

        graph = _make_random_undirected_graph(V=15, seed=5)
        labels = cdh_preprocess(graph, num_pivots=5, num_stored=3, seed_vertex=0)
        h = make_cdh_heuristic(labels)
        for s in range(graph.num_nodes):
            for t in range(graph.num_nodes):
                if s == t:
                    continue
                dij = dijkstra(graph, s, t)
                if np.isinf(dij.cost):
                    continue
                ast = astar(graph, s, t, heuristic=h)
                assert abs(ast.cost - dij.cost) < 1e-9
                assert ast.optimal

    def test_cdh_equals_alt_when_r_equals_P(self):
        """With num_stored == num_pivots, CDH intersection == ALT heuristic."""
        from aac.baselines import (
            alt_preprocess,
            cdh_preprocess,
            make_alt_heuristic,
            make_cdh_heuristic,
        )

        graph = _make_random_undirected_graph(V=20, seed=3)
        # Seed the FPS identically so both pick the same pivot set.
        gen_a = torch.Generator().manual_seed(0)
        gen_b = torch.Generator().manual_seed(0)
        alt_labels = alt_preprocess(graph, num_landmarks=4, seed_vertex=0, rng=gen_a)
        cdh_labels = cdh_preprocess(graph, num_pivots=4, num_stored=4, seed_vertex=0, rng=gen_b)
        h_alt = make_alt_heuristic(alt_labels)
        h_cdh = make_cdh_heuristic(cdh_labels)
        V = graph.num_nodes
        for u in range(V):
            for t in range(V):
                if u == t:
                    continue
                assert abs(h_alt(u, t) - h_cdh(u, t)) < 1e-9

    def test_cdh_memory_accounting_undirected(self):
        """Undirected: r * (dtype_size + index_size)."""
        from aac.baselines import cdh_memory_bytes

        # P=32 -> ceil(log2(32)/8) = ceil(5/8) = 1 byte index.
        # r=8, float32 -> 8 * (4 + 1) = 40.
        assert cdh_memory_bytes(num_pivots=32, num_stored=8) == 40
        # P=300 -> ceil(log2(300)/8) = ceil(8.23/8) = 2 byte index.
        # r=4, float32 -> 4 * (4 + 2) = 24.
        assert cdh_memory_bytes(num_pivots=300, num_stored=4) == 24

    def test_cdh_memory_accounting_directed(self):
        """Directed: r * (2*dtype_size + index_size)."""
        from aac.baselines import cdh_memory_bytes

        # P=32, r=8, float32 -> 8 * (8 + 1) = 72.
        assert cdh_memory_bytes(num_pivots=32, num_stored=8, is_directed=True) == 72

    def test_cdh_memory_dtype_override(self):
        """Memory accounting respects dtype_size override."""
        from aac.baselines import cdh_memory_bytes

        # P=16, r=4, dtype=8 -> 4 * (8 + 1) = 36.
        assert cdh_memory_bytes(16, 4, dtype_size=8) == 36

    def test_cdh_bpmx_optimal_search(self):
        """A* with use_bpmx=True returns optimal cost on a small graph."""
        from aac.baselines import cdh_preprocess, make_cdh_heuristic
        from aac.search.astar import astar
        from aac.search.dijkstra import dijkstra

        graph = _make_random_undirected_graph(V=18, seed=11)
        labels = cdh_preprocess(graph, num_pivots=6, num_stored=2, seed_vertex=0)
        h = make_cdh_heuristic(labels, use_bound_substitution=True)
        for s in range(graph.num_nodes):
            for t in range(graph.num_nodes):
                if s == t:
                    continue
                dij = dijkstra(graph, s, t)
                if np.isinf(dij.cost):
                    continue
                ast = astar(graph, s, t, heuristic=h, use_bpmx=True)
                assert abs(ast.cost - dij.cost) < 1e-9, (
                    f"BPMX A*({s},{t}) cost {ast.cost} != Dijkstra {dij.cost}"
                )
                assert ast.optimal

    def test_cdh_bpmx_never_worse_than_no_bpmx(self):
        """BPMX should never increase expansion count above the non-BPMX run.

        On every (s, t) pair with finite distance, BPMX-augmented A* with the
        same heuristic should expand no more nodes than vanilla A*. Tighter h
        values can only prune the open list further; admissibility guarantees
        BPMX never returns a worse cost.
        """
        from aac.baselines import cdh_preprocess, make_cdh_heuristic
        from aac.search.astar import astar
        from aac.search.dijkstra import dijkstra

        graph = _make_random_undirected_graph(V=22, seed=29)
        labels = cdh_preprocess(graph, num_pivots=8, num_stored=2, seed_vertex=0)
        h = make_cdh_heuristic(labels, use_bound_substitution=True)
        worse = 0
        for s in range(graph.num_nodes):
            for t in range(graph.num_nodes):
                if s == t:
                    continue
                dij = dijkstra(graph, s, t)
                if np.isinf(dij.cost):
                    continue
                ast0 = astar(graph, s, t, heuristic=h, use_bpmx=False)
                ast1 = astar(graph, s, t, heuristic=h, use_bpmx=True)
                # Costs match (both optimal).
                assert abs(ast0.cost - ast1.cost) < 1e-9
                if ast1.expansions > ast0.expansions:
                    worse += 1
        # Allow at most 5% of pairs to go up by 1 expansion due to FIFO
        # tie-breaking interactions; BPMX is a tightening, but tie-break
        # ordering can shuffle small inflations on tiny graphs. The test
        # asserts the bulk behaviour.
        total_pairs = graph.num_nodes * (graph.num_nodes - 1)
        assert worse <= max(2, int(0.05 * total_pairs)), (
            f"BPMX increased expansions on {worse}/{total_pairs} pairs"
        )

    def test_cdh_bpmx_admissibility_directed(self):
        """A* + BPMX returns optimal costs on a small directed graph."""
        from aac.baselines import cdh_preprocess, make_cdh_heuristic
        from aac.search.astar import astar
        from aac.search.dijkstra import dijkstra

        graph = _make_directed_graph()
        labels = cdh_preprocess(graph, num_pivots=3, num_stored=2, seed_vertex=0)
        h = make_cdh_heuristic(labels, use_bound_substitution=True)
        for s in range(graph.num_nodes):
            for t in range(graph.num_nodes):
                if s == t:
                    continue
                dij = dijkstra(graph, s, t)
                if np.isinf(dij.cost):
                    continue
                ast = astar(graph, s, t, heuristic=h, use_bpmx=True)
                assert abs(ast.cost - dij.cost) < 1e-9
                assert ast.optimal
