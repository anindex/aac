"""Paper invariant tests: verify each theorem/proposition against code.

These tests encode the paper's core theoretical guarantees as executable
assertions, testing properties that MUST hold regardless of graph structure,
training quality, or hyperparameters.

Reference: "Admissibility by Architecture: Learning A* Heuristics
via Row-Stochastic Label Compression"
"""

from __future__ import annotations

import pytest
import torch

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.convert import edges_to_graph
from aac.search.astar import astar

# ---------------------------------------------------------------------------
# Test graphs
# ---------------------------------------------------------------------------


@pytest.fixture
def directed_8():
    """8-node strongly connected directed graph with varied edge weights."""
    s = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 5, 6], dtype=torch.int64)
    t = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0, 3, 4, 5, 6, 0, 1], dtype=torch.int64)
    w = torch.tensor(
        [2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 1.0, 3.0, 7.0, 6.0, 4.0, 8.0, 9.0, 2.0],
        dtype=torch.float64,
    )
    return edges_to_graph(s, t, w, 8, is_directed=True)


@pytest.fixture
def undirected_8():
    """8-node undirected graph with varied edge weights."""
    s = torch.tensor([0, 0, 1, 1, 2, 3, 3, 4, 5, 6], dtype=torch.int64)
    t = torch.tensor([1, 3, 2, 4, 3, 4, 5, 5, 6, 7], dtype=torch.int64)
    w = torch.tensor(
        [2.0, 7.0, 3.0, 6.0, 1.0, 4.0, 2.0, 5.0, 1.0, 3.0],
        dtype=torch.float64,
    )
    return edges_to_graph(s, t, w, 8, is_directed=False)


def _all_pairs_distances(graph):
    """Compute all-pairs shortest path distances via repeated Dijkstra."""
    V = graph.num_nodes
    anchors = torch.arange(V, dtype=torch.int64)
    tl = compute_teacher_labels(graph, anchors, use_gpu=False)
    return tl.d_out  # (V, V) for directed, symmetric for undirected


# ---------------------------------------------------------------------------
# T-INV-01: Row-stochastic property
# Theorem 2: Selection matrix A has non-negative entries, each row sums to 1
# ---------------------------------------------------------------------------


class TestRowStochasticProperty:
    """Verify row-stochastic invariant in both train and eval modes."""

    def test_eval_mode_one_hot(self, directed_8):
        """Eval mode produces one-hot rows (special case of row-stochastic)."""
        torch.manual_seed(42)
        K, m = 8, 4
        comp = LinearCompressor(K=K, m=m, is_directed=True).double()
        comp.eval()

        A_fwd = comp._get_A_hard(comp.W_fwd)
        A_bwd = comp._get_A_hard(comp.W_bwd)

        for name, A in [("fwd", A_fwd), ("bwd", A_bwd)]:
            assert (A >= 0).all(), f"{name}: negative entries"
            row_sums = A.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums)), (
                f"{name}: row sums != 1: {row_sums}"
            )
            # One-hot: exactly one 1 per row
            assert (A.max(dim=-1).values == 1.0).all(), f"{name}: not one-hot"
            assert (A.sum(dim=-1) == 1.0).all(), f"{name}: row sum != 1"

    def test_directed_split_floor(self):
        """Directed split uses floor (int()) not round() for m_fwd.

        Paper says m_fwd = floor(m/2). With m_fwd_ratio=0.5:
        - Even m: both agree (m=8 -> m_fwd=4)
        - Odd m: int(m*0.5)=floor gives m_fwd < m_bwd (asymmetric)
        Admissibility is preserved regardless, but code must match paper.
        """
        for m in [3, 4, 5, 6, 7, 8, 9, 11, 16]:
            comp = LinearCompressor(K=16, m=m, is_directed=True)
            expected_fwd = max(1, int(m * 0.5))
            expected_bwd = m - expected_fwd
            assert comp.m_fwd == expected_fwd, (
                f"m={m}: m_fwd={comp.m_fwd} != expected {expected_fwd}"
            )
            assert comp.m_bwd == expected_bwd, (
                f"m={m}: m_bwd={comp.m_bwd} != expected {expected_bwd}"
            )
            assert comp.m_fwd + comp.m_bwd == m, (
                f"m={m}: m_fwd + m_bwd = {comp.m_fwd + comp.m_bwd} != {m}"
            )

    def test_gumbel_hard_extreme_tau(self):
        """Gumbel-softmax hard=True produces one-hot at extreme temperatures."""
        torch.manual_seed(42)
        comp = LinearCompressor(K=16, m=4, is_directed=False).double()
        comp.train()

        for tau in [0.01, 0.1, 1.0, 5.0, 10.0]:
            for _ in range(5):
                A = comp._get_A_soft(comp.W, tau=tau)
                assert (A >= 0).all(), f"Negative at tau={tau}"
                row_sums = A.sum(dim=-1)
                assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), (
                    f"Row sums != 1 at tau={tau}: {row_sums}"
                )
                # hard=True guarantees one-hot in forward
                assert (A.max(dim=-1).values == 1.0).all(), (
                    f"Not one-hot at tau={tau}"
                )

    def test_train_mode_gumbel_one_hot(self, undirected_8):
        """Training mode (hard=True) also produces one-hot rows."""
        torch.manual_seed(42)
        K, m = 8, 4
        comp = LinearCompressor(K=K, m=m, is_directed=False).double()
        comp.train()

        for _ in range(10):  # Multiple Gumbel noise realizations
            A = comp._get_A_soft(comp.W, tau=1.0)
            assert (A >= 0).all(), "negative entries in Gumbel-softmax"
            row_sums = A.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), (
                f"row sums != 1: {row_sums}"
            )


# ---------------------------------------------------------------------------
# T-INV-02: Hard-argmax equals ALT on selected subset
# Proposition 1: Deployed heuristic = ALT restricted to selected landmarks
# ---------------------------------------------------------------------------


class TestHardArgmaxEqualsALTSubset:
    """At inference, AAC heuristic must equal ALT on the selected landmark subset."""

    def test_directed(self, directed_8):
        torch.manual_seed(42)
        K = 8
        m = 4  # 2 fwd + 2 bwd
        anchors = torch.arange(K, dtype=torch.int64)
        tl = compute_teacher_labels(directed_8, anchors, use_gpu=False)

        comp = LinearCompressor(K=K, m=m, is_directed=True).double()
        comp.eval()

        # Get selected landmarks
        sel = comp.selected_landmarks()
        fwd_idx = sel["fwd"]
        bwd_idx = sel["bwd"]

        d_out_t = tl.d_out.t().to(torch.float64)  # (V, K)
        d_in_t = tl.d_in.t().to(torch.float64)

        with torch.no_grad():
            y_fwd, y_bwd = comp(d_out_t, d_in_t)

        h_aac = make_linear_heuristic(y_fwd, y_bwd, is_directed=True)

        # Build ALT heuristic restricted to the same landmark subset
        V = directed_8.num_nodes
        for s in range(V):
            for t in range(V):
                if s == t:
                    continue
                h_val = h_aac(s, t)

                # Manually compute ALT on the selected subset
                fwd_terms = [
                    (tl.d_out[fwd_idx[i], t] - tl.d_out[fwd_idx[i], s]).item()
                    for i in range(len(fwd_idx))
                ]
                bwd_terms = [
                    (tl.d_in[bwd_idx[i], s] - tl.d_in[bwd_idx[i], t]).item()
                    for i in range(len(bwd_idx))
                ]
                expected = max(0.0, max(fwd_terms + bwd_terms))

                assert abs(h_val - expected) < 1e-10, (
                    f"AAC h({s},{t})={h_val} != ALT-subset {expected}"
                )

    def test_undirected(self, undirected_8):
        torch.manual_seed(42)
        K = 8
        m = 4
        anchors = torch.arange(K, dtype=torch.int64)
        tl = compute_teacher_labels(undirected_8, anchors, use_gpu=False)

        comp = LinearCompressor(K=K, m=m, is_directed=False).double()
        comp.eval()

        sel = comp.selected_landmarks()
        idx = sel["landmarks"]

        d_out_t = tl.d_out.t().to(torch.float64)
        with torch.no_grad():
            y = comp(d_out_t)

        h_aac = make_linear_heuristic(y, y, is_directed=False)

        V = undirected_8.num_nodes
        for s in range(V):
            for t in range(V):
                if s == t:
                    continue
                h_val = h_aac(s, t)
                terms = [
                    abs(tl.d_out[idx[i], s] - tl.d_out[idx[i], t]).item()
                    for i in range(len(idx))
                ]
                expected = max(0.0, max(terms))
                assert abs(h_val - expected) < 1e-10, (
                    f"AAC h({s},{t})={h_val} != ALT-subset {expected}"
                )


# ---------------------------------------------------------------------------
# T-INV-03: Identity case recovery
# Proposition 1 corollary: m=K0 with identity-like W recovers exact ALT
# ---------------------------------------------------------------------------


class TestIdentityCaseRecovery:
    """When m=K (or m=2K for directed) with W=large diagonal, AAC = ALT exactly."""

    def test_undirected_identity(self, undirected_8):
        torch.manual_seed(42)
        K = 8
        m = K  # Full recovery
        anchors = torch.arange(K, dtype=torch.int64)
        tl = compute_teacher_labels(undirected_8, anchors, use_gpu=False)

        comp = LinearCompressor(K=K, m=m, is_directed=False).double()
        # Set W to large diagonal so argmax selects each landmark exactly once
        with torch.no_grad():
            comp.W.fill_(-100.0)
            for i in range(m):
                comp.W[i, i] = 100.0
        comp.eval()

        d_out_t = tl.d_out.t().to(torch.float64)
        with torch.no_grad():
            y = comp(d_out_t)

        h_aac = make_linear_heuristic(y, y, is_directed=False)
        h_alt = make_alt_heuristic(tl)

        V = undirected_8.num_nodes
        for s in range(V):
            for t in range(V):
                assert abs(h_aac(s, t) - h_alt(s, t)) < 1e-10, (
                    f"Identity recovery failed: h_aac({s},{t})={h_aac(s,t)} "
                    f"!= h_alt={h_alt(s,t)}"
                )

    def test_directed_identity(self, directed_8):
        torch.manual_seed(42)
        K = 8
        m = 2 * K  # Full directed recovery (K fwd + K bwd)
        anchors = torch.arange(K, dtype=torch.int64)
        tl = compute_teacher_labels(directed_8, anchors, use_gpu=False)

        comp = LinearCompressor(K=K, m=m, is_directed=True).double()
        # Set both W_fwd and W_bwd to identity-like
        with torch.no_grad():
            comp.W_fwd.fill_(-100.0)
            comp.W_bwd.fill_(-100.0)
            for i in range(K):
                comp.W_fwd[i, i] = 100.0
                comp.W_bwd[i, i] = 100.0
        comp.eval()

        d_out_t = tl.d_out.t().to(torch.float64)
        d_in_t = tl.d_in.t().to(torch.float64)
        with torch.no_grad():
            y_fwd, y_bwd = comp(d_out_t, d_in_t)

        h_aac = make_linear_heuristic(y_fwd, y_bwd, is_directed=True)
        h_alt = make_alt_heuristic(tl)

        V = directed_8.num_nodes
        for s in range(V):
            for t in range(V):
                assert abs(h_aac(s, t) - h_alt(s, t)) < 1e-10, (
                    f"Directed identity recovery failed: h_aac({s},{t})="
                    f"{h_aac(s,t)} != h_alt={h_alt(s,t)}"
                )


# ---------------------------------------------------------------------------
# T-INV-04: Float32 storage safety
# Section 5.1: No admissibility violations with float32 storage
# ---------------------------------------------------------------------------


class TestFloat32StorageSafety:
    """Verify that casting labels from float64 to float32 does not create
    admissibility violations (h_compressed > d_true)."""

    def test_no_violations_undirected(self, undirected_8):
        torch.manual_seed(42)
        K = 8
        m = 4
        anchors = torch.arange(K, dtype=torch.int64)
        tl = compute_teacher_labels(undirected_8, anchors, use_gpu=False)
        dist = _all_pairs_distances(undirected_8)

        comp = LinearCompressor(K=K, m=m, is_directed=False).double()
        comp.eval()

        d_out_t = tl.d_out.t().to(torch.float64)
        with torch.no_grad():
            y_f64 = comp(d_out_t)

        # Cast to float32 (deployment)
        y_f32 = y_f64.float()
        h_f32 = make_linear_heuristic(y_f32, y_f32, is_directed=False)

        V = undirected_8.num_nodes
        violations = 0
        for s in range(V):
            for t in range(V):
                h_val = h_f32(s, t)
                d_val = dist[s, t].item()
                if h_val > d_val + 1e-6:
                    violations += 1

        assert violations == 0, f"Float32 violations: {violations}/{V*V}"

    def test_no_violations_directed(self, directed_8):
        torch.manual_seed(42)
        K = 8
        m = 4
        anchors = torch.arange(K, dtype=torch.int64)
        tl = compute_teacher_labels(directed_8, anchors, use_gpu=False)
        dist = _all_pairs_distances(directed_8)

        comp = LinearCompressor(K=K, m=m, is_directed=True).double()
        comp.eval()

        d_out_t = tl.d_out.t().to(torch.float64)
        d_in_t = tl.d_in.t().to(torch.float64)
        with torch.no_grad():
            y_fwd, y_bwd = comp(d_out_t, d_in_t)

        y_fwd_f32 = y_fwd.float()
        y_bwd_f32 = y_bwd.float()
        h_f32 = make_linear_heuristic(y_fwd_f32, y_bwd_f32, is_directed=True)

        V = directed_8.num_nodes
        violations = 0
        for s in range(V):
            for t in range(V):
                h_val = h_f32(s, t)
                d_val = dist[s, t].item()
                if d_val < 1e15 and h_val > d_val + 1e-6:
                    violations += 1

        assert violations == 0, f"Float32 violations: {violations}/{V*V}"


# ---------------------------------------------------------------------------
# T-INV-05: Consistency in all-finite regime
# Section 2.1: h(u,t) <= w(u,v) + h(v,t) for all edges (u,v)
# ---------------------------------------------------------------------------


class TestConsistencyAllFinite:
    """ALT and deployed AAC heuristics must be consistent when all
    landmark distances are finite."""

    def _check_consistency(self, graph, h_func):
        """Check h(u,t) <= w(u,v) + h(v,t) for all edges and targets."""
        V = graph.num_nodes
        crow = graph.crow_indices
        col = graph.col_indices
        vals = graph.values
        violations = 0

        for t in range(V):
            for u in range(V):
                start = crow[u].item()
                end = crow[u + 1].item()
                for idx in range(start, end):
                    v = col[idx].item()
                    w = vals[idx].item()
                    h_u = h_func(u, t)
                    h_v = h_func(v, t)
                    if h_u > w + h_v + 1e-10:
                        violations += 1
        return violations

    def test_alt_consistent_undirected(self, undirected_8):
        torch.manual_seed(42)
        tl = alt_preprocess(undirected_8, num_landmarks=8, seed_vertex=0)
        h_alt = make_alt_heuristic(tl)
        violations = self._check_consistency(undirected_8, h_alt)
        assert violations == 0, f"ALT consistency violations: {violations}"

    def test_aac_consistent_undirected(self, undirected_8):
        torch.manual_seed(42)
        K = 8
        m = 4
        anchors = torch.arange(K, dtype=torch.int64)
        tl = compute_teacher_labels(undirected_8, anchors, use_gpu=False)

        comp = LinearCompressor(K=K, m=m, is_directed=False).double()
        comp.eval()

        d_out_t = tl.d_out.t().to(torch.float64)
        with torch.no_grad():
            y = comp(d_out_t)

        h_aac = make_linear_heuristic(y, y, is_directed=False)
        violations = self._check_consistency(undirected_8, h_aac)
        assert violations == 0, f"AAC consistency violations: {violations}"

    def test_aac_consistent_directed(self, directed_8):
        torch.manual_seed(42)
        K = 8
        m = 4
        anchors = torch.arange(K, dtype=torch.int64)
        tl = compute_teacher_labels(directed_8, anchors, use_gpu=False)

        comp = LinearCompressor(K=K, m=m, is_directed=True).double()
        comp.eval()

        d_out_t = tl.d_out.t().to(torch.float64)
        d_in_t = tl.d_in.t().to(torch.float64)
        with torch.no_grad():
            y_fwd, y_bwd = comp(d_out_t, d_in_t)

        h_aac = make_linear_heuristic(y_fwd, y_bwd, is_directed=True)
        violations = self._check_consistency(directed_8, h_aac)
        assert violations == 0, f"AAC consistency violations: {violations}"


# ---------------------------------------------------------------------------
# T-INV-06: Admissibility preserved under directed masking
# Section 2.1: Masking unreachable landmarks preserves admissibility
# ---------------------------------------------------------------------------


class TestDirectedMaskingAdmissibility:
    """On a graph with some unreachable vertex-landmark pairs,
    masking (setting to sentinel) and excluding those terms must
    still produce an admissible heuristic."""

    def test_masking_preserves_admissibility(self):
        """Create a directed graph where some landmarks are not reachable
        from some vertices. Verify h <= d still holds."""
        # Graph: 0->1->2->3, plus 4->5 (disconnected component)
        s = torch.tensor([0, 1, 2, 4], dtype=torch.int64)
        t = torch.tensor([1, 2, 3, 5], dtype=torch.int64)
        w = torch.tensor([1.0, 2.0, 3.0, 1.0], dtype=torch.float64)
        graph = edges_to_graph(s, t, w, 6, is_directed=True)

        # Use all 6 vertices as landmarks
        anchors = torch.arange(6, dtype=torch.int64)
        tl = compute_teacher_labels(graph, anchors, use_gpu=False)

        # The ALT heuristic should handle sentinels internally
        h_alt = make_alt_heuristic(tl)

        # Compute true distances
        dist = _all_pairs_distances(graph)

        # Check admissibility for all reachable pairs
        V = 6
        for s_node in range(V):
            for t_node in range(V):
                d_val = dist[s_node, t_node].item()
                if d_val > 1e15:
                    continue  # unreachable pair
                h_val = h_alt(s_node, t_node)
                assert h_val <= d_val + 1e-10, (
                    f"Admissibility violation with masking: "
                    f"h({s_node},{t_node})={h_val} > d={d_val}"
                )


# ---------------------------------------------------------------------------
# T-INV-07: Covering radius bounds the gap
# Theorem 3: d(u,t) - h_ALT^S(u,t) <= 2 * r_m
# ---------------------------------------------------------------------------


class TestCoveringRadiusBound:
    """Verify that the gap between true distance and ALT heuristic
    on a landmark subset is bounded by 2 * covering_radius."""

    def test_covering_radius_bound_undirected(self, undirected_8):
        torch.manual_seed(42)
        V = undirected_8.num_nodes
        dist = _all_pairs_distances(undirected_8)

        # Select m=3 landmarks via FPS
        m = 3
        anchors = farthest_point_sampling(undirected_8, m, seed_vertex=0)
        tl = compute_teacher_labels(undirected_8, anchors, use_gpu=False)
        h_alt = make_alt_heuristic(tl)

        # Compute covering radius: r_m = max_v min_l d(v, l)
        # For undirected, d(v,l) = d(l,v) = dist[l,v]
        anchor_list = anchors.tolist()
        r_m = 0.0
        for v in range(V):
            min_dist_to_landmark = min(dist[l, v].item() for l in anchor_list)
            r_m = max(r_m, min_dist_to_landmark)

        # Check: d(u,t) - h_ALT^S(u,t) <= 2*r_m
        max_gap = 0.0
        for u in range(V):
            for t in range(V):
                d_val = dist[u, t].item()
                if d_val > 1e15:
                    continue
                h_val = h_alt(u, t)
                gap = d_val - h_val
                max_gap = max(max_gap, gap)
                assert gap <= 2.0 * r_m + 1e-10, (
                    f"Covering radius bound violated: "
                    f"gap({u},{t})={gap:.4f} > 2*r_m={2*r_m:.4f}"
                )

    def test_covering_radius_bound_directed(self, directed_8):
        torch.manual_seed(42)
        V = directed_8.num_nodes
        dist = _all_pairs_distances(directed_8)

        m = 3
        anchors = farthest_point_sampling(directed_8, m, seed_vertex=0)
        tl = compute_teacher_labels(directed_8, anchors, use_gpu=False)
        h_alt = make_alt_heuristic(tl)

        # Symmetrized covering radius for directed:
        # r_m^sym = max_v min_l max(d(l,v), d(v,l))
        anchor_list = anchors.tolist()
        r_m_sym = 0.0
        for v in range(V):
            min_sym_dist = float("inf")
            for l in anchor_list:
                d_lv = dist[l, v].item()
                d_vl = dist[v, l].item()
                if d_lv > 1e15 or d_vl > 1e15:
                    continue
                sym = max(d_lv, d_vl)
                min_sym_dist = min(min_sym_dist, sym)
            if min_sym_dist < 1e15:
                r_m_sym = max(r_m_sym, min_sym_dist)

        # Check: d(u,t) - h_ALT^S(u,t) <= 2*r_m^sym
        for u in range(V):
            for t in range(V):
                d_val = dist[u, t].item()
                if d_val > 1e15:
                    continue
                h_val = h_alt(u, t)
                gap = d_val - h_val
                assert gap <= 2.0 * r_m_sym + 1e-10, (
                    f"Directed covering radius bound violated: "
                    f"gap({u},{t})={gap:.4f} > 2*r_m_sym={2*r_m_sym:.4f}"
                )


# ---------------------------------------------------------------------------
# T-INV-08: A* with admissible+consistent heuristic finds optimal paths
# Section 2.1: Graph-search A* without reopenings is optimal
# ---------------------------------------------------------------------------


class TestAStarOptimality:
    """Verify that A* with AAC heuristic finds optimal paths (same cost
    as Dijkstra) on small graphs."""

    def test_optimality_undirected(self, undirected_8):
        torch.manual_seed(42)
        K = 8
        m = 4
        anchors = torch.arange(K, dtype=torch.int64)
        tl = compute_teacher_labels(undirected_8, anchors, use_gpu=False)

        comp = LinearCompressor(K=K, m=m, is_directed=False).double()
        comp.eval()

        d_out_t = tl.d_out.t().to(torch.float64)
        with torch.no_grad():
            y = comp(d_out_t)
        h_aac = make_linear_heuristic(y, y, is_directed=False)

        # Dijkstra heuristic (h=0)
        h_zero = lambda u, t: 0.0

        V = undirected_8.num_nodes
        for s in range(V):
            for t in range(V):
                if s == t:
                    continue
                r_astar = astar(undirected_8, s, t, h_aac)
                r_dijkstra = astar(undirected_8, s, t, h_zero)
                assert abs(r_astar.cost - r_dijkstra.cost) < 1e-10, (
                    f"A* cost {r_astar.cost} != Dijkstra cost {r_dijkstra.cost} "
                    f"for ({s},{t})"
                )
                assert r_astar.optimal, f"A* reports non-optimal for ({s},{t})"

    def test_optimality_directed(self, directed_8):
        torch.manual_seed(42)
        K = 8
        m = 4
        anchors = torch.arange(K, dtype=torch.int64)
        tl = compute_teacher_labels(directed_8, anchors, use_gpu=False)

        comp = LinearCompressor(K=K, m=m, is_directed=True).double()
        comp.eval()

        d_out_t = tl.d_out.t().to(torch.float64)
        d_in_t = tl.d_in.t().to(torch.float64)
        with torch.no_grad():
            y_fwd, y_bwd = comp(d_out_t, d_in_t)
        h_aac = make_linear_heuristic(y_fwd, y_bwd, is_directed=True)

        h_zero = lambda u, t: 0.0

        V = directed_8.num_nodes
        for s in range(V):
            for t in range(V):
                if s == t:
                    continue
                r_astar = astar(directed_8, s, t, h_aac)
                r_dijkstra = astar(directed_8, s, t, h_zero)
                if not r_dijkstra.optimal:
                    continue  # unreachable
                assert abs(r_astar.cost - r_dijkstra.cost) < 1e-10, (
                    f"A* cost {r_astar.cost} != Dijkstra cost {r_dijkstra.cost} "
                    f"for ({s},{t})"
                )

    def test_optimality_directed_with_sentinel_masking(self):
        """A* with ALT on a graph with unreachable landmarks finds optimal paths."""
        # Graph: 0->1->2->3, plus 4->5 (disconnected)
        s = torch.tensor([0, 1, 2, 4], dtype=torch.int64)
        t = torch.tensor([1, 2, 3, 5], dtype=torch.int64)
        w = torch.tensor([1.0, 2.0, 3.0, 1.0], dtype=torch.float64)
        graph = edges_to_graph(s, t, w, 6, is_directed=True)

        anchors = torch.arange(6, dtype=torch.int64)
        tl = compute_teacher_labels(graph, anchors, use_gpu=False)
        h_alt = make_alt_heuristic(tl)

        h_zero = lambda u, t: 0.0

        # Check all reachable pairs in component {0,1,2,3}
        for s_node in range(4):
            for t_node in range(s_node + 1, 4):
                r_astar = astar(graph, s_node, t_node, h_alt)
                r_dijkstra = astar(graph, s_node, t_node, h_zero)
                if not r_dijkstra.optimal:
                    continue
                assert abs(r_astar.cost - r_dijkstra.cost) < 1e-10, (
                    f"A* cost {r_astar.cost} != Dijkstra cost {r_dijkstra.cost} "
                    f"for ({s_node},{t_node}) with sentinel landmarks"
                )

    def test_fewer_expansions_than_dijkstra(self, undirected_8):
        """AAC-guided A* should expand no more nodes than Dijkstra."""
        torch.manual_seed(42)
        K = 8
        m = K  # Full budget for best heuristic
        anchors = torch.arange(K, dtype=torch.int64)
        tl = compute_teacher_labels(undirected_8, anchors, use_gpu=False)

        comp = LinearCompressor(K=K, m=m, is_directed=False).double()
        with torch.no_grad():
            comp.W.fill_(-100.0)
            for i in range(m):
                comp.W[i, i] = 100.0
        comp.eval()

        d_out_t = tl.d_out.t().to(torch.float64)
        with torch.no_grad():
            y = comp(d_out_t)
        h_aac = make_linear_heuristic(y, y, is_directed=False)
        h_zero = lambda u, t: 0.0

        V = undirected_8.num_nodes
        for s in range(V):
            for t in range(V):
                if s == t:
                    continue
                r_astar = astar(undirected_8, s, t, h_aac)
                r_dijkstra = astar(undirected_8, s, t, h_zero)
                assert r_astar.expansions <= r_dijkstra.expansions, (
                    f"A* expanded more nodes ({r_astar.expansions}) than "
                    f"Dijkstra ({r_dijkstra.expansions}) for ({s},{t})"
                )
