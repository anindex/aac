"""Regression tests for heuristic edge cases.

Covers sentinel masking, directed masking with unreachable landmarks,
smooth Bellman-Ford overflow, mutual unreachability fallback, and
path reconstruction.
"""

from __future__ import annotations

import torch

from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.convert import edges_to_graph
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.utils.numerics import SENTINEL

# ---------------------------------------------------------------------------
# Sentinel masking in make_linear_heuristic
# ---------------------------------------------------------------------------


class TestSentinelMasking:
    """Verify that make_linear_heuristic handles sentinel values correctly."""

    def test_sentinel_in_compressed_labels_directed(self):
        """If compressed labels contain sentinels, heuristic must still be admissible."""
        V, m_fwd, m_bwd = 5, 3, 3
        y_fwd = torch.tensor([
            [1.0, 2.0, 3.0],
            [2.0, SENTINEL, 4.0],  # vertex 1 has sentinel in dim 1
            [3.0, 4.0, 5.0],
            [1.5, 2.5, SENTINEL],  # vertex 3 has sentinel in dim 2
            [4.0, 5.0, 6.0],
        ], dtype=torch.float64)
        y_bwd = torch.tensor([
            [1.0, 2.0, 3.0],
            [SENTINEL, 3.0, 4.0],  # vertex 1 has sentinel in dim 0
            [3.0, 4.0, 5.0],
            [1.5, 2.5, 3.5],
            [4.0, SENTINEL, 6.0],  # vertex 4 has sentinel in dim 1
        ], dtype=torch.float64)

        h = make_linear_heuristic(y_fwd, y_bwd, is_directed=True)

        # Query between vertex 1 (has sentinels) and vertex 2 (no sentinels)
        val = h(1, 2)
        # Must NOT be ~1e18 (sentinel leaking through)
        assert val < 1e10, f"Sentinel leaked into heuristic: h(1,2) = {val}"
        assert val >= 0.0, f"Heuristic must be non-negative: h(1,2) = {val}"

        # Query between two vertices with sentinels
        val = h(1, 3)
        assert val < 1e10, f"Sentinel leaked: h(1,3) = {val}"
        assert val >= 0.0

    def test_sentinel_in_compressed_labels_undirected(self):
        """Undirected case: sentinel values must be masked."""
        V, m = 4, 3
        y = torch.tensor([
            [1.0, 2.0, 3.0],
            [SENTINEL, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [2.0, SENTINEL, 4.0],
        ], dtype=torch.float64)

        h = make_linear_heuristic(y, y, is_directed=False)

        val = h(1, 2)
        assert val < 1e10, f"Sentinel leaked: h(1,2) = {val}"
        assert val >= 0.0

    def test_no_sentinel_unchanged(self, strongly_connected_directed):
        """When no sentinels present, behavior should be identical to before fix."""
        graph = strongly_connected_directed
        teacher = compute_teacher_labels(graph, torch.tensor([0, 1, 2, 3]))
        compressor = LinearCompressor(K=4, m=4, is_directed=True)

        # Train briefly to get non-trivial weights
        compressor.eval()
        d_out_t = teacher.d_out.t()
        d_in_t = teacher.d_in.t()
        with torch.no_grad():
            y_fwd, y_bwd = compressor(d_out_t, d_in_t)

        # Verify no sentinels
        assert (y_fwd.abs() < 0.99 * SENTINEL).all(), "Unexpected sentinel in y_fwd"
        assert (y_bwd.abs() < 0.99 * SENTINEL).all(), "Unexpected sentinel in y_bwd"

        h = make_linear_heuristic(y_fwd, y_bwd, is_directed=True)

        # Check admissibility
        for u in range(graph.num_nodes):
            for t in range(graph.num_nodes):
                if u == t:
                    continue
                d_true = dijkstra(graph, u, t).cost
                h_val = h(u, t)
                assert h_val <= d_true + 1e-9, (
                    f"h({u},{t})={h_val:.4f} > d={d_true:.4f}"
                )


# ---------------------------------------------------------------------------
# Sentinel + high beta overflow in smooth Bellman-Ford
# ---------------------------------------------------------------------------


class TestSmoothBFOverflow:
    """Verify smooth Bellman-Ford handles sentinel values with high beta safely."""

    def test_high_beta_no_nan(self, strongly_connected_directed):
        """Smooth BF with beta=100 should not produce NaN or Inf."""
        from aac.contextual.smooth_bf import smooth_bellman_ford_batched

        graph = strongly_connected_directed
        sources = torch.tensor([0, 2], dtype=torch.int64)

        dist = smooth_bellman_ford_batched(graph, sources, beta=100.0)

        assert not torch.isnan(dist).any(), "NaN in smooth BF output with beta=100"
        assert not torch.isinf(dist).any(), "Inf in smooth BF output with beta=100"

    def test_high_beta_on_disconnected_graph(self):
        """Smooth BF with high beta on graph with unreachable vertices."""
        from aac.contextual.smooth_bf import smooth_bellman_ford_batched

        # Two disconnected components: {0,1,2} and {3,4}
        s = torch.tensor([0, 1, 3], dtype=torch.int64)
        t = torch.tensor([1, 2, 4], dtype=torch.int64)
        w = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
        graph = edges_to_graph(s, t, w, 5, is_directed=True)

        sources = torch.tensor([0], dtype=torch.int64)
        dist = smooth_bellman_ford_batched(graph, sources, beta=50.0)

        assert not torch.isnan(dist).any(), "NaN with disconnected graph"
        assert not torch.isinf(dist).any(), "Inf with disconnected graph"

        # Reachable vertices should have reasonable distances
        assert dist[0, 0] < 1e-6, "Source distance should be ~0"
        assert dist[0, 1] < 10.0, "d(0,1) should be reasonable"
        assert dist[0, 2] < 10.0, "d(0,2) should be reasonable"


# ---------------------------------------------------------------------------
# Mutual unreachability -- h=0 fallback
# ---------------------------------------------------------------------------


class TestMutualUnreachability:
    """Verify h=0 fallback when no landmark provides useful information."""

    def test_h_zero_when_all_landmarks_unreachable(self):
        """When all landmarks have sentinel distances, h should return 0."""
        y_fwd = torch.full((5, 3), SENTINEL, dtype=torch.float64)
        y_bwd = torch.full((5, 3), SENTINEL, dtype=torch.float64)

        h = make_linear_heuristic(y_fwd, y_bwd, is_directed=True)

        for u in range(5):
            for t in range(5):
                if u == t:
                    continue
                val = h(u, t)
                assert val == 0.0, f"Expected h=0, got {val} for ({u},{t})"


# ---------------------------------------------------------------------------
# Path reconstruction edge cases
# ---------------------------------------------------------------------------


class TestPathReconstruction:
    """Verify path reconstruction handles edge cases."""

    def test_source_equals_target(self, strongly_connected_directed):
        """Path from source to itself."""
        graph = strongly_connected_directed

        def h(u, t):
            return 0.0

        result = astar(graph, 3, 3, h)
        assert result.path == [3]
        assert result.cost == 0.0
        assert result.optimal is True

    def test_unreachable_target(self):
        """Target unreachable from source."""
        # Two disconnected components
        s = torch.tensor([0, 1], dtype=torch.int64)
        t = torch.tensor([1, 0], dtype=torch.int64)
        w = torch.tensor([1.0, 2.0], dtype=torch.float64)
        graph = edges_to_graph(s, t, w, 4, is_directed=True)

        def h(u, t):
            return 0.0

        result = astar(graph, 0, 3, h)
        assert result.path == []
        assert result.cost == float("inf")
        assert result.optimal is False

    def test_long_path_reconstruction(self):
        """Path reconstruction on a long chain graph."""
        V = 100
        s = torch.arange(V - 1, dtype=torch.int64)
        t = torch.arange(1, V, dtype=torch.int64)
        w = torch.ones(V - 1, dtype=torch.float64)
        graph = edges_to_graph(s, t, w, V, is_directed=True)

        def h(u, target):
            return max(0.0, float(target - u))

        result = astar(graph, 0, V - 1, h)
        assert len(result.path) == V
        assert result.path[0] == 0
        assert result.path[-1] == V - 1
        assert abs(result.cost - (V - 1)) < 1e-9


# ---------------------------------------------------------------------------
# Integration: Full pipeline admissibility on weakly-connected directed graph
# ---------------------------------------------------------------------------


class TestFullPipelineWeaklyConnected:
    """End-to-end test on a weakly-connected directed graph with unreachable landmarks."""

    def test_aac_admissible_on_weakly_connected(self, weakly_connected_directed):
        """AAC must maintain admissibility even on weakly-connected directed graphs."""
        graph = weakly_connected_directed
        # Use all vertices as potential anchors; FPS will select from them
        K = 2
        anchors = torch.tensor([0, 3], dtype=torch.int64)  # Manually chosen
        teacher = compute_teacher_labels(graph, anchors)

        # Some teacher distances will be sentinel (unreachable)
        has_sentinels = (teacher.d_out.abs() > 0.99 * SENTINEL).any()
        # This is expected for a weakly-connected directed graph

        compressor = LinearCompressor(K=K, m=2, is_directed=True)
        compressor.eval()

        d_out_t = teacher.d_out.t()
        d_in_t = teacher.d_in.t()
        with torch.no_grad():
            y_fwd, y_bwd = compressor(d_out_t, d_in_t)

        h = make_linear_heuristic(y_fwd, y_bwd, is_directed=True)

        # Check admissibility for all reachable pairs
        violations = []
        for u in range(graph.num_nodes):
            for t in range(graph.num_nodes):
                if u == t:
                    continue
                d_result = dijkstra(graph, u, t)
                if d_result.cost == float("inf"):
                    continue  # unreachable pair
                h_val = h(u, t)
                if h_val > d_result.cost + 1e-9:
                    violations.append(
                        f"h({u},{t})={h_val:.4f} > d={d_result.cost:.4f}"
                    )

        assert len(violations) == 0, (
            "Admissibility violations on weakly-connected graph:\n"
            + "\n".join(violations[:10])
        )
