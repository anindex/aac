"""End-to-end smoke tests for the full AAC pipeline.

Runs the entire pipeline on tiny graphs to verify:
1. FPS anchor selection
2. SSSP teacher label computation
3. LinearCompressor training
4. Compressed heuristic construction
5. A* search with compressed heuristic
6. Admissibility of the full pipeline output
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
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from aac.utils.numerics import SENTINEL


def _build_8node_directed():
    """8-node directed graph with known structure for smoke testing.

    A connected graph with enough structure to be non-trivial but small
    enough to run quickly.
    """
    edges = [
        (0, 1, 2.0), (1, 2, 3.0), (2, 3, 1.0), (3, 0, 5.0),
        (0, 4, 4.0), (4, 5, 2.0), (5, 6, 1.0), (6, 7, 3.0),
        (7, 3, 2.0), (1, 5, 6.0), (2, 6, 4.0), (4, 1, 1.0),
        (5, 2, 3.0), (6, 3, 2.0), (7, 0, 8.0), (3, 7, 1.0),
    ]
    s = torch.tensor([e[0] for e in edges], dtype=torch.int64)
    t = torch.tensor([e[1] for e in edges], dtype=torch.int64)
    w = torch.tensor([e[2] for e in edges], dtype=torch.float64)
    return edges_to_graph(s, t, w, 8, is_directed=True)


def _build_8node_undirected():
    """8-node undirected graph for smoke testing."""
    edges = [
        (0, 1, 2.0), (1, 2, 3.0), (2, 3, 1.0), (3, 4, 2.0),
        (4, 5, 1.0), (5, 6, 3.0), (6, 7, 2.0), (0, 7, 5.0),
        (1, 5, 4.0), (2, 6, 6.0), (0, 3, 7.0), (4, 7, 3.0),
    ]
    s = torch.tensor([e[0] for e in edges], dtype=torch.int64)
    t = torch.tensor([e[1] for e in edges], dtype=torch.int64)
    w = torch.tensor([e[2] for e in edges], dtype=torch.float64)
    return edges_to_graph(s, t, w, 8, is_directed=False)


class TestFullPipelineDirected:
    """Smoke test: full pipeline on directed 8-node graph."""

    def test_full_pipeline_directed(self):
        """FPS -> SSSP -> train -> compress -> A* -> admissible."""
        torch.manual_seed(42)
        graph = _build_8node_directed()
        V = graph.num_nodes

        # Step 1: FPS anchor selection
        K0 = 4
        anchors = farthest_point_sampling(graph, K0, seed_vertex=0)
        assert anchors.shape[0] == K0
        assert anchors.dtype == torch.int64
        assert torch.all(anchors < V)

        # Step 2: SSSP teacher labels
        teacher = compute_teacher_labels(graph, anchors, use_gpu=False)
        assert teacher.d_out.shape == (K0, V)
        assert teacher.d_in.shape == (K0, V)
        assert teacher.is_directed

        # Step 3: Train LinearCompressor
        m = 4  # 2 fwd + 2 bwd
        compressor = LinearCompressor(K=K0, m=m, is_directed=True).double()
        cfg = TrainConfig(num_epochs=50, batch_size=16, seed=42)
        result = train_linear_compressor(compressor, teacher, cfg)
        assert "train_loss" in result
        assert len(result["train_loss"]) > 0

        # Step 4: Compress labels
        compressor.eval()
        d_out_t = teacher.d_out.t()
        d_in_t = teacher.d_in.t()
        with torch.no_grad():
            y_fwd, y_bwd = compressor(d_out_t, d_in_t)
        assert y_fwd.shape == (V, compressor.m_fwd)
        assert y_bwd.shape == (V, compressor.m_bwd)

        # Step 5: Build heuristic and run A*
        h = make_linear_heuristic(y_fwd, y_bwd, is_directed=True)
        result = astar(graph, 0, 7, heuristic=h)
        assert result.cost < SENTINEL * 0.99, "No path found"
        assert len(result.path) >= 2
        assert result.path[0] == 0
        assert result.path[-1] == 7

        # Step 6: Verify admissibility on all reachable pairs
        violations = 0
        for s in range(V):
            for t in range(V):
                if s == t:
                    continue
                h_val = h(s, t)
                d_val = dijkstra(graph, s, t).cost
                if d_val < SENTINEL * 0.99 and h_val > d_val + 1e-10:
                    violations += 1
        assert violations == 0, f"{violations} admissibility violations"

    def test_aac_vs_alt_comparison(self):
        """AAC heuristic should be <= ALT heuristic (it's compressed ALT)."""
        torch.manual_seed(42)
        graph = _build_8node_directed()
        V = graph.num_nodes

        K0 = 4
        anchors = farthest_point_sampling(graph, K0, seed_vertex=0)
        teacher = compute_teacher_labels(graph, anchors, use_gpu=False)

        # ALT heuristic (uncompressed)
        h_alt = make_alt_heuristic(teacher)

        # AAC heuristic (compressed, m < K0)
        compressor = LinearCompressor(K=K0, m=4, is_directed=True).double()
        cfg = TrainConfig(num_epochs=50, batch_size=16, seed=42)
        train_linear_compressor(compressor, teacher, cfg)
        compressor.eval()
        with torch.no_grad():
            y_fwd, y_bwd = compressor(teacher.d_out.t(), teacher.d_in.t())
        h_aac = make_linear_heuristic(y_fwd, y_bwd, is_directed=True)

        # h_aac <= h_alt for all pairs (AAC selects a subset of ALT landmarks)
        violations = 0
        for s in range(V):
            for t in range(V):
                if s == t:
                    continue
                diff = h_aac(s, t) - h_alt(s, t)
                if diff > 1e-10:
                    violations += 1
        assert violations == 0, f"AAC > ALT for {violations} pairs"


class TestFullPipelineUndirected:
    """Smoke test: full pipeline on undirected 8-node graph."""

    def test_full_pipeline_undirected(self):
        """FPS -> SSSP -> train -> compress -> A* -> admissible."""
        torch.manual_seed(42)
        graph = _build_8node_undirected()
        V = graph.num_nodes

        K0 = 4
        anchors = farthest_point_sampling(graph, K0, seed_vertex=0)
        teacher = compute_teacher_labels(graph, anchors, use_gpu=False)
        assert not teacher.is_directed

        m = 3
        compressor = LinearCompressor(K=K0, m=m, is_directed=False).double()
        cfg = TrainConfig(num_epochs=50, batch_size=16, seed=42)
        train_linear_compressor(compressor, teacher, cfg)

        compressor.eval()
        d_out_t = teacher.d_out.t()
        with torch.no_grad():
            y = compressor(d_out_t)
        h = make_linear_heuristic(y, y, is_directed=False)

        # Run A* on a few queries
        for s, t in [(0, 7), (1, 6), (3, 5)]:
            result_aac = astar(graph, s, t, heuristic=h)
            result_dij = dijkstra(graph, s, t)
            assert abs(result_aac.cost - result_dij.cost) < 1e-10, (
                f"A*({s},{t}) cost={result_aac.cost} != Dijkstra cost={result_dij.cost}"
            )

        # Full admissibility check
        violations = 0
        for s in range(V):
            for t in range(V):
                if s == t:
                    continue
                h_val = h(s, t)
                d_val = dijkstra(graph, s, t).cost
                if d_val < SENTINEL * 0.99 and h_val > d_val + 1e-10:
                    violations += 1
        assert violations == 0, f"{violations} admissibility violations"


class TestAsymmetricPipeline:
    """Smoke test: asymmetric split on directed graph."""

    @pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
    def test_asymmetric_pipeline(self, ratio):
        """Full pipeline with different m_fwd_ratio values stays admissible."""
        torch.manual_seed(42)
        graph = _build_8node_directed()
        V = graph.num_nodes

        K0 = 4
        m = 4
        anchors = farthest_point_sampling(graph, K0, seed_vertex=0)
        teacher = compute_teacher_labels(graph, anchors, use_gpu=False)

        compressor = LinearCompressor(
            K=K0, m=m, is_directed=True, m_fwd_ratio=ratio
        ).double()
        cfg = TrainConfig(num_epochs=30, batch_size=16, seed=42)
        train_linear_compressor(compressor, teacher, cfg)

        compressor.eval()
        with torch.no_grad():
            y_fwd, y_bwd = compressor(teacher.d_out.t(), teacher.d_in.t())

        assert y_fwd.shape[1] == compressor.m_fwd
        assert y_bwd.shape[1] == compressor.m_bwd
        assert compressor.m_fwd + compressor.m_bwd == m

        h = make_linear_heuristic(y_fwd, y_bwd, is_directed=True)

        violations = 0
        for s in range(V):
            for t in range(V):
                if s == t:
                    continue
                h_val = h(s, t)
                d_val = dijkstra(graph, s, t).cost
                if d_val < SENTINEL * 0.99 and h_val > d_val + 1e-10:
                    violations += 1
        assert violations == 0, (
            f"ratio={ratio}: {violations} admissibility violations"
        )


class TestALTFairness:
    """Verify ALT preprocessing accepts valid_vertices parameter."""

    def test_alt_with_valid_vertices(self):
        """ALT with valid_vertices restricts landmarks to specified subset."""
        graph = _build_8node_directed()
        lcc = torch.arange(8, dtype=torch.int64)
        teacher = alt_preprocess(graph, 3, seed_vertex=0, valid_vertices=lcc)
        assert teacher.d_out.shape[0] == 3
        h = make_alt_heuristic(teacher)
        # Should produce valid heuristic
        val = h(0, 7)
        assert val >= 0.0
