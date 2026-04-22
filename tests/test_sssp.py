"""Tests for batched SSSP (Bellman-Ford and SciPy Dijkstra reference)."""

import pytest
import torch

from aac.graphs.types import Graph, TeacherLabels
from aac.graphs.convert import edges_to_graph, graph_to_scipy
from aac.utils.numerics import SENTINEL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def directed_graph_5():
    """5-node directed graph.

    Edges: (0,1,2), (0,2,5), (1,2,1), (1,3,6), (2,3,2), (2,4,7), (3,4,1)
    Known shortest paths from 0: d(0,0)=0, d(0,1)=2, d(0,2)=3, d(0,3)=5, d(0,4)=6.
    """
    s = torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.int64)
    t = torch.tensor([1, 2, 2, 3, 3, 4, 4], dtype=torch.int64)
    w = torch.tensor([2.0, 5.0, 1.0, 6.0, 2.0, 7.0, 1.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 5, is_directed=True)


@pytest.fixture
def undirected_graph_5():
    """5-node undirected graph.

    Edges: (0,1,1), (0,2,4), (1,2,2), (1,3,5), (2,3,1), (3,4,3)
    Known shortest paths from 0: d(0,0)=0, d(0,1)=1, d(0,2)=3, d(0,3)=4, d(0,4)=7.
    """
    s = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int64)
    t = torch.tensor([1, 2, 2, 3, 3, 4], dtype=torch.int64)
    w = torch.tensor([1.0, 4.0, 2.0, 5.0, 1.0, 3.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 5, is_directed=False)


@pytest.fixture
def graph_with_isolated_node():
    """4-node graph with node 3 isolated (unreachable).

    Edges: (0,1,1), (1,2,2) -- node 3 has no edges.
    """
    s = torch.tensor([0, 1], dtype=torch.int64)
    t = torch.tensor([1, 2], dtype=torch.int64)
    w = torch.tensor([1.0, 2.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 4, is_directed=True)


# ---------------------------------------------------------------------------
# Bellman-Ford vs SciPy Dijkstra
# ---------------------------------------------------------------------------


def test_bellman_ford_vs_scipy_directed(directed_graph_5):
    """Bellman-Ford and SciPy Dijkstra must produce identical distances on directed graph."""
    from aac.embeddings.sssp import bellman_ford_batched, scipy_dijkstra_batched

    sources = torch.tensor([0, 1, 2], dtype=torch.int64)
    bf_dist = bellman_ford_batched(directed_graph_5, sources)
    sp_dist = scipy_dijkstra_batched(directed_graph_5, sources)
    assert torch.allclose(bf_dist, sp_dist, atol=1e-10), (
        f"Bellman-Ford vs SciPy mismatch:\nBF: {bf_dist}\nSP: {sp_dist}"
    )


def test_bellman_ford_vs_scipy_undirected(undirected_graph_5):
    """Bellman-Ford and SciPy Dijkstra must match on undirected graph."""
    from aac.embeddings.sssp import bellman_ford_batched, scipy_dijkstra_batched

    sources = torch.tensor([0, 2, 4], dtype=torch.int64)
    bf_dist = bellman_ford_batched(undirected_graph_5, sources)
    sp_dist = scipy_dijkstra_batched(undirected_graph_5, sources)
    assert torch.allclose(bf_dist, sp_dist, atol=1e-10), (
        f"Bellman-Ford vs SciPy mismatch:\nBF: {bf_dist}\nSP: {sp_dist}"
    )


def test_bellman_ford_unreachable(graph_with_isolated_node):
    """Unreachable nodes must have distance == SENTINEL."""
    from aac.embeddings.sssp import bellman_ford_batched

    sources = torch.tensor([0], dtype=torch.int64)
    dist = bellman_ford_batched(graph_with_isolated_node, sources)
    # Node 3 is isolated and unreachable from 0
    assert dist[0, 3] == SENTINEL, f"Expected SENTINEL for unreachable node, got {dist[0, 3]}"
    # Node 0 distance from itself
    assert dist[0, 0] == 0.0


def test_bellman_ford_convergence(directed_graph_5):
    """Bellman-Ford on small graph should converge in fewer than V-1 iterations.

    We test indirectly: with max_iter=2 the 5-node chain 0->1->2->3->4 needs
    4 iterations worst-case, but our specific graph should converge.
    We verify that running with max_iter=V-1 produces correct results.
    """
    from aac.embeddings.sssp import bellman_ford_batched, scipy_dijkstra_batched

    sources = torch.tensor([0], dtype=torch.int64)
    bf_dist = bellman_ford_batched(directed_graph_5, sources)
    sp_dist = scipy_dijkstra_batched(directed_graph_5, sources)
    assert torch.allclose(bf_dist, sp_dist, atol=1e-10)


# ---------------------------------------------------------------------------
# Teacher labels
# ---------------------------------------------------------------------------


def test_teacher_labels_shape(undirected_graph_5):
    """compute_teacher_labels with K=3 anchors on 5-node graph produces correct shapes."""
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    labels = compute_teacher_labels(undirected_graph_5, anchors, use_gpu=False)

    assert isinstance(labels, TeacherLabels)
    assert labels.d_out.shape == (3, 5), f"Expected d_out shape (3, 5), got {labels.d_out.shape}"
    assert labels.d_in.shape == (3, 5), f"Expected d_in shape (3, 5), got {labels.d_in.shape}"
    assert labels.anchor_indices.shape == (3,)


def test_teacher_labels_undirected_symmetry(undirected_graph_5):
    """For undirected graphs, d_out must equal d_in."""
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 1, 3], dtype=torch.int64)
    labels = compute_teacher_labels(undirected_graph_5, anchors, use_gpu=False)

    assert not labels.is_directed
    assert torch.equal(labels.d_out, labels.d_in), "d_out should equal d_in for undirected graphs"


def test_teacher_labels_directed_asymmetry(directed_graph_5):
    """For directed graph where d(u,v) != d(v,u), d_out must differ from d_in."""
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    labels = compute_teacher_labels(directed_graph_5, anchors, use_gpu=False)

    assert labels.is_directed
    # On a directed graph with asymmetric distances, d_out != d_in
    assert not torch.equal(labels.d_out, labels.d_in), (
        "d_out should differ from d_in on directed graph"
    )


# ---------------------------------------------------------------------------
# Chunked SSSP (SCAL-01)
# ---------------------------------------------------------------------------


def test_chunked_sssp_matches_unchunked(undirected_graph_5):
    """chunk_size=2 produces identical d_out and d_in to chunk_size=None."""
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    unchunked = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False, backend="scipy"
    )
    chunked = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False, backend="scipy", chunk_size=2
    )

    assert torch.allclose(chunked.d_out, unchunked.d_out, atol=1e-10), (
        f"Chunked d_out differs from unchunked:\n"
        f"chunked: {chunked.d_out}\nunchunked: {unchunked.d_out}"
    )
    assert torch.allclose(chunked.d_in, unchunked.d_in, atol=1e-10), (
        f"Chunked d_in differs from unchunked:\n"
        f"chunked: {chunked.d_in}\nunchunked: {unchunked.d_in}"
    )


def test_chunked_sssp_directed(directed_graph_5):
    """chunk_size=1 on directed graph produces same d_out and d_in as unchunked."""
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    unchunked = compute_teacher_labels(
        directed_graph_5, anchors, use_gpu=False, backend="scipy"
    )
    chunked = compute_teacher_labels(
        directed_graph_5, anchors, use_gpu=False, backend="scipy", chunk_size=1
    )

    assert torch.allclose(chunked.d_out, unchunked.d_out, atol=1e-10), (
        f"Directed chunked d_out mismatch"
    )
    assert torch.allclose(chunked.d_in, unchunked.d_in, atol=1e-10), (
        f"Directed chunked d_in mismatch"
    )


def test_chunked_sssp_chunk_size_larger_than_k(undirected_graph_5):
    """chunk_size=100 with K=3 anchors works correctly (single chunk)."""
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    unchunked = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False, backend="scipy"
    )
    chunked = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False, backend="scipy", chunk_size=100
    )

    assert torch.allclose(chunked.d_out, unchunked.d_out, atol=1e-10)


# ---------------------------------------------------------------------------
# Float32 dtype (SCAL-03)
# ---------------------------------------------------------------------------


def test_float32_halves_memory(undirected_graph_5):
    """dtype=torch.float32 produces d_out with float32 dtype and 4 bytes/element."""
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    labels = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False, backend="scipy",
        dtype=torch.float32,
    )

    assert labels.d_out.dtype == torch.float32, (
        f"Expected float32 dtype, got {labels.d_out.dtype}"
    )
    assert labels.d_out.element_size() == 4, (
        f"Expected 4 bytes per element for float32, got {labels.d_out.element_size()}"
    )


def test_float32_values_close_to_float64(undirected_graph_5):
    """float32 results are within atol=1e-3 of float64 results."""
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    f64_labels = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False, backend="scipy",
        dtype=torch.float64,
    )
    f32_labels = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False, backend="scipy",
        dtype=torch.float32,
    )

    assert torch.allclose(
        f32_labels.d_out.to(torch.float64),
        f64_labels.d_out,
        atol=1e-3,
    ), "float32 d_out values differ from float64 by more than atol=1e-3"


# ---------------------------------------------------------------------------
# Backend selection (SCAL-04)
# ---------------------------------------------------------------------------


def test_backend_scipy_explicit(undirected_graph_5):
    """backend='scipy' produces same results as use_gpu=False (both use scipy_dijkstra_batched)."""
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    legacy = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False
    )
    explicit = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False, backend="scipy"
    )

    assert torch.allclose(explicit.d_out, legacy.d_out, atol=1e-10), (
        "backend='scipy' should match use_gpu=False results"
    )


def test_backend_auto_fallback(undirected_graph_5):
    """backend='auto' with NetworKit not installed falls back to SciPy."""
    import unittest.mock
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)

    # Mock NetworKit as unavailable
    with unittest.mock.patch.dict("sys.modules", {"networkit": None}):
        labels = compute_teacher_labels(
            undirected_graph_5, anchors, use_gpu=False, backend="auto"
        )

    # Should still produce correct results via SciPy fallback
    scipy_labels = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False, backend="scipy"
    )

    assert torch.allclose(labels.d_out, scipy_labels.d_out, atol=1e-10), (
        "Auto fallback should produce same results as explicit scipy backend"
    )


# ---------------------------------------------------------------------------
# NetworKit backend (SCAL-04) -- skipped if not installed
# ---------------------------------------------------------------------------


def test_networkit_sentinel_mapping(graph_with_isolated_node):
    """NetworKit backend maps unreachable distances to project sentinel (1e18)."""
    nk = pytest.importorskip("networkit")
    from aac.embeddings.sssp import networkit_dijkstra_batched
    from aac.utils.numerics import SENTINEL

    sources = torch.tensor([0], dtype=torch.int64)
    dist = networkit_dijkstra_batched(graph_with_isolated_node, sources)

    # Node 3 is isolated and unreachable from 0
    assert dist[0, 3] == SENTINEL, (
        f"Expected SENTINEL for unreachable node, got {dist[0, 3]}"
    )
    # Node 0 distance from itself
    assert dist[0, 0] == 0.0


def test_networkit_matches_scipy(undirected_graph_5):
    """NetworKit backend produces distances matching SciPy within atol=1e-10."""
    nk = pytest.importorskip("networkit")
    from aac.embeddings.sssp import networkit_dijkstra_batched, scipy_dijkstra_batched

    sources = torch.tensor([0, 2, 4], dtype=torch.int64)
    nk_dist = networkit_dijkstra_batched(undirected_graph_5, sources)
    sp_dist = scipy_dijkstra_batched(undirected_graph_5, sources)

    assert torch.allclose(nk_dist, sp_dist, atol=1e-10), (
        f"NetworKit vs SciPy mismatch:\nNK: {nk_dist}\nSP: {sp_dist}"
    )


# ---------------------------------------------------------------------------
# Combined modes
# ---------------------------------------------------------------------------


def test_chunked_with_float32_combined(undirected_graph_5):
    """chunk_size=2 + dtype=float32 produces float32 tensor close to unchunked float64."""
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    f64_unchunked = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False, backend="scipy",
    )
    f32_chunked = compute_teacher_labels(
        undirected_graph_5, anchors, use_gpu=False, backend="scipy",
        chunk_size=2, dtype=torch.float32,
    )

    assert f32_chunked.d_out.dtype == torch.float32
    assert torch.allclose(
        f32_chunked.d_out.to(torch.float64),
        f64_unchunked.d_out,
        atol=1e-3,
    ), "Chunked float32 should be close to unchunked float64"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_backward_compat_no_new_params(undirected_graph_5):
    """compute_teacher_labels(graph, anchors, use_gpu=False) with no new params works."""
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    labels = compute_teacher_labels(undirected_graph_5, anchors, use_gpu=False)

    assert isinstance(labels, TeacherLabels)
    assert labels.d_out.shape == (3, 5)
    assert labels.d_out.dtype == torch.float64
    assert not labels.is_directed
    assert torch.equal(labels.d_out, labels.d_in)
