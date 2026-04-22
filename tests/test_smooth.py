"""Tests for smoothed heuristic evaluation (SMOOTH-01..04) and AAC heuristic callable."""

from __future__ import annotations

import math

import pytest
import torch

from aac.graphs.convert import edges_to_graph
from aac.graphs.types import Graph
from aac.utils.numerics import SENTINEL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def undirected_5():
    """5-node undirected graph."""
    s = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int64)
    t = torch.tensor([1, 2, 2, 3, 3, 4], dtype=torch.int64)
    w = torch.tensor([1.0, 4.0, 2.0, 5.0, 1.0, 3.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 5, is_directed=False)


@pytest.fixture
def strongly_connected_5():
    """5-node strongly connected directed graph."""
    s = torch.tensor([0, 1, 2, 3, 4, 0, 2, 1], dtype=torch.int64)
    t = torch.tensor([1, 2, 3, 4, 0, 2, 4, 3], dtype=torch.int64)
    w = torch.tensor([2.0, 1.0, 2.0, 1.0, 3.0, 5.0, 7.0, 6.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 5, is_directed=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_compressed_labels_undirected(graph: Graph, m: int = 4, seed: int = 7):
    """Build compressed labels from undirected graph via teacher -> hilbert -> compress."""
    from aac.embeddings.sssp import compute_teacher_labels
    from aac.embeddings.hilbert import build_hilbert_embedding
    from aac.compression.compressor import PositiveCompressor

    V = graph.num_nodes
    anchors = torch.arange(V, dtype=torch.int64)
    labels = compute_teacher_labels(graph, anchors, use_gpu=False)
    emb = build_hilbert_embedding(labels)
    phi = emb.phi  # (V, 2K)

    torch.manual_seed(seed)
    comp = PositiveCompressor(input_dim=phi.shape[1], compressed_dim=m).double()
    y = comp.forward(phi)  # (V, m)
    return y, emb


def _build_compressed_labels_directed(graph: Graph, m: int = 4, seed: int = 7):
    """Build compressed labels from directed graph via teacher -> tropical -> compress."""
    from aac.embeddings.sssp import compute_teacher_labels
    from aac.embeddings.tropical import build_tropical_embedding
    from aac.compression.compressor import PositiveCompressor

    V = graph.num_nodes
    anchors = torch.arange(V, dtype=torch.int64)
    labels = compute_teacher_labels(graph, anchors, use_gpu=False)
    emb = build_tropical_embedding(labels)
    phi = emb.phi

    torch.manual_seed(seed)
    comp = PositiveCompressor(input_dim=phi.shape[1], compressed_dim=m).double()
    y = comp.forward(phi)
    return y, emb


# ---------------------------------------------------------------------------
# SMOOTH-01: Lower bound property
# ---------------------------------------------------------------------------


def test_smooth_lower_bound_directed():
    """M_T(delta) <= max(delta) for 100 random delta vectors, T in [0.1, 1.0, 10.0, 100.0]."""
    from aac.compression.smooth import smoothed_heuristic_directed

    torch.manual_seed(42)
    m = 8
    delta_vecs = torch.randn(100, m, dtype=torch.float64)
    # Split into y_source, y_target so delta = y_source - y_target
    y_source = delta_vecs
    y_target = torch.zeros(100, m, dtype=torch.float64)

    for T in [0.1, 1.0, 10.0, 100.0]:
        h_smooth = smoothed_heuristic_directed(y_source, y_target, temperature=T)
        h_hard = delta_vecs.max(dim=1).values
        violations = (h_smooth > h_hard + 1e-10).sum().item()
        assert violations == 0, (
            f"Lower bound violated at T={T}: {violations}/100 violations. "
            f"Max violation: {(h_smooth - h_hard).max().item():.8f}"
        )


def test_smooth_lower_bound_undirected():
    """M_T(delta) + M_T(-delta) <= max(delta) - min(delta) for 100 random delta vectors."""
    from aac.compression.smooth import smoothed_heuristic_undirected

    torch.manual_seed(42)
    m = 8
    delta_vecs = torch.randn(100, m, dtype=torch.float64)
    y_source = delta_vecs
    y_target = torch.zeros(100, m, dtype=torch.float64)

    for T in [0.1, 1.0, 10.0, 100.0]:
        h_smooth = smoothed_heuristic_undirected(y_source, y_target, temperature=T)
        h_hard = delta_vecs.max(dim=1).values - delta_vecs.min(dim=1).values
        violations = (h_smooth > h_hard + 1e-10).sum().item()
        assert violations == 0, (
            f"Undirected lower bound violated at T={T}: {violations}/100 violations. "
            f"Max violation: {(h_smooth - h_hard).max().item():.8f}"
        )


def test_smooth_formula_directed():
    """M_T(x) == (1/T)*logsumexp(T*x) - log(m)/T verified element-wise."""
    from aac.compression.smooth import smoothed_heuristic_directed

    torch.manual_seed(0)
    m = 5
    x = torch.randn(10, m, dtype=torch.float64)
    y_source = x
    y_target = torch.zeros(10, m, dtype=torch.float64)
    T = 3.0

    h = smoothed_heuristic_directed(y_source, y_target, temperature=T)

    # Manual formula
    expected = torch.logsumexp(T * x, dim=-1) / T - math.log(m) / T

    assert torch.allclose(h, expected, atol=1e-12), (
        f"Formula mismatch: max diff = {(h - expected).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# SMOOTH-02: Annealing convergence
# ---------------------------------------------------------------------------


def test_annealing_convergence():
    """|M_T(delta) - max(delta)| decreases monotonically as T increases from 1.0 to 1000.0."""
    from aac.compression.smooth import smoothed_heuristic_directed

    torch.manual_seed(42)
    m = 8
    delta = torch.randn(10, m, dtype=torch.float64)
    y_source = delta
    y_target = torch.zeros(10, m, dtype=torch.float64)
    h_hard = delta.max(dim=1).values

    temperatures = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    prev_gap = float("inf")

    for T in temperatures:
        h_smooth = smoothed_heuristic_directed(y_source, y_target, temperature=T)
        gap = (h_smooth - h_hard).abs().max().item()
        assert gap <= prev_gap + 1e-10, (
            f"Gap did not decrease: T={T} gap={gap:.8f} > prev_gap={prev_gap:.8f}"
        )
        prev_gap = gap


def test_smooth_equals_hard_at_high_T():
    """At T=10000, |M_T(delta) - max(delta)| < 0.001 for random delta."""
    from aac.compression.smooth import smoothed_heuristic_directed

    torch.manual_seed(42)
    m = 8
    delta = torch.randn(50, m, dtype=torch.float64)
    y_source = delta
    y_target = torch.zeros(50, m, dtype=torch.float64)

    h_smooth = smoothed_heuristic_directed(y_source, y_target, temperature=10000.0)
    h_hard = delta.max(dim=1).values

    max_diff = (h_smooth - h_hard).abs().max().item()
    assert max_diff < 0.001, f"At T=10000, max diff = {max_diff:.6f}, expected < 0.001"


# ---------------------------------------------------------------------------
# SMOOTH-03: Gradient flow (gradcheck)
# ---------------------------------------------------------------------------


def test_smooth_heuristic_gradcheck_directed():
    """torch.autograd.gradcheck passes for smoothed_heuristic_directed with fp64 inputs."""
    from aac.compression.smooth import smoothed_heuristic_directed

    torch.manual_seed(0)
    y_s = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    y_t = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)

    def fn(ys, yt):
        return smoothed_heuristic_directed(ys, yt, temperature=2.0)

    assert torch.autograd.gradcheck(fn, (y_s, y_t), eps=1e-6, atol=1e-4), (
        "gradcheck failed for smoothed_heuristic_directed"
    )


def test_smooth_heuristic_gradcheck_undirected():
    """torch.autograd.gradcheck passes for smoothed_heuristic_undirected with fp64 inputs."""
    from aac.compression.smooth import smoothed_heuristic_undirected

    torch.manual_seed(0)
    y_s = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    y_t = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)

    def fn(ys, yt):
        return smoothed_heuristic_undirected(ys, yt, temperature=2.0)

    assert torch.autograd.gradcheck(fn, (y_s, y_t), eps=1e-6, atol=1e-4), (
        "gradcheck failed for smoothed_heuristic_undirected"
    )


# ---------------------------------------------------------------------------
# SMOOTH-04: Admissibility in fp32
# ---------------------------------------------------------------------------


def test_smooth_admissibility_fp32(undirected_5):
    """With fp32 compressed labels, h_smooth(s,t) <= d(s,t) + 1e-4 for all pairs."""
    from aac.compression.smooth import smoothed_heuristic_undirected
    from aac.embeddings.sssp import scipy_dijkstra_batched

    y_fp64, emb = _build_compressed_labels_undirected(undirected_5, m=4, seed=7)
    y = y_fp64.float()  # Cast to fp32

    # True distances
    all_sources = torch.arange(5, dtype=torch.int64)
    ref_dist = scipy_dijkstra_batched(undirected_5, all_sources)  # (5, 5) fp64

    violations = 0
    for u in range(5):
        for t in range(5):
            h = smoothed_heuristic_undirected(
                y[u].unsqueeze(0), y[t].unsqueeze(0), temperature=10.0
            ).item()
            d = ref_dist[u, t].item()
            if d < SENTINEL * 0.99 and h > d + 1e-4:
                violations += 1

    assert violations == 0, f"Found {violations} fp32 admissibility violations out of 25 pairs"


# ---------------------------------------------------------------------------
# AAC heuristic callable tests
# ---------------------------------------------------------------------------


def test_make_aac_heuristic_directed(strongly_connected_5):
    """make_aac_heuristic returns callable, h(node, target) returns float."""
    from aac.compression.smooth import make_aac_heuristic

    y, emb = _build_compressed_labels_directed(strongly_connected_5, m=4)
    h = make_aac_heuristic(y, is_directed=True)

    # Should be callable
    assert callable(h)

    # Returns float
    val = h(0, 4)
    assert isinstance(val, float), f"Expected float, got {type(val)}"


def test_make_aac_heuristic_undirected(undirected_5):
    """make_aac_heuristic returns callable with variation norm for undirected."""
    from aac.compression.smooth import make_aac_heuristic

    y, emb = _build_compressed_labels_undirected(undirected_5, m=4)
    h = make_aac_heuristic(y, is_directed=False)

    # Returns float
    val = h(0, 4)
    assert isinstance(val, float)

    # Undirected: h(u,t) == h(t,u) for variation norm (symmetric)
    h_04 = h(0, 4)
    h_40 = h(4, 0)
    assert abs(h_04 - h_40) < 1e-10, (
        f"Undirected heuristic should be symmetric: h(0,4)={h_04}, h(4,0)={h_40}"
    )


def test_aac_heuristic_admissible(undirected_5):
    """make_aac_heuristic callable produces admissible values on small graph."""
    from aac.compression.smooth import make_aac_heuristic
    from aac.embeddings.sssp import scipy_dijkstra_batched

    y, emb = _build_compressed_labels_undirected(undirected_5, m=4)
    h = make_aac_heuristic(y, is_directed=False)

    all_sources = torch.arange(5, dtype=torch.int64)
    ref_dist = scipy_dijkstra_batched(undirected_5, all_sources)

    violations = 0
    for u in range(5):
        for t in range(5):
            h_val = h(u, t)
            d_val = ref_dist[u, t].item()
            if d_val < SENTINEL * 0.99 and h_val > d_val + 1e-10:
                violations += 1

    assert violations == 0, f"Found {violations} admissibility violations in AAC heuristic"
