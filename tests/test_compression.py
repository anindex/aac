"""Tests for positive compression module (COMP-01..05, SMOOTH-04 admissibility)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from aac.graphs.convert import edges_to_graph
from aac.graphs.types import Graph
from aac.utils.numerics import SENTINEL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_teacher_and_embedding_undirected(graph: Graph, K: int | None = None):
    """Build teacher labels + Hilbert embedding from undirected graph."""
    from aac.embeddings.hilbert import build_hilbert_embedding
    from aac.embeddings.sssp import compute_teacher_labels

    V = graph.num_nodes
    if K is None:
        K = V
    anchors = torch.arange(K, dtype=torch.int64)
    labels = compute_teacher_labels(graph, anchors, use_gpu=False)
    emb = build_hilbert_embedding(labels)
    return labels, emb


def _build_teacher_and_embedding_directed(graph: Graph, K: int | None = None):
    """Build teacher labels + tropical embedding from directed graph."""
    from aac.embeddings.sssp import compute_teacher_labels
    from aac.embeddings.tropical import build_tropical_embedding

    V = graph.num_nodes
    if K is None:
        K = V
    anchors = torch.arange(K, dtype=torch.int64)
    labels = compute_teacher_labels(graph, anchors, use_gpu=False)
    emb = build_tropical_embedding(labels)
    return labels, emb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def undirected_5():
    """5-node undirected graph (same as conftest small_undirected_graph)."""
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
# COMP-01: Positivity of A
# ---------------------------------------------------------------------------


def test_compressor_positivity():
    """PositiveCompressor(input_dim=10, compressed_dim=4).A is all > 0."""
    from aac.compression.compressor import PositiveCompressor

    comp = PositiveCompressor(input_dim=10, compressed_dim=4)
    A = comp.A
    assert A.shape == (4, 10)
    assert (A > 0).all(), f"Found non-positive values in A: min={A.min().item()}"


def test_compressor_positivity_extreme():
    """A stays positive even when alpha contains values in [-100, 100]."""
    from aac.compression.compressor import PositiveCompressor

    comp = PositiveCompressor(input_dim=10, compressed_dim=4)
    with torch.no_grad():
        comp.alpha.copy_(torch.linspace(-100, 100, 40).reshape(4, 10))
    A = comp.A
    assert (A > 0).all(), f"Found non-positive values with extreme alpha: min={A.min().item()}"


def test_softplus_inv():
    """softplus(softplus_inv(x)) == x for various x values."""
    from aac.compression.compressor import softplus_inv

    for x in [0.1, 1.0, 5.0, 20.0, 50.0]:
        y = softplus_inv(x)
        recovered = F.softplus(torch.tensor(y)).item()
        assert abs(recovered - x) < 1e-5, (
            f"softplus(softplus_inv({x})) = {recovered}, expected {x}"
        )


def test_identity_init():
    """At initialization, A = exp(alpha) has block-sparse structure.

    Block entries (alpha ~ 0 + noise) -> A ~ 1.0.
    Non-block entries (alpha ~ -10) -> A ~ exp(-10) ~ 0.
    """
    from aac.compression.compressor import PositiveCompressor

    torch.manual_seed(42)
    comp = PositiveCompressor(input_dim=10, compressed_dim=4)
    A = comp.A
    # Check that the max value is near exp(0) = 1.0 (block entries)
    assert A.max().item() > 0.5, f"Expected block entries near 1.0, got max={A.max().item():.4f}"
    # Check that min value is near 0 (non-block entries with alpha ~ -10)
    assert A.min().item() < 0.01, f"Expected non-block entries near 0, got min={A.min().item():.4f}"
    # All values must be positive (exp is always positive)
    assert (A > 0).all(), "All A values must be positive"


# ---------------------------------------------------------------------------
# COMP-02: Log-domain compression
# ---------------------------------------------------------------------------


def test_log_domain_compression():
    """compressor.forward(phi) produces (V, m) output with correct shape."""
    from aac.compression.compressor import PositiveCompressor

    comp = PositiveCompressor(input_dim=20, compressed_dim=8)
    phi = torch.randn(50, 20, dtype=torch.float64)
    y = comp.double().forward(phi)
    assert y.shape == (50, 8), f"Expected shape (50, 8), got {y.shape}"


def test_log_domain_no_overflow():
    """With phi containing values up to 10000 (DIMACS scale), output is finite."""
    from aac.compression.compressor import PositiveCompressor

    comp = PositiveCompressor(input_dim=20, compressed_dim=8)
    # Synthetic DIMACS-scale phi
    phi = torch.rand(100, 20, dtype=torch.float64) * 10000.0
    y = comp.double().forward(phi)
    assert torch.isfinite(y).all(), f"Found inf/nan in output: inf={torch.isinf(y).sum()}, nan={torch.isnan(y).sum()}"


def test_compression_matches_formula():
    """y_i(v) == logsumexp(log_A[i,:] + phi[v,:]) verified element-wise."""
    from aac.compression.compressor import PositiveCompressor

    torch.manual_seed(0)
    comp = PositiveCompressor(input_dim=6, compressed_dim=3).double()
    phi = torch.randn(4, 6, dtype=torch.float64)

    y = comp.forward(phi)
    log_A = comp.log_A  # (3, 6)

    # Manual computation
    for v in range(4):
        for i in range(3):
            expected = torch.logsumexp(log_A[i, :] + phi[v, :], dim=-1)
            assert torch.allclose(y[v, i], expected, atol=1e-12), (
                f"Mismatch at y[{v},{i}]: got {y[v, i].item()}, expected {expected.item()}"
            )


# ---------------------------------------------------------------------------
# COMP-03: Admissibility (PositiveCompressor max-plus contraction)
# ---------------------------------------------------------------------------


def test_positive_compressor_admissibility(undirected_5):
    """PositiveCompressor: h_compressed <= h_teacher for 100 random (s,t) pairs on undirected graph (max-plus contraction)."""
    from aac.compression.compressor import PositiveCompressor
    from aac.embeddings.heuristic import evaluate_heuristic_batch

    labels, emb = _build_teacher_and_embedding_undirected(undirected_5)
    phi = emb.phi  # (5, 2K)
    input_dim = phi.shape[1]

    torch.manual_seed(7)
    comp = PositiveCompressor(input_dim=input_dim, compressed_dim=4).double()
    y = comp.forward(phi)  # (5, 4)

    # Generate random pairs (with replacement, 100 pairs from 5 nodes)
    torch.manual_seed(99)
    src_idx = torch.randint(0, 5, (100,))
    tgt_idx = torch.randint(0, 5, (100,))

    # Teacher heuristic
    h_teacher = evaluate_heuristic_batch(phi[src_idx], phi[tgt_idx], is_directed=False)

    # Compressed heuristic: same formula (max - min) on compressed labels
    delta_c = y[src_idx] - y[tgt_idx]
    h_compressed = delta_c.max(dim=1).values - delta_c.min(dim=1).values

    violations = (h_compressed > h_teacher + 1e-10).sum().item()
    assert violations == 0, (
        f"PositiveCompressor admissibility violated for {violations}/100 pairs. "
        f"Max violation: {(h_compressed - h_teacher).max().item():.6f}"
    )


def test_positive_compressor_admissibility_directed(strongly_connected_5):
    """PositiveCompressor: h_compressed <= h_teacher for random (s,t) pairs on directed graph (max-plus contraction)."""
    from aac.compression.compressor import PositiveCompressor
    from aac.embeddings.heuristic import evaluate_heuristic_batch

    labels, emb = _build_teacher_and_embedding_directed(strongly_connected_5)
    phi = emb.phi
    input_dim = phi.shape[1]

    torch.manual_seed(7)
    comp = PositiveCompressor(input_dim=input_dim, compressed_dim=4).double()
    y = comp.forward(phi)

    torch.manual_seed(99)
    src_idx = torch.randint(0, 5, (100,))
    tgt_idx = torch.randint(0, 5, (100,))

    # Teacher heuristic (directed: max)
    h_teacher = evaluate_heuristic_batch(phi[src_idx], phi[tgt_idx], is_directed=True)

    # Compressed heuristic (directed: max of delta)
    delta_c = y[src_idx] - y[tgt_idx]
    h_compressed = delta_c.max(dim=1).values

    violations = (h_compressed > h_teacher + 1e-10).sum().item()
    assert violations == 0, (
        f"PositiveCompressor admissibility violated for {violations}/100 pairs (directed). "
        f"Max violation: {(h_compressed - h_teacher).max().item():.6f}"
    )


def test_identity_compression_recovers_teacher(undirected_5):
    """With A=I (identity-sized), h_compressed approximately equals h_teacher."""
    from aac.compression.compressor import PositiveCompressor

    labels, emb = _build_teacher_and_embedding_undirected(undirected_5)
    phi = emb.phi  # (5, 2K)
    dim = phi.shape[1]

    comp = PositiveCompressor(input_dim=dim, compressed_dim=dim).double()
    # Set alpha so that A = exp(alpha) ~ identity
    # Diagonal alpha = 0 -> exp(0) = 1, off-diagonal alpha = -100 -> exp(-100) ~ 0
    with torch.no_grad():
        identity_alpha = torch.full((dim, dim), -100.0, dtype=torch.float64)
        identity_alpha.fill_diagonal_(0.0)
        comp.alpha.copy_(identity_alpha)

    y = comp.forward(phi)

    # With A ~ I, y_i(v) = logsumexp(log(1)*e_i + phi) ~ phi_i(v) + small correction
    # The logsumexp over near-zero off-diagonal + phi should be close to phi
    # h_compressed should be close to h_teacher
    for u in range(5):
        for t in range(5):
            delta_teacher = phi[u] - phi[t]
            delta_comp = y[u] - y[t]
            h_teacher = (delta_teacher.max() - delta_teacher.min()).item()
            h_comp = (delta_comp.max() - delta_comp.min()).item()
            # With near-identity A, compressed heuristic should be close
            assert abs(h_comp - h_teacher) < 0.5, (
                f"Identity compression too far: h_comp={h_comp:.4f}, h_teacher={h_teacher:.4f}"
            )


def test_zero_admissibility_violations(undirected_5):
    """On small graph with known SSSP, h_compressed(s,t) <= d(s,t) for all pairs."""
    from aac.compression.compressor import PositiveCompressor
    from aac.embeddings.sssp import scipy_dijkstra_batched

    labels, emb = _build_teacher_and_embedding_undirected(undirected_5)
    phi = emb.phi
    input_dim = phi.shape[1]

    torch.manual_seed(7)
    comp = PositiveCompressor(input_dim=input_dim, compressed_dim=4).double()
    y = comp.forward(phi)

    # True distances
    all_sources = torch.arange(5, dtype=torch.int64)
    ref_dist = scipy_dijkstra_batched(undirected_5, all_sources)  # (5, 5)

    violations = 0
    for u in range(5):
        for t in range(5):
            delta = y[u] - y[t]
            h = (delta.max() - delta.min()).item()
            d = ref_dist[u, t].item()
            if d < SENTINEL * 0.99 and h > d + 1e-10:
                violations += 1

    assert violations == 0, f"Found {violations} admissibility violations out of 25 pairs"


# ---------------------------------------------------------------------------
# COMP-04: Compression dimension
# ---------------------------------------------------------------------------


def test_compression_dimension():
    """PositiveCompressor(input_dim=20, compressed_dim=m).forward(phi) produces (V, m)."""
    from aac.compression.compressor import PositiveCompressor

    phi = torch.randn(10, 20, dtype=torch.float64)
    for m in [4, 8, 16, 32]:
        comp = PositiveCompressor(input_dim=20, compressed_dim=m).double()
        y = comp.forward(phi)
        assert y.shape == (10, m), f"Expected shape (10, {m}), got {y.shape}"


# ---------------------------------------------------------------------------
# COMP-05: Condition number regularization
# ---------------------------------------------------------------------------


def test_condition_regularization():
    """compressor.condition_regularization() returns scalar tensor."""
    from aac.compression.compressor import PositiveCompressor

    comp = PositiveCompressor(input_dim=10, compressed_dim=4)
    reg = comp.condition_regularization()
    assert reg.dim() == 0, f"Expected scalar, got shape {reg.shape}"
    assert torch.isfinite(reg), f"Regularization is not finite: {reg.item()}"


def test_condition_regularization_gradient():
    """Gradient of condition_regularization() w.r.t. alpha is nonzero."""
    from aac.compression.compressor import PositiveCompressor

    comp = PositiveCompressor(input_dim=10, compressed_dim=4)
    reg = comp.condition_regularization()
    reg.backward()
    grad = comp.alpha.grad
    assert grad is not None, "No gradient computed for alpha"
    assert (grad != 0).any(), "Gradient is all zeros"


def test_condition_number_monitoring():
    """compressor.condition_number() returns finite float > 0."""
    from aac.compression.compressor import PositiveCompressor

    comp = PositiveCompressor(input_dim=10, compressed_dim=4)
    cn = comp.condition_number()
    assert isinstance(cn, float), f"Expected float, got {type(cn)}"
    assert cn > 0, f"Condition number should be > 0, got {cn}"
    assert math.isfinite(cn), f"Condition number should be finite, got {cn}"
