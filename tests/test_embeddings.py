"""Tests for anchor selection, embeddings, and heuristic evaluation."""

import pytest
import torch

from aac.graphs.convert import edges_to_graph
from aac.graphs.types import Embedding
from aac.utils.numerics import SENTINEL

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def undirected_graph_5():
    """5-node undirected graph.

    Edges: (0,1,1), (0,2,4), (1,2,2), (1,3,5), (2,3,1), (3,4,3)
    """
    s = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int64)
    t = torch.tensor([1, 2, 2, 3, 3, 4], dtype=torch.int64)
    w = torch.tensor([1.0, 4.0, 2.0, 5.0, 1.0, 3.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 5, is_directed=False)


@pytest.fixture
def directed_graph_5():
    """5-node directed graph (DAG, NOT strongly connected).

    Edges: (0,1,2), (0,2,5), (1,2,1), (1,3,6), (2,3,2), (2,4,7), (3,4,1)
    d(0,4)=6 via 0->1->2->3->4, but d(4,0)=SENTINEL (no reverse path).
    """
    s = torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.int64)
    t = torch.tensor([1, 2, 2, 3, 3, 4, 4], dtype=torch.int64)
    w = torch.tensor([2.0, 5.0, 1.0, 6.0, 2.0, 7.0, 1.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 5, is_directed=True)


@pytest.fixture
def strongly_connected_directed_5():
    """5-node strongly connected directed graph with asymmetric weights.

    Forward: 0->1(2), 1->2(1), 2->3(2), 3->4(1), 4->0(3)
    Shortcuts: 0->2(5), 2->4(7), 1->3(6)
    All pairs reachable in both directions (strongly connected).
    d(u,v) != d(v,u) for most pairs (asymmetric).
    """
    s = torch.tensor([0, 1, 2, 3, 4, 0, 2, 1], dtype=torch.int64)
    t = torch.tensor([1, 2, 3, 4, 0, 2, 4, 3], dtype=torch.int64)
    w = torch.tensor([2.0, 1.0, 2.0, 1.0, 3.0, 5.0, 7.0, 6.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 5, is_directed=True)


@pytest.fixture
def medium_undirected_graph():
    """100-node random undirected connected graph for property testing.

    Built with seed=42: spanning tree + random edges, weights in [1, 100].
    """
    torch.manual_seed(42)
    num_nodes = 100
    sources = []
    targets = []
    weights = []

    # Spanning tree for connectivity
    for i in range(1, num_nodes):
        j = torch.randint(0, i, (1,)).item()
        w = torch.randint(1, 101, (1,)).item()
        sources.append(i)
        targets.append(j)
        weights.append(float(w))

    # Extra random edges
    for _ in range(300):
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()
        if u != v:
            w = torch.randint(1, 101, (1,)).item()
            sources.append(u)
            targets.append(v)
            weights.append(float(w))

    s = torch.tensor(sources, dtype=torch.int64)
    t = torch.tensor(targets, dtype=torch.int64)
    w_t = torch.tensor(weights, dtype=torch.float64)
    return edges_to_graph(s, t, w_t, num_nodes, is_directed=False)


# ---------------------------------------------------------------------------
# Anchor selection tests (Task 1)
# ---------------------------------------------------------------------------


def test_fps_returns_unique_indices(medium_undirected_graph):
    """FPS with K=5 on 100-node graph returns 5 unique indices in [0, 100)."""
    from aac.embeddings.anchors import farthest_point_sampling

    anchors = farthest_point_sampling(medium_undirected_graph, num_anchors=5)
    assert anchors.shape == (5,), f"Expected shape (5,), got {anchors.shape}"
    assert anchors.dtype == torch.int64
    assert len(torch.unique(anchors)) == 5, "FPS anchors must be unique"
    assert (anchors >= 0).all() and (anchors < 100).all(), "Anchors must be in [0, 100)"


def test_fps_coverage(medium_undirected_graph):
    """FPS min pairwise distance among anchors > random selection min pairwise distance."""
    from aac.embeddings.anchors import farthest_point_sampling, random_anchors
    from aac.embeddings.sssp import scipy_dijkstra_batched

    K = 10
    fps_anchors = farthest_point_sampling(
        medium_undirected_graph, num_anchors=K, seed_vertex=0
    )
    rng = torch.Generator().manual_seed(42)
    rand_anchors = random_anchors(100, num_anchors=K, rng=rng)

    # Compute pairwise distances between anchors using SciPy
    fps_dist = scipy_dijkstra_batched(medium_undirected_graph, fps_anchors)
    rand_dist = scipy_dijkstra_batched(medium_undirected_graph, rand_anchors)

    # Min pairwise distance among FPS anchors
    fps_pw = fps_dist[:, fps_anchors]
    fps_pw = fps_pw + torch.eye(K, dtype=torch.float64) * SENTINEL  # mask diagonal
    fps_min = fps_pw.min().item()

    rand_pw = rand_dist[:, rand_anchors]
    rand_pw = rand_pw + torch.eye(K, dtype=torch.float64) * SENTINEL
    rand_min = rand_pw.min().item()

    assert fps_min >= rand_min, (
        f"FPS min pairwise distance ({fps_min}) should be >= random ({rand_min})"
    )


def test_random_anchors_unique():
    """random_anchors returns unique indices."""
    from aac.embeddings.anchors import random_anchors

    rng = torch.Generator().manual_seed(123)
    anchors = random_anchors(50, num_anchors=10, rng=rng)
    assert anchors.shape == (10,)
    assert anchors.dtype == torch.int64
    assert len(torch.unique(anchors)) == 10, "Random anchors must be unique"
    assert (anchors >= 0).all() and (anchors < 50).all()


def test_boundary_anchors_uses_extremes():
    """Boundary anchors include vertices at coordinate extremes."""
    from aac.embeddings.anchors import boundary_anchors

    # Create graph with coordinates where extremes are clear
    coords = torch.tensor(
        [
            [0.0, 0.0],  # 0: min-x, min-y
            [10.0, 5.0],  # 1: max-x
            [5.0, 10.0],  # 2: max-y
            [5.0, 5.0],  # 3: center
            [3.0, 3.0],  # 4: interior
        ],
        dtype=torch.float64,
    )

    s = torch.tensor([0, 0, 1, 2, 3], dtype=torch.int64)
    t = torch.tensor([1, 2, 3, 3, 4], dtype=torch.int64)
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    g = edges_to_graph(s, t, w, 5, is_directed=False, coordinates=coords)

    anchors = boundary_anchors(g, num_anchors=4)
    assert anchors.shape == (4,)
    anchor_set = set(anchors.tolist())
    # Must include vertex 0 (min-x), 1 (max-x), 0 (min-y), 2 (max-y)
    # Since vertex 0 is both min-x and min-y, we expect {0, 1, 2} plus one more
    assert 0 in anchor_set, "min-x/min-y vertex must be in boundary anchors"
    assert 1 in anchor_set, "max-x vertex must be in boundary anchors"
    assert 2 in anchor_set, "max-y vertex must be in boundary anchors"


# ---------------------------------------------------------------------------
# Hilbert embedding tests (Task 2)
# ---------------------------------------------------------------------------


def test_hilbert_embedding_shape(undirected_graph_5):
    """Hilbert embedding with K=3 anchors on 5-node undirected graph -> phi shape (5, 6)."""
    from aac.embeddings.hilbert import build_hilbert_embedding
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    labels = compute_teacher_labels(undirected_graph_5, anchors, use_gpu=False)
    emb = build_hilbert_embedding(labels)

    assert isinstance(emb, Embedding)
    assert emb.phi.shape == (5, 6), f"Expected phi shape (5, 6), got {emb.phi.shape}"
    assert emb.kind == "hilbert"
    assert not emb.is_directed
    assert emb.num_anchors == 3


def test_hilbert_rejects_directed(directed_graph_5):
    """Hilbert embedding must raise AssertionError on directed TeacherLabels."""
    from aac.embeddings.hilbert import build_hilbert_embedding
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    labels = compute_teacher_labels(directed_graph_5, anchors, use_gpu=False)

    with pytest.raises(AssertionError, match="undirected"):
        build_hilbert_embedding(labels)


def test_hilbert_admissibility(undirected_graph_5):
    """Hilbert heuristic h(u,t) <= d(u,t) for ALL (u,t) pairs on undirected graph.

    Uses all 5 vertices as anchors for strong coverage.
    """
    from aac.embeddings.heuristic import evaluate_heuristic
    from aac.embeddings.hilbert import build_hilbert_embedding
    from aac.embeddings.sssp import compute_teacher_labels, scipy_dijkstra_batched

    all_anchors = torch.arange(5, dtype=torch.int64)
    labels = compute_teacher_labels(undirected_graph_5, all_anchors, use_gpu=False)
    emb = build_hilbert_embedding(labels)

    # Reference distances
    ref_dist = scipy_dijkstra_batched(undirected_graph_5, all_anchors)  # (5, 5)

    for u in range(5):
        for t in range(5):
            h = evaluate_heuristic(emb.phi[u], emb.phi[t], is_directed=False)
            d = ref_dist[u, t]
            assert h.item() <= d.item() + 1e-10, (
                f"Admissibility violated: h({u},{t})={h.item()} > d({u},{t})={d.item()}"
            )


def test_hilbert_exactness_all_anchors(undirected_graph_5):
    """With K=V=5 anchors, Hilbert heuristic == d(u,t) for all (u,t) pairs (Theorem 1)."""
    from aac.embeddings.heuristic import evaluate_heuristic
    from aac.embeddings.hilbert import build_hilbert_embedding
    from aac.embeddings.sssp import compute_teacher_labels, scipy_dijkstra_batched

    all_anchors = torch.arange(5, dtype=torch.int64)
    labels = compute_teacher_labels(undirected_graph_5, all_anchors, use_gpu=False)
    emb = build_hilbert_embedding(labels)

    ref_dist = scipy_dijkstra_batched(undirected_graph_5, all_anchors)  # (5, 5)

    for u in range(5):
        for t in range(5):
            if u == t:
                continue
            h = evaluate_heuristic(emb.phi[u], emb.phi[t], is_directed=False)
            d = ref_dist[u, t]
            assert abs(h.item() - d.item()) < 1e-10, (
                f"Exactness failed: h({u},{t})={h.item()} != d({u},{t})={d.item()}"
            )


# ---------------------------------------------------------------------------
# Tropical embedding tests (Task 2)
# ---------------------------------------------------------------------------


def test_tropical_embedding_shape(strongly_connected_directed_5):
    """Tropical embedding with K=3 anchors on directed graph -> phi shape (5, 6)."""
    from aac.embeddings.sssp import compute_teacher_labels
    from aac.embeddings.tropical import build_tropical_embedding

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    labels = compute_teacher_labels(strongly_connected_directed_5, anchors, use_gpu=False)
    emb = build_tropical_embedding(labels)

    assert isinstance(emb, Embedding)
    assert emb.phi.shape == (5, 6), f"Expected phi shape (5, 6), got {emb.phi.shape}"
    assert emb.kind == "tropical"
    assert emb.is_directed
    assert emb.num_anchors == 3


def test_tropical_admissibility(strongly_connected_directed_5):
    """Tropical heuristic h(u,t) <= d(u,t) for ALL (u,t) pairs on strongly connected graph.

    Uses all 5 vertices as anchors. Strongly connected ensures all pairs are reachable.
    """
    from aac.embeddings.heuristic import evaluate_heuristic
    from aac.embeddings.sssp import compute_teacher_labels, scipy_dijkstra_batched
    from aac.embeddings.tropical import build_tropical_embedding

    all_anchors = torch.arange(5, dtype=torch.int64)
    labels = compute_teacher_labels(strongly_connected_directed_5, all_anchors, use_gpu=False)
    emb = build_tropical_embedding(labels)

    ref_dist = scipy_dijkstra_batched(strongly_connected_directed_5, all_anchors)  # (5, 5)

    for u in range(5):
        for t in range(5):
            h = evaluate_heuristic(emb.phi[u], emb.phi[t], is_directed=True)
            d = ref_dist[u, t]
            assert h.item() <= d.item() + 1e-10, (
                f"Admissibility violated: h({u},{t})={h.item()} > d({u},{t})={d.item()}"
            )


def test_tropical_exactness_all_anchors(strongly_connected_directed_5):
    """With K=V=5 anchors on strongly connected graph, h(u,t) == d(u,t) (Theorem 2)."""
    from aac.embeddings.heuristic import evaluate_heuristic
    from aac.embeddings.sssp import compute_teacher_labels, scipy_dijkstra_batched
    from aac.embeddings.tropical import build_tropical_embedding

    all_anchors = torch.arange(5, dtype=torch.int64)
    labels = compute_teacher_labels(strongly_connected_directed_5, all_anchors, use_gpu=False)
    emb = build_tropical_embedding(labels)

    ref_dist = scipy_dijkstra_batched(strongly_connected_directed_5, all_anchors)  # (5, 5)

    for u in range(5):
        for t in range(5):
            if u == t:
                continue
            h = evaluate_heuristic(emb.phi[u], emb.phi[t], is_directed=True)
            d = ref_dist[u, t]
            assert abs(h.item() - d.item()) < 1e-10, (
                f"Exactness failed: h({u},{t})={h.item()} != d({u},{t})={d.item()}"
            )


# ---------------------------------------------------------------------------
# Heuristic evaluation tests (Task 2)
# ---------------------------------------------------------------------------


def test_heuristic_batch_shape(undirected_graph_5):
    """evaluate_heuristic_batch with B=10 queries returns (10,) tensor."""
    from aac.embeddings.heuristic import evaluate_heuristic_batch
    from aac.embeddings.hilbert import build_hilbert_embedding
    from aac.embeddings.sssp import compute_teacher_labels

    anchors = torch.tensor([0, 2, 4], dtype=torch.int64)
    labels = compute_teacher_labels(undirected_graph_5, anchors, use_gpu=False)
    emb = build_hilbert_embedding(labels)

    # Create 10 random queries by repeating source/target pairs
    sources = emb.phi[torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])]
    targets = emb.phi[torch.tensor([4, 3, 2, 1, 0, 2, 4, 0, 2, 1])]
    h = evaluate_heuristic_batch(sources, targets, is_directed=False)

    assert h.shape == (10,), f"Expected shape (10,), got {h.shape}"


def test_heuristic_nonnegative(undirected_graph_5):
    """All heuristic values must be >= -1e-10 (non-negative)."""
    from aac.embeddings.heuristic import evaluate_heuristic_batch
    from aac.embeddings.hilbert import build_hilbert_embedding
    from aac.embeddings.sssp import compute_teacher_labels

    all_anchors = torch.arange(5, dtype=torch.int64)
    labels = compute_teacher_labels(undirected_graph_5, all_anchors, use_gpu=False)
    emb = build_hilbert_embedding(labels)

    # All 25 pairs
    u_indices = torch.arange(5).repeat_interleave(5)
    t_indices = torch.arange(5).repeat(5)
    sources = emb.phi[u_indices]
    targets = emb.phi[t_indices]
    h = evaluate_heuristic_batch(sources, targets, is_directed=False)

    assert (h >= -1e-10).all(), f"Found negative heuristic values: {h[h < -1e-10]}"


def test_heuristic_asymmetric_directed(strongly_connected_directed_5):
    """On strongly connected directed graph, h(0,4) != h(4,0) due to asymmetric distances."""
    from aac.embeddings.heuristic import evaluate_heuristic
    from aac.embeddings.sssp import compute_teacher_labels, scipy_dijkstra_batched
    from aac.embeddings.tropical import build_tropical_embedding

    all_anchors = torch.arange(5, dtype=torch.int64)
    labels = compute_teacher_labels(strongly_connected_directed_5, all_anchors, use_gpu=False)
    emb = build_tropical_embedding(labels)

    h_04 = evaluate_heuristic(emb.phi[0], emb.phi[4], is_directed=True)
    h_40 = evaluate_heuristic(emb.phi[4], emb.phi[0], is_directed=True)

    # With Theorem 2 exactness (K=V), h(0,4) = d(0,4) and h(4,0) = d(4,0)
    # Since the graph is asymmetric, these should differ
    ref = scipy_dijkstra_batched(strongly_connected_directed_5, all_anchors)
    assert ref[0, 4].item() != ref[4, 0].item(), "Precondition: d(0,4) != d(4,0)"
    assert h_04.item() != h_40.item(), (
        f"Heuristic should be asymmetric: h(0,4)={h_04.item()}, h(4,0)={h_40.item()}"
    )


def test_evaluate_from_embedding(undirected_graph_5):
    """Convenience function produces same results as manual lookup + evaluate."""
    from aac.embeddings.heuristic import evaluate_from_embedding, evaluate_heuristic_batch
    from aac.embeddings.hilbert import build_hilbert_embedding
    from aac.embeddings.sssp import compute_teacher_labels

    all_anchors = torch.arange(5, dtype=torch.int64)
    labels = compute_teacher_labels(undirected_graph_5, all_anchors, use_gpu=False)
    emb = build_hilbert_embedding(labels)

    src_idx = torch.tensor([0, 1, 2])
    tgt_idx = torch.tensor([4, 3, 0])

    # Convenience function
    h_conv = evaluate_from_embedding(emb, src_idx, tgt_idx)
    # Manual
    h_manual = evaluate_heuristic_batch(emb.phi[src_idx], emb.phi[tgt_idx], is_directed=False)

    assert torch.allclose(h_conv, h_manual, atol=1e-15), (
        f"Convenience vs manual mismatch: {h_conv} vs {h_manual}"
    )


def test_hilbert_admissibility_medium_graph(medium_undirected_graph):
    """On 100-node graph with K=10 FPS anchors, h(u,t) <= d(u,t) for 200 random pairs."""
    from aac.embeddings.anchors import farthest_point_sampling
    from aac.embeddings.heuristic import evaluate_heuristic_batch
    from aac.embeddings.hilbert import build_hilbert_embedding
    from aac.embeddings.sssp import compute_teacher_labels, scipy_dijkstra_batched

    anchors = farthest_point_sampling(medium_undirected_graph, num_anchors=10, seed_vertex=0)
    labels = compute_teacher_labels(medium_undirected_graph, anchors, use_gpu=False)
    emb = build_hilbert_embedding(labels)

    # Generate 200 random (u,t) pairs
    torch.manual_seed(99)
    V = 100
    u_indices = torch.randint(0, V, (200,))
    t_indices = torch.randint(0, V, (200,))

    # Heuristic values
    h_vals = evaluate_heuristic_batch(emb.phi[u_indices], emb.phi[t_indices], is_directed=False)

    # Reference distances: need SSSP from all sources that appear in u_indices
    unique_sources = torch.unique(u_indices)
    ref_dist = scipy_dijkstra_batched(medium_undirected_graph, unique_sources)

    # Map u_indices to rows in ref_dist
    src_map = {s.item(): i for i, s in enumerate(unique_sources)}
    for q in range(200):
        u = u_indices[q].item()
        t = t_indices[q].item()
        d = ref_dist[src_map[u], t].item()
        h = h_vals[q].item()
        assert h <= d + 1e-10, (
            f"Admissibility violated at query {q}: h({u},{t})={h} > d({u},{t})={d}"
        )
