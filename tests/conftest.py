"""Shared test fixtures for AAC test suite."""

import pytest
import torch

from aac.graphs.types import Graph


def _build_csr_from_edges(
    sources: list[int],
    targets: list[int],
    weights: list[float],
    num_nodes: int,
    is_directed: bool,
) -> Graph:
    """Build a Graph in CSR format from edge lists.

    For undirected graphs, automatically adds reverse edges.
    """
    src = list(sources)
    tgt = list(targets)
    wgt = list(weights)

    if not is_directed:
        # Add reverse edges
        src_rev = list(targets)
        tgt_rev = list(sources)
        wgt_rev = list(weights)
        src = src + src_rev
        tgt = tgt + tgt_rev
        wgt = wgt + wgt_rev

    num_edges = len(src)

    # Sort by (source, target) for CSR construction
    edge_order = sorted(range(num_edges), key=lambda i: (src[i], tgt[i]))
    src_sorted = [src[i] for i in edge_order]
    tgt_sorted = [tgt[i] for i in edge_order]
    wgt_sorted = [wgt[i] for i in edge_order]

    # Build crow_indices
    crow_indices = [0] * (num_nodes + 1)
    for s in src_sorted:
        crow_indices[s + 1] += 1
    for i in range(1, num_nodes + 1):
        crow_indices[i] += crow_indices[i - 1]

    return Graph(
        crow_indices=torch.tensor(crow_indices, dtype=torch.int64),
        col_indices=torch.tensor(tgt_sorted, dtype=torch.int64),
        values=torch.tensor(wgt_sorted, dtype=torch.float64),
        num_nodes=num_nodes,
        num_edges=num_edges,
        is_directed=is_directed,
    )


@pytest.fixture
def small_directed_graph() -> Graph:
    """5-node directed graph with known shortest paths.

    Edges: (0,1,2), (0,2,5), (1,2,1), (1,3,6), (2,3,2), (2,4,7), (3,4,1)
    Known: d(0,4) = 6 via 0->1->2->3->4
    """
    sources = [0, 0, 1, 1, 2, 2, 3]
    targets = [1, 2, 2, 3, 3, 4, 4]
    weights = [2.0, 5.0, 1.0, 6.0, 2.0, 7.0, 1.0]
    return _build_csr_from_edges(sources, targets, weights, num_nodes=5, is_directed=True)


@pytest.fixture
def small_undirected_graph() -> Graph:
    """5-node undirected graph with known shortest paths.

    Edges: (0,1,1), (0,2,4), (1,2,2), (1,3,5), (2,3,1), (3,4,3)
    Known: d(0,4) = 7 via 0->1->2->3->4
    Stored as symmetric directed edges (both directions).
    """
    sources = [0, 0, 1, 1, 2, 3]
    targets = [1, 2, 2, 3, 3, 4]
    weights = [1.0, 4.0, 2.0, 5.0, 1.0, 3.0]
    return _build_csr_from_edges(sources, targets, weights, num_nodes=5, is_directed=False)


@pytest.fixture
def medium_random_graph() -> Graph:
    """100-node random connected graph with positive integer weights in [1, 100].

    Generated with fixed seed=42 using torch.manual_seed(42).
    Ensures connectivity by first building a spanning tree, then adding random edges.
    """
    torch.manual_seed(42)
    num_nodes = 100
    sources = []
    targets = []
    weights = []

    # Build spanning tree: connect node i to a random node in [0, i-1]
    for i in range(1, num_nodes):
        j = torch.randint(0, i, (1,)).item()
        w = torch.randint(1, 101, (1,)).item()
        sources.append(i)
        targets.append(j)
        weights.append(float(w))

    # Add extra random edges for density
    num_extra = 300
    for _ in range(num_extra):
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()
        if u != v:
            w = torch.randint(1, 101, (1,)).item()
            sources.append(u)
            targets.append(v)
            weights.append(float(w))

    return _build_csr_from_edges(sources, targets, weights, num_nodes=num_nodes, is_directed=True)


@pytest.fixture
def device() -> torch.device:
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 8-node graph fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def directed_8() -> Graph:
    """8-node strongly connected directed graph."""
    from aac.graphs.convert import edges_to_graph

    s = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 5, 6], dtype=torch.int64)
    t = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0, 3, 4, 5, 6, 0, 1], dtype=torch.int64)
    w = torch.tensor(
        [2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 1.0, 3.0, 7.0, 6.0, 4.0, 8.0, 9.0, 2.0],
        dtype=torch.float64,
    )
    return edges_to_graph(s, t, w, 8, is_directed=True)


@pytest.fixture
def strongly_connected_directed(directed_8) -> Graph:
    """Same graph as ``directed_8``."""
    return directed_8


@pytest.fixture
def undirected_8() -> Graph:
    """8-node undirected graph with varied edge weights."""
    from aac.graphs.convert import edges_to_graph

    s = torch.tensor([0, 0, 1, 1, 2, 3, 3, 4, 5, 6], dtype=torch.int64)
    t = torch.tensor([1, 3, 2, 4, 3, 4, 5, 5, 6, 7], dtype=torch.int64)
    w = torch.tensor(
        [2.0, 7.0, 3.0, 6.0, 1.0, 4.0, 2.0, 5.0, 1.0, 3.0],
        dtype=torch.float64,
    )
    return edges_to_graph(s, t, w, 8, is_directed=False)


@pytest.fixture
def weakly_connected_directed() -> Graph:
    """Directed graph that is weakly but NOT strongly connected.

    Nodes 0-4 form a chain: 0->1->2->3->4
    Node 5 connects back: 5->0
    But 4 cannot reach any node (dead end), and 0 cannot be reached from 1.
    Landmarks at 0 and 4 create varying finite-landmark sets.
    """
    from aac.graphs.convert import edges_to_graph

    s = torch.tensor([0, 1, 2, 3, 5, 0, 2], dtype=torch.int64)
    t = torch.tensor([1, 2, 3, 4, 0, 3, 5], dtype=torch.int64)
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 5.0, 3.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 6, is_directed=True)

