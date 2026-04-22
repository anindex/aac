"""Adversarial graph fixtures for deep correctness testing.

Each fixture returns a Graph constructed to stress a specific edge case:
sentinel propagation, SCC boundaries, float32 cancellation, etc.
"""

import torch
from aac.graphs.convert import edges_to_graph
from aac.graphs.types import Graph


def disconnected_directed() -> Graph:
    """Two disconnected 3-node directed components.

    Component A: 0 -> 1 -> 2 -> 0  (cycle, weights 1.0)
    Component B: 3 -> 4 -> 5 -> 3  (cycle, weights 2.0)

    No edges between A and B. Cross-component queries should produce
    sentinel distances and heuristic h=0.
    """
    s = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
    t = torch.tensor([1, 2, 0, 4, 5, 3], dtype=torch.int64)
    w = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 6, is_directed=True)


def weakly_not_strongly_connected() -> Graph:
    """6-node one-way chain: 0 -> 1 -> 2 -> 3 -> 4 -> 5.

    Weakly connected but NOT strongly connected (no back-edges).
    The largest SCC is any single node. Stress-tests compute_strong_lcc()
    and query generation which must restrict to strong components.
    """
    s = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    t = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 6, is_directed=True)


def scc_boundary_graph() -> Graph:
    """Two 4-node SCCs connected by a one-way bridge.

    SCC-A: {0,1,2,3} (strongly connected cycle + shortcuts)
    SCC-B: {4,5,6,7} (strongly connected cycle + shortcuts)
    Bridge: 3 -> 4 (one-way, so SCC-A can reach SCC-B but not vice versa)

    Landmarks in SCC-B produce sentinel d_in for nodes in SCC-A.
    Landmarks in SCC-A produce sentinel d_in for... actually reverse SSSP
    from SCC-A cannot reach SCC-B. Tests cross-SCC sentinel masking.
    """
    edges = [
        # SCC-A cycle + shortcuts
        (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0),
        (0, 2, 3.0), (1, 3, 3.0),
        # Bridge
        (3, 4, 2.0),
        # SCC-B cycle + shortcuts
        (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0), (7, 4, 1.0),
        (4, 6, 3.0), (5, 7, 3.0),
    ]
    s = torch.tensor([e[0] for e in edges], dtype=torch.int64)
    t = torch.tensor([e[1] for e in edges], dtype=torch.int64)
    w = torch.tensor([e[2] for e in edges], dtype=torch.float64)
    return edges_to_graph(s, t, w, 8, is_directed=True)


def extreme_weights_graph() -> Graph:
    """8-node directed graph with edge weights spanning 1e-6 to 1e6.

    Stresses float32 cancellation: path sums can reach ~1e6 while
    differences between paths can be ~1e-6, exceeding float32 ULP.
    The graph is strongly connected.
    """
    edges = [
        # Long-distance backbone
        (0, 1, 1e6), (1, 2, 1e-6), (2, 3, 1e6),
        (3, 4, 1e-6), (4, 5, 1e3), (5, 6, 1e-3),
        (6, 7, 1e3), (7, 0, 1e-3),
        # Back-edges for strong connectivity
        (3, 0, 5e6), (7, 4, 5e3),
        # Shortcuts
        (0, 4, 2e6 + 1.0), (4, 0, 5e3 + 1.0),
    ]
    s = torch.tensor([e[0] for e in edges], dtype=torch.int64)
    t = torch.tensor([e[1] for e in edges], dtype=torch.int64)
    w = torch.tensor([e[2] for e in edges], dtype=torch.float64)
    return edges_to_graph(s, t, w, 8, is_directed=True)


def duplicate_teacher_landmarks() -> Graph:
    """Graph where FPS is likely to produce near-duplicate landmarks.

    10-node star graph: center node 0 connected to 1..9.
    Nodes 1..9 are at identical distances from center.
    FPS starting from 0 may select multiple near-equivalent leaves,
    testing deduplicate_selections() in the compressor.
    """
    s_list, t_list, w_list = [], [], []
    # Star edges: center (0) to leaves (1..9)
    for i in range(1, 10):
        s_list.extend([0, i])
        t_list.extend([i, 0])
        w_list.extend([1.0, 1.0])
    # Tiny inter-leaf connections for strong connectivity
    for i in range(1, 9):
        s_list.extend([i, i + 1])
        t_list.extend([i + 1, i])
        w_list.extend([10.0, 10.0])
    # Close the ring
    s_list.extend([9, 1])
    t_list.extend([1, 9])
    w_list.extend([10.0, 10.0])

    s = torch.tensor(s_list, dtype=torch.int64)
    t = torch.tensor(t_list, dtype=torch.int64)
    w = torch.tensor(w_list, dtype=torch.float64)
    return edges_to_graph(s, t, w, 10, is_directed=True)


def partial_reachability_graph() -> Graph:
    """8-node directed graph: SCC {0,1,2,3} + dangling tail {4,5,6,7}.

    SCC: 0->1->2->3->0 (strongly connected, weights 1.0)
    Tail: 3->4->5->6->7 (one-way, no return)

    Nodes 4-7 are reachable FROM the SCC but cannot reach back.
    Forward SSSP from SCC landmarks reaches all 8 nodes.
    Reverse SSSP (for d_in) from SCC landmarks: only SCC nodes are reachable.
    So d_in for nodes 4-7 = SENTINEL.

    Tests sentinel masking: heuristic must return 0 for queries involving
    tail nodes in the backward direction.
    """
    edges = [
        # SCC
        (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0),
        (0, 2, 3.0), (1, 3, 3.0),
        # Dangling tail
        (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0),
    ]
    s = torch.tensor([e[0] for e in edges], dtype=torch.int64)
    t = torch.tensor([e[1] for e in edges], dtype=torch.int64)
    w = torch.tensor([e[2] for e in edges], dtype=torch.float64)
    return edges_to_graph(s, t, w, 8, is_directed=True)


def float32_cancellation_graph() -> Graph:
    """4-node graph with weights at float32 ULP boundaries.

    Designed so that d(0,3) - d(1,3) is very small relative to d(0,3),
    exercising float32 cancellation. At values near 100,000 the float32
    ULP is ~0.008, so differences below that are lost.

    Path 0->1: weight 99999.0
    Path 1->2: weight 1.0
    Path 2->3: weight 0.004  (below float32 ULP at ~100000)
    Path 0->3 direct: weight 100000.004

    In float64: d(0,3) via 0->1->2->3 = 100000.004
    In float32: 99999.0 + 1.0 + 0.004 = 100000.0 (0.004 lost to rounding)

    This graph is strongly connected (bidirectional edges).
    """
    edges = [
        (0, 1, 99999.0), (1, 0, 99999.0),
        (1, 2, 1.0), (2, 1, 1.0),
        (2, 3, 0.004), (3, 2, 0.004),
        (0, 3, 100000.004), (3, 0, 100000.004),
        # Extra connectivity
        (0, 2, 100000.0), (2, 0, 100000.0),
        (1, 3, 1.004), (3, 1, 1.004),
    ]
    s = torch.tensor([e[0] for e in edges], dtype=torch.int64)
    t = torch.tensor([e[1] for e in edges], dtype=torch.int64)
    w = torch.tensor([e[2] for e in edges], dtype=torch.float64)
    return edges_to_graph(s, t, w, 4, is_directed=False)


def strongly_connected_directed_10() -> Graph:
    """10-node strongly connected directed graph for general testing.

    Has varied edge weights and multiple shortest-path options.
    Good baseline for property-based tests.
    """
    edges = [
        # Hamiltonian cycle
        (0, 1, 2.0), (1, 2, 3.0), (2, 3, 1.0), (3, 4, 4.0),
        (4, 5, 2.0), (5, 6, 5.0), (6, 7, 1.0), (7, 8, 3.0),
        (8, 9, 2.0), (9, 0, 3.0),
        # Shortcuts creating alternative paths
        (0, 3, 7.0), (1, 4, 6.0), (2, 5, 4.0),
        (3, 6, 8.0), (4, 7, 6.0), (5, 8, 4.0),
        (6, 9, 3.0), (7, 0, 9.0), (8, 1, 7.0), (9, 2, 5.0),
    ]
    s = torch.tensor([e[0] for e in edges], dtype=torch.int64)
    t = torch.tensor([e[1] for e in edges], dtype=torch.int64)
    w = torch.tensor([e[2] for e in edges], dtype=torch.float64)
    return edges_to_graph(s, t, w, 10, is_directed=True)
