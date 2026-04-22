#!/usr/bin/env python
"""Demo 1: 2D Grid Navigation with Learned Heuristics.

Shows how AAC learns a compressed admissible heuristic for A* search
on a 2D grid with obstacles. Compares:
  - Dijkstra (no heuristic, explores everything)
  - ALT (classical landmark heuristic)
  - AAC (learned compressed heuristic)

This is the simplest possible use case: given a grid world, precompute a
small heuristic table that speeds up pathfinding at query time.

Usage:
    python examples/demo_grid_navigation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor
from aac.compression.smooth import make_aac_heuristic
from aac.graphs.convert import edges_to_graph
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor


def build_grid_graph(grid_size: int, obstacles: set[tuple[int, int]]) -> tuple:
    """Build an 8-connected grid graph with obstacles.

    Args:
        grid_size: Width and height of the grid.
        obstacles: Set of (row, col) cells that are impassable.

    Returns:
        (graph, node_to_rc, rc_to_node) where graph is in CSR format.
    """
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    sqrt2 = 1.4142135623730951

    node_to_rc: dict[int, tuple[int, int]] = {}
    rc_to_node: dict[tuple[int, int], int] = {}
    node_id = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) not in obstacles:
                node_to_rc[node_id] = (r, c)
                rc_to_node[(r, c)] = node_id
                node_id += 1

    num_nodes = node_id
    sources, targets, weights = [], [], []

    for nid, (r, c) in node_to_rc.items():
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if (nr, nc) in rc_to_node:
                neighbor_id = rc_to_node[(nr, nc)]
                if nid < neighbor_id:
                    w = sqrt2 if (dr != 0 and dc != 0) else 1.0
                    sources.append(nid)
                    targets.append(neighbor_id)
                    weights.append(w)

    graph = edges_to_graph(
        torch.tensor(sources, dtype=torch.int64),
        torch.tensor(targets, dtype=torch.int64),
        torch.tensor(weights, dtype=torch.float64),
        num_nodes=num_nodes,
        is_directed=False,
    )
    return graph, node_to_rc, rc_to_node


def print_grid(
    grid_size: int,
    obstacles: set[tuple[int, int]],
    path: list[int],
    node_to_rc: dict[int, tuple[int, int]],
    src_rc: tuple[int, int],
    tgt_rc: tuple[int, int],
) -> None:
    """Print ASCII visualization of the grid with path."""
    path_cells = {node_to_rc[n] for n in path if n in node_to_rc}
    grid = []
    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            if (r, c) == src_rc:
                row.append("S")
            elif (r, c) == tgt_rc:
                row.append("G")
            elif (r, c) in obstacles:
                row.append("#")
            elif (r, c) in path_cells:
                row.append("*")
            else:
                row.append(".")
        grid.append(" ".join(row))
    print("\n".join(grid))


def main() -> None:
    torch.manual_seed(42)
    grid_size = 20

    # Create obstacle pattern (walls with gaps)
    obstacles: set[tuple[int, int]] = set()
    # Vertical wall at col 5, gap at rows 3-4
    for r in range(grid_size):
        if r not in (3, 4):
            obstacles.add((r, 5))
    # Vertical wall at col 14, gap at rows 15-16
    for r in range(grid_size):
        if r not in (15, 16):
            obstacles.add((r, 14))
    # Horizontal wall at row 10, gap at cols 8-9
    for c in range(grid_size):
        if c not in (8, 9) and (10, c) not in obstacles:
            obstacles.add((10, c))

    graph, node_to_rc, rc_to_node = build_grid_graph(grid_size, obstacles)
    V = graph.num_nodes

    print("=" * 60)
    print("  AAC Demo: 2D Grid Navigation")
    print("=" * 60)
    print(f"  Grid: {grid_size}x{grid_size} with obstacles")
    print(f"  Nodes: {V} (after removing obstacles)")
    print(f"  Edges: {graph.num_edges}")
    print()

    # Source and target
    src_rc, tgt_rc = (0, 0), (19, 19)
    src = rc_to_node[src_rc]
    tgt = rc_to_node[tgt_rc]

    # === Method 1: Dijkstra (baseline) ===
    result_dij = dijkstra(graph, src, tgt)
    print(f"[Dijkstra]  Cost: {result_dij.cost:.2f}  Expansions: {result_dij.expansions}")

    # === Method 2: ALT with K=16 landmarks ===
    K = 16
    teacher_alt = alt_preprocess(graph, K)
    h_alt = make_alt_heuristic(teacher_alt)
    result_alt = astar(graph, src, tgt, h_alt)
    alt_reduction = (1 - result_alt.expansions / result_dij.expansions) * 100
    print(
        f"[ALT K={K}]  Cost: {result_alt.cost:.2f}  Expansions: {result_alt.expansions}"
        f"  ({alt_reduction:.1f}% reduction)"
    )

    # === Method 3: AAC -- select m=16 landmarks from K0=32 pool ===
    m = 16
    K0 = 32
    teacher_aac = alt_preprocess(graph, K0)
    compressor = LinearCompressor(K=K0, m=m, is_directed=False)

    config = TrainConfig(
        num_epochs=300,
        batch_size=256,
        lr=1e-2,
        cond_lambda=0.01,
        T_init=1.0,
        gamma=1.05,
        seed=42,
        patience=30,
    )

    _metrics = train_linear_compressor(compressor, teacher_aac, config)

    # Get compressed labels for heuristic
    compressor.eval()
    with torch.no_grad():
        d_out_t = teacher_aac.d_out.t().to(torch.float64)  # (V, K0)
        compressed = compressor(d_out_t)  # (V, m)

    h_aac = make_aac_heuristic(compressed, is_directed=False)
    result_aac = astar(graph, src, tgt, h_aac)
    aac_reduction = (1 - result_aac.expansions / result_dij.expansions) * 100
    print(
        f"[AAC m={m}] Cost: {result_aac.cost:.2f}  Expansions: {result_aac.expansions}"
        f"  ({aac_reduction:.1f}% reduction)"
    )

    # Memory comparison
    print()
    if m == K:
        print(f"  Memory: ALT = {K} values/vertex, AAC = {m} values/vertex (matched)")
    else:
        print(f"  Memory: ALT = {K} values/vertex, AAC = {m} values/vertex"
              f" ({(1 - m / K) * 100:.0f}% less)")
    print(f"  Selected landmarks: {compressor.selected_landmarks()}")

    # Verify admissibility (A* found optimal path in all cases)
    assert abs(result_dij.cost - result_alt.cost) < 1e-6, "ALT path not optimal!"
    assert abs(result_dij.cost - result_aac.cost) < 1e-6, "AAC path not optimal!"
    print(f"\n  All paths optimal (cost = {result_dij.cost:.2f})")

    # Show the grid with AAC path
    print("\n  Grid (S=start, G=goal, #=wall, *=path):")
    print()
    print_grid(grid_size, obstacles, result_aac.path, node_to_rc, src_rc, tgt_rc)

    # Run multiple queries to show average speedup
    print(f"\n{'=' * 60}")
    print("  Benchmark: 50 random queries")
    print("=" * 60)

    gen = torch.Generator().manual_seed(123)
    n_queries = 50
    dij_total, alt_total, aac_total = 0, 0, 0

    for _ in range(n_queries):
        s = torch.randint(0, V, (1,), generator=gen).item()
        t = torch.randint(0, V, (1,), generator=gen).item()
        if s == t:
            continue
        r_dij = dijkstra(graph, s, t)
        if r_dij.cost == float("inf"):
            continue  # skip unreachable pairs
        r_alt = astar(graph, s, t, h_alt)
        r_aac = astar(graph, s, t, h_aac)
        dij_total += r_dij.expansions
        alt_total += r_alt.expansions
        aac_total += r_aac.expansions

    print(f"  Dijkstra total expansions: {dij_total}")
    print(f"  ALT (K={K}) total expansions: {alt_total}"
          f" ({(1 - alt_total / dij_total) * 100:.1f}% reduction)")
    print(f"  AAC (m={m}) total expansions: {aac_total}"
          f" ({(1 - aac_total / dij_total) * 100:.1f}% reduction)")
    if m == K:
        print(f"\n  AAC and ALT use matched memory ({m} values/vertex)")
    else:
        print(f"\n  AAC uses {(1 - m / K) * 100:.0f}% less memory than ALT")
    print("=" * 60)


if __name__ == "__main__":
    main()
