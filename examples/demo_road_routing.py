#!/usr/bin/env python
"""Demo 2: Road Network Routing -- Memory vs Accuracy Tradeoff.

Shows how AAC provides a Pareto-optimal memory-accuracy tradeoff
on a synthetic road network. At the same memory budget, AAC can match or
exceed ALT quality by selecting from a larger candidate pool.

Key insight: ALT with K landmarks stores K values/vertex. AAC with K0
candidates compressed to m stores m values/vertex. When m < K and K0 > K,
AAC can achieve similar quality at lower memory by learning which
landmarks matter most.

Usage:
    python examples/demo_road_routing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor
from aac.compression.smooth import make_aac_heuristic
from aac.graphs.convert import edges_to_graph
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor


def build_road_network(n_nodes: int, seed: int = 42) -> tuple:
    """Build a synthetic road network with spatial structure.

    Creates a random geometric graph: nodes are placed in 2D, edges connect
    nearby nodes with weights proportional to Euclidean distance.

    Args:
        n_nodes: Number of intersections.
        seed: Random seed.

    Returns:
        (graph, coords) where coords is (V, 2).
    """
    gen = torch.Generator().manual_seed(seed)
    coords = torch.rand(n_nodes, 2, generator=gen, dtype=torch.float64)

    # Connect nodes within radius, plus some long-range connections
    radius = 2.5 / (n_nodes ** 0.5)  # ~8-12 neighbors on average
    sources, targets, weights = [], [], []

    for i in range(n_nodes):
        dists = ((coords - coords[i]) ** 2).sum(dim=1).sqrt()
        neighbors = (dists < radius).nonzero(as_tuple=True)[0]
        for j in neighbors:
            j = j.item()
            if j > i:
                w = dists[j].item()
                sources.append(i)
                targets.append(j)
                weights.append(w)

    graph = edges_to_graph(
        torch.tensor(sources, dtype=torch.int64),
        torch.tensor(targets, dtype=torch.int64),
        torch.tensor(weights, dtype=torch.float64),
        num_nodes=n_nodes,
        is_directed=False,
        coordinates=coords,
    )
    return graph, coords


def benchmark_method(
    graph, method_name: str, heuristic, queries: list[tuple[int, int]]
) -> dict:
    """Run queries and collect expansion stats."""
    total_exp = 0
    total_dij = 0
    n_valid = 0
    for s, t in queries:
        r = astar(graph, s, t, heuristic)
        r_dij = dijkstra(graph, s, t)
        if r_dij.cost == float("inf"):
            continue
        total_exp += r.expansions
        total_dij += r_dij.expansions
        n_valid += 1
    reduction = (1 - total_exp / total_dij) * 100 if total_dij > 0 else 0
    return {"name": method_name, "expansions": total_exp, "reduction": reduction}


def main() -> None:
    torch.manual_seed(42)
    V = 500

    graph, coords = build_road_network(V)
    print("=" * 65)
    print("  AAC Demo: Road Network Memory-Accuracy Tradeoff")
    print("=" * 65)
    print(f"  Network: {V} intersections, {graph.num_edges} road segments")
    print()

    # Generate random queries
    gen = torch.Generator().manual_seed(99)
    queries = []
    for _ in range(100):
        s = torch.randint(0, V, (1,), generator=gen).item()
        t = torch.randint(0, V, (1,), generator=gen).item()
        if s != t:
            queries.append((s, t))

    # Sweep memory budgets and compare ALT vs AAC
    print(f"  {'Method':<25} {'Memory':>10} {'Expansions':>12} {'Reduction':>10}")
    print(f"  {'-' * 57}")

    # Dijkstra baseline
    zero_h = lambda n, t: 0.0
    dij_stats = benchmark_method(graph, "Dijkstra", zero_h, queries)
    print(f"  {'Dijkstra':<25} {'0':>10} {dij_stats['expansions']:>12} {'---':>10}")

    results = []

    # ALT at various K
    for K in [4, 8, 16, 32]:
        teacher = alt_preprocess(graph, K)
        h_alt = make_alt_heuristic(teacher)
        stats = benchmark_method(graph, f"ALT K={K}", h_alt, queries)
        mem = K  # values per vertex
        print(f"  {'ALT K=' + str(K):<25} {str(mem) + ' val/v':>10}"
              f" {stats['expansions']:>12} {stats['reduction']:>9.1f}%")
        results.append(("ALT", K, mem, stats["reduction"]))

    print()

    # AAC at various (K0, m) -- same memory as ALT K=m but selected from K0 candidates
    for K0, m in [(8, 4), (16, 4), (16, 8), (32, 8), (32, 16)]:
        teacher = alt_preprocess(graph, K0)
        compressor = LinearCompressor(K=K0, m=m, is_directed=False)
        config = TrainConfig(num_epochs=300, batch_size=256, lr=1e-2, seed=42, patience=30)
        train_linear_compressor(compressor, teacher, config)

        compressor.eval()
        with torch.no_grad():
            d_out_t = teacher.d_out.t().to(torch.float64)
            compressed = compressor(d_out_t)
        h_aac = make_aac_heuristic(compressed, is_directed=False)
        stats = benchmark_method(graph, f"AAC K0={K0}->m={m}", h_aac, queries)
        mem = m  # values per vertex
        print(f"  {'AAC K0=' + str(K0) + '->m=' + str(m):<25} {str(mem) + ' val/v':>10}"
              f" {stats['expansions']:>12} {stats['reduction']:>9.1f}%")
        results.append(("AAC", K0, mem, stats["reduction"]))

    # Pareto analysis
    print()
    print(f"  {'=' * 57}")
    print("  Pareto Analysis: Memory Budget vs Expansion Reduction")
    print(f"  {'=' * 57}")

    # Group by memory budget
    budgets: dict[int, list] = {}
    for method, K_or_K0, mem, reduction in results:
        budgets.setdefault(mem, []).append((method, K_or_K0, reduction))

    for mem in sorted(budgets):
        entries = budgets[mem]
        print(f"\n  Memory = {mem} values/vertex:")
        for method, K_or_K0, reduction in sorted(entries, key=lambda x: -x[2]):
            label = f"    {method} (K{'0' if method == 'AAC' else ''}={K_or_K0})"
            print(f"  {label:<30} {reduction:.1f}% reduction")

    print()
    print("  Key insight: At the same memory budget, AAC with a larger")
    print("  candidate pool (K0 > m) can match or exceed ALT quality")
    print("  by learning which landmarks to keep.")
    print("=" * 65)


if __name__ == "__main__":
    main()
