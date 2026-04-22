#!/usr/bin/env python
"""Synthetic graph experiments: AAC vs ALT on community (SBM) and power-law (BA) graphs.

Generates two graph families, tests at two budget levels each, and runs
paired Wilcoxon signed-rank tests across 3 training seeds.

Output:
    results/synthetic/community_results.csv
    results/synthetic/powerlaw_results.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import networkx as nx
import numpy as np
import scipy.stats
import torch

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.convert import edges_to_graph
from aac.heuristics import make_hybrid_heuristic
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = _PROJECT_ROOT / "results" / "synthetic"
GRAPH_SEED = 42  # Fixed structure across all training seeds
NUM_QUERIES = 100
QUERY_SEED = 42
SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds for parity with road-network experiments

# Budget levels: (label, aac_K0, aac_m, alt_K)
# On UNDIRECTED graphs (SBM, BA), ALT stores d_in = d_out as a single table,
# i.e., K floats/vertex (not 2K). Matched memory at B bytes/vertex therefore
# corresponds to AAC m = B/4 and ALT K = B/4 (equal landmark count).
# This differs from the directed-road-network convention where ALT stores
# 2K floats/vertex and matched memory corresponds to K = m/2. See
# aac.baselines.alt.alt_memory_bytes(..., is_directed=False) and
# aac.embeddings.sssp.compute_teacher_labels (d_in = d_out on undirected).
#
#  32 B/v: AAC m=8  floats (8*4=32),  ALT K=8  floats (8*4=32)
#  64 B/v: AAC m=16 floats (16*4=64), ALT K=16 floats (16*4=64)
# 128 B/v: AAC m=32 floats (32*4=128),ALT K=32 floats (32*4=128)
BUDGET_LEVELS = [
    {"label": "32", "aac_K0": 32, "aac_m": 8, "alt_K": 8},
    {"label": "64", "aac_K0": 64, "aac_m": 16, "alt_K": 16},
    {"label": "128", "aac_K0": 128, "aac_m": 32, "alt_K": 32},
]

# Query distribution modes for sensitivity analysis
QUERY_MODES = ["uniform"]  # hotspot/powerlaw less meaningful on synthetic without spatial structure


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------
def generate_community_graph(seed: int) -> nx.Graph:
    """Generate a Stochastic Block Model graph with 5 communities of 2K each (10K total)."""
    sizes = [2000, 2000, 2000, 2000, 2000]
    p_matrix = [
        [0.05, 0.001, 0.001, 0.001, 0.001],
        [0.001, 0.05, 0.001, 0.001, 0.001],
        [0.001, 0.001, 0.05, 0.001, 0.001],
        [0.001, 0.001, 0.001, 0.05, 0.001],
        [0.001, 0.001, 0.001, 0.001, 0.05],
    ]
    G = nx.stochastic_block_model(sizes, p_matrix, seed=seed)
    return G


def generate_powerlaw_graph(seed: int) -> nx.Graph:
    """Generate a Barabasi-Albert preferential attachment graph (10K nodes)."""
    G = nx.barabasi_albert_graph(10000, 5, seed=seed)
    return G


def nx_to_graph(G: nx.Graph, weight_seed: int = 42) -> "Graph":
    """Convert a networkx graph to AAC Graph format with random weights.

    Assigns random uniform [1, 10] weights to each edge and makes the graph
    undirected by adding both directions.
    """
    rng = np.random.RandomState(weight_seed)

    edges = list(G.edges())
    num_nx_edges = len(edges)
    num_nodes = G.number_of_nodes()

    sources = torch.zeros(num_nx_edges, dtype=torch.int64)
    targets = torch.zeros(num_nx_edges, dtype=torch.int64)
    weights = torch.zeros(num_nx_edges, dtype=torch.float64)

    for i, (u, v) in enumerate(edges):
        w = rng.uniform(1.0, 10.0)
        sources[i] = u
        targets[i] = v
        weights[i] = w

    # edges_to_graph handles adding reverse edges for undirected graphs
    graph = edges_to_graph(
        sources=sources,
        targets=targets,
        weights=weights,
        num_nodes=num_nodes,
        is_directed=False,
    )
    return graph


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    graph_type: str,
    graph: "Graph",
    budget_levels: list[dict],
    seeds: list[int],
    num_queries: int,
    query_seed: int,
) -> list[dict]:
    """Run AAC vs ALT comparison at multiple budget levels and seeds."""

    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    lcc_size = len(lcc_nodes)
    print(f"  LCC: {lcc_size:,} / {graph.num_nodes:,} nodes "
          f"({100.0 * lcc_size / graph.num_nodes:.1f}%)")

    if lcc_size < 0.8 * graph.num_nodes:
        print(f"  WARNING: LCC covers only {100.0 * lcc_size / graph.num_nodes:.1f}% "
              f"of nodes -- graph may be too disconnected")

    # Generate queries (same for all seeds and budgets)
    queries = generate_queries(graph, num_queries, seed=query_seed)

    # Dijkstra baseline (deterministic, run once)
    print("  Running Dijkstra baseline...")
    dij_results = [dijkstra(graph, s, t) for s, t in queries]
    dij_exps = np.array([r.expansions for r in dij_results])
    dij_mean = float(np.mean(dij_exps))
    print(f"  Dijkstra: mean={dij_mean:.0f} expansions")

    all_rows = []

    for budget in budget_levels:
        bpv_label = budget["label"]
        aac_K0 = budget["aac_K0"]
        aac_m = budget["aac_m"]
        alt_K = budget["alt_K"]

        print(f"\n  --- Budget {bpv_label} B/v: AAC(K0={aac_K0}, m={aac_m}) "
              f"vs ALT(K={alt_K}) ---")

        # ALT baseline (deterministic, run once per budget level)
        print(f"    ALT preprocessing (K={alt_K})...")
        t0 = time.perf_counter()
        teacher_alt = alt_preprocess(
            graph, alt_K, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
        )
        alt_h = make_alt_heuristic(teacher_alt)
        alt_prep_time = time.perf_counter() - t0
        alt_exps = np.array([
            astar(graph, s, t, heuristic=alt_h).expansions for s, t in queries
        ])
        alt_mean = float(np.mean(alt_exps))
        alt_reduction = 100.0 * (1.0 - alt_mean / dij_mean) if dij_mean > 0 else 0.0
        print(f"    ALT: mean={alt_mean:.0f} expansions "
              f"({alt_reduction:.1f}% reduction) [{alt_prep_time:.1f}s]")

        for seed in seeds:
            print(f"    Seed {seed}: AAC preprocessing...", end=" ", flush=True)
            torch.manual_seed(seed)
            t0 = time.perf_counter()

            # Anchor selection and teacher label computation
            anchors = farthest_point_sampling(
                graph, aac_K0, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
            )
            teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)

            # Train linear compressor
            compressor = LinearCompressor(K=aac_K0, m=aac_m, is_directed=graph.is_directed)
            cfg = TrainConfig(num_epochs=200, batch_size=256, lr=1e-3, seed=seed)
            train_linear_compressor(
                compressor, teacher_labels, cfg, valid_vertices=lcc_tensor,
            )

            # Extract compressed labels
            d_out_t = teacher_labels.d_out.t()  # (V, K)
            d_in_t = teacher_labels.d_in.t()    # (V, K)
            compressor.eval()
            with torch.no_grad():
                if graph.is_directed:
                    y_fwd, y_bwd = compressor(d_out_t, d_in_t)
                    y_fwd, y_bwd = y_fwd.detach(), y_bwd.detach()
                else:
                    y = compressor(d_out_t)
                    y_fwd = y_bwd = y.detach()

            aac_h = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)
            prep_time = time.perf_counter() - t0

            # Evaluate AAC
            aac_exps = np.array([
                astar(graph, s, t, heuristic=aac_h).expansions for s, t in queries
            ])
            aac_mean = float(np.mean(aac_exps))
            aac_reduction = 100.0 * (1.0 - aac_mean / dij_mean) if dij_mean > 0 else 0.0

            # Paired Wilcoxon signed-rank test: AAC vs ALT
            diff = aac_exps - alt_exps
            try:
                stat, p_two = scipy.stats.wilcoxon(diff, alternative="two-sided")
            except ValueError:
                # All differences are zero (identical results)
                stat, p_two = 0.0, 1.0

            # Hybrid: max(AAC, ALT) -- admissible by construction
            hybrid_h = make_hybrid_heuristic(aac_h, alt_h)
            hybrid_exps = np.array([
                astar(graph, s, t, heuristic=hybrid_h).expansions for s, t in queries
            ])
            hybrid_mean = float(np.mean(hybrid_exps))
            hybrid_reduction = 100.0 * (1.0 - hybrid_mean / dij_mean) if dij_mean > 0 else 0.0

            print(f"done ({prep_time:.1f}s) mean={aac_mean:.0f} "
                  f"({aac_reduction:.1f}% red.) hybrid={hybrid_mean:.0f} "
                  f"({hybrid_reduction:.1f}% red.) p={p_two:.2e}")

            all_rows.append({
                "graph_type": graph_type,
                "budget_bpv": int(bpv_label),
                "aac_K0": aac_K0,
                "aac_m": aac_m,
                "alt_K": alt_K,
                "seed": seed,
                "dij_mean_exp": f"{dij_mean:.1f}",
                "aac_mean_exp": f"{aac_mean:.1f}",
                "alt_mean_exp": f"{alt_mean:.1f}",
                "hybrid_mean_exp": f"{hybrid_mean:.1f}",
                "aac_reduction_pct": f"{aac_reduction:.2f}",
                "alt_reduction_pct": f"{alt_reduction:.2f}",
                "hybrid_reduction_pct": f"{hybrid_reduction:.2f}",
                "wilcoxon_stat": f"{stat:.1f}",
                "p_value_twosided": f"{p_two:.6e}",
            })

    return all_rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "graph_type",
    "budget_bpv",
    "aac_K0",
    "aac_m",
    "alt_K",
    "seed",
    "dij_mean_exp",
    "aac_mean_exp",
    "alt_mean_exp",
    "hybrid_mean_exp",
    "aac_reduction_pct",
    "alt_reduction_pct",
    "hybrid_reduction_pct",
    "wilcoxon_stat",
    "p_value_twosided",
]


def write_csv(rows: list[dict], path: Path) -> None:
    """Write results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Community graph (Stochastic Block Model)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  COMMUNITY GRAPH (Stochastic Block Model)")
    print("  5 communities x 2000 nodes = 10000 total, p_in=0.05, p_out=0.001")
    print(f"{'='*70}")

    G_comm = generate_community_graph(GRAPH_SEED)
    print(f"  NetworkX: {G_comm.number_of_nodes()} nodes, {G_comm.number_of_edges()} edges")

    # Check connectivity
    n_components = nx.number_connected_components(G_comm)
    if n_components > 1:
        largest_cc = max(nx.connected_components(G_comm), key=len)
        print(f"  WARNING: {n_components} connected components. "
              f"Largest CC: {len(largest_cc)} nodes")
    else:
        print("  Connected: 1 component")

    graph_comm = nx_to_graph(G_comm, weight_seed=GRAPH_SEED)
    print(f"  Graph: {graph_comm.num_nodes:,} nodes, {graph_comm.num_edges:,} edges "
          f"(directed={graph_comm.is_directed})")

    community_rows = run_experiment(
        graph_type="community_sbm",
        graph=graph_comm,
        budget_levels=BUDGET_LEVELS,
        seeds=SEEDS,
        num_queries=NUM_QUERIES,
        query_seed=QUERY_SEED,
    )
    write_csv(community_rows, OUTPUT_DIR / "community_results.csv")

    # -----------------------------------------------------------------------
    # Power-law graph (Barabasi-Albert)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  POWER-LAW GRAPH (Barabasi-Albert)")
    print("  10000 nodes, m=5 (attachments per new node)")
    print(f"{'='*70}")

    G_pl = generate_powerlaw_graph(GRAPH_SEED)
    print(f"  NetworkX: {G_pl.number_of_nodes()} nodes, {G_pl.number_of_edges()} edges")
    print(f"  Connected: {nx.is_connected(G_pl)}")

    graph_pl = nx_to_graph(G_pl, weight_seed=GRAPH_SEED)
    print(f"  Graph: {graph_pl.num_nodes:,} nodes, {graph_pl.num_edges:,} edges "
          f"(directed={graph_pl.is_directed})")

    powerlaw_rows = run_experiment(
        graph_type="powerlaw_ba",
        graph=graph_pl,
        budget_levels=BUDGET_LEVELS,
        seeds=SEEDS,
        num_queries=NUM_QUERIES,
        query_seed=QUERY_SEED,
    )
    write_csv(powerlaw_rows, OUTPUT_DIR / "powerlaw_results.csv")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for label, rows in [("Community", community_rows), ("Power-law", powerlaw_rows)]:
        print(f"\n  {label}:")
        for row in rows:
            sig = "***" if float(row["p_value_twosided"]) < 0.001 else (
                "**" if float(row["p_value_twosided"]) < 0.01 else (
                    "*" if float(row["p_value_twosided"]) < 0.05 else "ns"))
            print(f"    {row['budget_bpv']}B/v seed={row['seed']}: "
                  f"AAC={row['aac_mean_exp']} ALT={row['alt_mean_exp']} "
                  f"Hybrid={row['hybrid_mean_exp']} "
                  f"Dij={row['dij_mean_exp']} "
                  f"AAC_red={row['aac_reduction_pct']}% "
                  f"ALT_red={row['alt_reduction_pct']}% "
                  f"Hybrid_red={row['hybrid_reduction_pct']}% "
                  f"p={row['p_value_twosided']} {sig}")

    print("\nOutput files:")
    print(f"  {OUTPUT_DIR / 'community_results.csv'}")
    print(f"  {OUTPUT_DIR / 'powerlaw_results.csv'}")


if __name__ == "__main__":
    main()
