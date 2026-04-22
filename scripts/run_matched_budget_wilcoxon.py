#!/usr/bin/env python
"""Wilcoxon signed-rank tests at matched storage budgets: AAC vs ALT.

Tests 3 matched-budget configurations on all 4 DIMACS graphs:
  - 32 B/v:  AAC (K0=64, m=8)  vs ALT (K=4)
  - 64 B/v:  AAC (K0=64, m=16) vs ALT (K=8)
  - 128 B/v: AAC (K0=64, m=32) vs ALT (K=16)

ALT stores 2*K float32 values per vertex (fwd + bwd), so K landmarks = 8*K B/v.
AAC stores m float32 values per vertex, so m dims = 4*m B/v.

For each config: 100 queries, 5 seeds [42, 123, 456, 789, 1024].
Paired Wilcoxon signed-rank test (two-sided) per seed.

Output: results/dimacs/wilcoxon_matched_budget.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Ensure project root and src are importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import torch
from scipy.stats import wilcoxon

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.loaders.dimacs import load_dimacs
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(_project_root) / "data" / "dimacs"
OUTPUT_DIR = Path(_project_root) / "results" / "dimacs"

DIMACS_GRAPHS = {
    "NY": ("USA-road-d.NY.gr", "USA-road-d.NY.co"),
    "BAY": ("USA-road-d.BAY.gr", "USA-road-d.BAY.co"),
    "COL": ("USA-road-d.COL.gr", "USA-road-d.COL.co"),
    "FLA": ("USA-road-d.FLA.gr", "USA-road-d.FLA.co"),
}

# Matched-budget configs: (budget_bpv, aac_K0, aac_m, alt_K)
# ALT: 2*K floats/v * 4 bytes = 8*K B/v   =>  K = budget / 8
# AAC: m  floats/v * 4 bytes = 4*m B/v     =>  m = budget / 4
BUDGET_CONFIGS = [
    (32,  64,  8,  4),   # 32 B/v
    (64,  64, 16,  8),   # 64 B/v
    (128, 64, 32, 16),   # 128 B/v
]

NUM_QUERIES = 100
SEEDS = [42, 123, 456, 789, 1024]
QUERY_SEED = 42  # Fixed seed for query generation (same queries across all runs)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for graph_name, (gr_file, co_file) in DIMACS_GRAPHS.items():
        gr_path = DATA_DIR / gr_file
        co_path = DATA_DIR / co_file
        if not gr_path.exists():
            print(f"SKIP: {gr_path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"  Graph: {graph_name}")
        print(f"{'='*70}")

        graph = load_dimacs(str(gr_path), str(co_path) if co_path.exists() else None)
        print(f"  {graph.num_nodes:,} nodes")

        # LCC (computed once per graph)
        lcc_nodes, lcc_seed = compute_strong_lcc(graph)
        lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
        print(f"  LCC: {len(lcc_nodes):,} nodes, seed vertex: {lcc_seed}")

        # Queries (fixed across all seeds and budgets)
        queries = generate_queries(graph, NUM_QUERIES, seed=QUERY_SEED)

        # Dijkstra baseline (once per graph)
        print("  Running Dijkstra baseline...", end=" ", flush=True)
        t0 = time.perf_counter()
        dij_exps = [dijkstra(graph, s, t).expansions for s, t in queries]
        dij_mean = np.mean(dij_exps)
        print(f"done ({time.perf_counter() - t0:.1f}s), mean expansions: {dij_mean:.0f}")

        for budget_bpv, aac_K0, aac_m, alt_K in BUDGET_CONFIGS:
            print(f"\n  --- Budget: {budget_bpv} B/v  "
                  f"(AAC K0={aac_K0} m={aac_m} vs ALT K={alt_K}) ---")

            # ALT preprocessing (deterministic, run once per budget config)
            print(f"  ALT (K={alt_K}) preprocessing...", end=" ", flush=True)
            t0 = time.perf_counter()
            alt_teacher = alt_preprocess(
                graph, alt_K, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
            )
            alt_h = make_alt_heuristic(alt_teacher)
            alt_exps = [astar(graph, s, t, heuristic=alt_h).expansions for s, t in queries]
            alt_time = time.perf_counter() - t0
            alt_mean = np.mean(alt_exps)
            alt_reduction = (1.0 - alt_mean / dij_mean) * 100.0
            print(f"done ({alt_time:.1f}s), mean exp: {alt_mean:.0f} "
                  f"({alt_reduction:.1f}% reduction)")

            for seed in SEEDS:
                print(f"    Seed {seed}: AAC (K0={aac_K0}, m={aac_m}) ...",
                      end=" ", flush=True)
                torch.manual_seed(seed)
                t0 = time.perf_counter()

                # AAC preprocessing: anchors + teacher labels + compression
                anchors = farthest_point_sampling(
                    graph, aac_K0, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
                )
                teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)

                compressor = LinearCompressor(
                    K=aac_K0, m=aac_m, is_directed=graph.is_directed,
                )
                cfg = TrainConfig(
                    num_epochs=200,
                    batch_size=256,
                    lr=1e-3,
                    cond_lambda=0.01,
                    T_init=1.0,
                    gamma=1.05,
                    seed=seed,
                )
                train_linear_compressor(
                    compressor, teacher_labels, cfg, valid_vertices=lcc_tensor,
                )

                # Extract compressed embeddings
                d_out_t = teacher_labels.d_out.t()
                d_in_t = teacher_labels.d_in.t()
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

                # Run A* with AAC heuristic
                aac_exps = [
                    astar(graph, s, t, heuristic=aac_h).expansions
                    for s, t in queries
                ]
                aac_mean = np.mean(aac_exps)
                aac_reduction = (1.0 - aac_mean / dij_mean) * 100.0

                # Paired Wilcoxon signed-rank test (two-sided)
                diff = np.array(aac_exps) - np.array(alt_exps)
                # Handle case where all differences are zero
                if np.all(diff == 0):
                    stat, p_two = 0.0, 1.0
                else:
                    stat, p_two = wilcoxon(diff, alternative="two-sided")

                alt_wins = int(np.sum(np.array(alt_exps) < np.array(aac_exps)))

                result = {
                    "graph": graph_name,
                    "budget_bpv": budget_bpv,
                    "aac_K0": aac_K0,
                    "aac_m": aac_m,
                    "alt_K": alt_K,
                    "seed": seed,
                    "aac_mean_exp": f"{aac_mean:.1f}",
                    "alt_mean_exp": f"{alt_mean:.1f}",
                    "aac_reduction_pct": f"{aac_reduction:.2f}",
                    "alt_reduction_pct": f"{alt_reduction:.2f}",
                    "wilcoxon_stat": f"{stat:.1f}",
                    "p_value_twosided": f"{p_two:.6e}",
                    "alt_wins_count": alt_wins,
                }
                all_results.append(result)

                print(f"done ({prep_time:.0f}s)  "
                      f"aac={aac_mean:.0f} alt={alt_mean:.0f}  "
                      f"W={stat:.0f} p={p_two:.2e}  alt_wins={alt_wins}")

    # Write CSV
    if not all_results:
        print("\nNo results to write (no graph data found).")
        return

    csv_path = OUTPUT_DIR / "wilcoxon_matched_budget.csv"
    cols = [
        "graph", "budget_bpv", "aac_K0", "aac_m", "alt_K", "seed",
        "aac_mean_exp", "alt_mean_exp", "aac_reduction_pct", "alt_reduction_pct",
        "wilcoxon_stat", "p_value_twosided", "alt_wins_count",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    import pandas as pd
    df = pd.DataFrame(all_results)
    df["p_value_twosided"] = df["p_value_twosided"].astype(float)
    df["aac_mean_exp"] = df["aac_mean_exp"].astype(float)
    df["alt_mean_exp"] = df["alt_mean_exp"].astype(float)
    df["alt_wins_count"] = df["alt_wins_count"].astype(int)

    for budget_bpv in [32, 64, 128]:
        bdf = df[df["budget_bpv"] == budget_bpv]
        if bdf.empty:
            continue
        print(f"\n  Budget {budget_bpv} B/v:")
        for gname in bdf["graph"].unique():
            gdf = bdf[bdf["graph"] == gname]
            all_sig = all(gdf["p_value_twosided"] < 0.05)
            p_range = f"{gdf['p_value_twosided'].min():.2e} -- {gdf['p_value_twosided'].max():.2e}"
            aac_avg = gdf["aac_mean_exp"].mean()
            alt_avg = gdf["alt_mean_exp"].mean()
            wins = gdf["alt_wins_count"].values.tolist()
            sig_str = "ALL SIG (p<0.05)" if all_sig else "NOT ALL SIG"
            print(f"    {gname:4s}: {sig_str}  p=[{p_range}]  "
                  f"aac_mean={aac_avg:.0f} alt_mean={alt_avg:.0f}  "
                  f"alt_wins={wins}")

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
