#!/usr/bin/env python
"""Multi-seed Wilcoxon signed-rank test: AAC vs ALT on all 4 DIMACS graphs.

Runs AAC (K0=64, m=16) vs ALT (K=16) with 3 training seeds on 100 queries.
Reports per-seed and aggregated Wilcoxon statistics.

Output: results/dimacs/wilcoxon_pvalues_multiseed.csv
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import scipy.stats
import torch

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.loaders.dimacs import load_dimacs
from aac.search.astar import astar
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

DATA_DIR = Path("data/dimacs")
OUTPUT_DIR = Path("results/dimacs")

GRAPHS = {
    "NY": ("USA-road-d.NY.gr", "USA-road-d.NY.co"),
    "BAY": ("USA-road-d.BAY.gr", "USA-road-d.BAY.co"),
    "COL": ("USA-road-d.COL.gr", "USA-road-d.COL.co"),
    "FLA": ("USA-road-d.FLA.gr", "USA-road-d.FLA.co"),
}

K0 = 64
M = 16
ALT_K = 16
NUM_QUERIES = 100
SEEDS = [42, 123, 456, 789, 1024]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for graph_name, (gr_file, co_file) in GRAPHS.items():
        gr_path = DATA_DIR / gr_file
        co_path = DATA_DIR / co_file
        if not gr_path.exists():
            print(f"SKIP: {gr_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Wilcoxon: {graph_name} (K0={K0} m={M} vs ALT K={ALT_K})")
        print(f"{'='*60}")

        graph = load_dimacs(str(gr_path), str(co_path) if co_path.exists() else None)
        print(f"  {graph.num_nodes:,} nodes")

        lcc_nodes, lcc_seed = compute_strong_lcc(graph)
        lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)

        # Same queries for all seeds
        queries = generate_queries(graph, NUM_QUERIES, seed=42)

        # ALT baseline (deterministic, run once)
        print("  ALT preprocessing...")
        teacher = alt_preprocess(
            graph, ALT_K, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
        )
        alt_h = make_alt_heuristic(teacher)
        alt_exps = [astar(graph, s, t, heuristic=alt_h).expansions for s, t in queries]

        for seed in SEEDS:
            print(f"  Seed {seed}: AAC preprocessing...", end=" ", flush=True)
            torch.manual_seed(seed)
            t0 = time.perf_counter()

            anchors = farthest_point_sampling(graph, K0, seed_vertex=lcc_seed, valid_vertices=lcc_tensor)
            teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)
            compressor = LinearCompressor(K=K0, m=M, is_directed=graph.is_directed)
            cfg = TrainConfig(num_epochs=200, batch_size=256, lr=1e-3, seed=seed)
            train_linear_compressor(compressor, teacher_labels, cfg, valid_vertices=lcc_tensor)

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
            prep = time.perf_counter() - t0

            aac_exps = [astar(graph, s, t, heuristic=aac_h).expansions for s, t in queries]

            # Wilcoxon signed-rank test
            diff = np.array(aac_exps) - np.array(alt_exps)
            stat, p_two = scipy.stats.wilcoxon(diff, alternative="two-sided")
            _, p_less = scipy.stats.wilcoxon(diff, alternative="less")
            aac_wins = int(np.sum(np.array(aac_exps) < np.array(alt_exps)))

            all_results.append({
                "graph": graph_name, "seed": seed,
                "aac_mean": np.mean(aac_exps), "alt_mean": np.mean(alt_exps),
                "wilcoxon_stat": stat, "p_twosided": p_two, "p_less": p_less,
                "aac_wins": aac_wins, "n_queries": NUM_QUERIES,
            })
            print(f"done ({prep:.0f}s) p_two={p_two:.2e} aac_wins={aac_wins}/{NUM_QUERIES}")

    # Write results
    csv_path = OUTPUT_DIR / "wilcoxon_pvalues_multiseed.csv"
    cols = ["graph", "seed", "aac_mean", "alt_mean", "wilcoxon_stat", "p_twosided", "p_less", "aac_wins", "n_queries"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_results)

    # Summary
    import pandas as pd
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print("  Summary: all seeds significant?")
    for graph_name in df["graph"].unique():
        gdf = df[df["graph"] == graph_name]
        all_sig = all(gdf["p_twosided"] < 1e-5)
        p_range = f"{gdf['p_twosided'].min():.2e} - {gdf['p_twosided'].max():.2e}"
        wins_range = f"{gdf['aac_wins'].min()}-{gdf['aac_wins'].max()}"
        print(f"  {graph_name}: {'ALL SIG' if all_sig else 'NOT ALL SIG'} p∈[{p_range}] aac_wins∈[{wins_range}]")

    print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    main()
