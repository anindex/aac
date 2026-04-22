#!/usr/bin/env python
"""Multi-seed Pareto sweep for AAC vs ALT vs FastMap on DIMACS road graphs.

Runs the Pareto sweep with multiple seeds and aggregates results with
means and standard deviations.

Output: results/dimacs/pareto_sweep_{GRAPH}_multiseed.csv
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.baselines.fastmap import fastmap_preprocess, make_fastmap_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.loaders.dimacs import load_dimacs
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRAPHS = {
    "NY": ("USA-road-d.NY.gr", "USA-road-d.NY.co"),
    "FLA": ("USA-road-d.FLA.gr", "USA-road-d.FLA.co"),
}
DATA_DIR = Path("data/dimacs")
OUTPUT_DIR = Path("results/dimacs")

SEEDS = [42, 123, 456, 789, 1024]
SWEEP_K0 = [32, 64, 128]
SWEEP_M = [8, 16, 32, 64]
ALT_K_VALUES = [4, 8, 16, 32]
FASTMAP_D_VALUES = [8, 16, 32, 64]

NUM_EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3
COND_LAMBDA = 0.01
T_INIT = 1.0
GAMMA = 1.05
NUM_QUERIES = 50


def run_queries(graph, queries, heuristic):
    expansions = []
    for s, t in queries:
        result = astar(graph, s, t, heuristic=heuristic)
        expansions.append(result.expansions)
    return expansions


def run_queries_with_admissibility(graph, queries, heuristic):
    """Run queries and return (expansions, costs) for admissibility checking."""
    expansions, costs = [], []
    for s, t in queries:
        result = astar(graph, s, t, heuristic=heuristic)
        expansions.append(result.expansions)
        costs.append(result.cost)
    return expansions, costs


def run_aac_config(graph, K0, m, queries, lcc_nodes, lcc_seed, seed):
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    anchors = farthest_point_sampling(graph, K0, seed_vertex=lcc_seed, valid_vertices=lcc_tensor)
    teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)
    compressor = LinearCompressor(K=K0, m=m, is_directed=graph.is_directed)
    cfg = TrainConfig(
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        cond_lambda=COND_LAMBDA, T_init=T_INIT, gamma=GAMMA, seed=seed,
    )
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
    heuristic = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)
    return heuristic, time.perf_counter() - t0


def run_single_seed(graph_name, graph, queries, lcc_nodes, lcc_seed, dij_mean, seed):
    """Run full sweep for one seed, return list of result dicts."""
    results = []
    # AAC sweep
    for K0 in SWEEP_K0:
        for m in SWEEP_M:
            if m > K0:
                continue
            try:
                h, prep = run_aac_config(graph, K0, m, queries, lcc_nodes, lcc_seed, seed)
                exps = run_queries(graph, queries, h)
                mean_exp = np.mean(exps)
                reduction = 100.0 * (1.0 - mean_exp / dij_mean) if dij_mean > 0 else 0.0
                results.append({
                    "method": "AAC", "K0": K0, "m": m,
                    "num_landmarks_or_dims": m, "bytes_per_vertex": m * 4,
                    "mean_expansions": mean_exp, "median_expansions": np.median(exps),
                    "expansion_reduction_pct": reduction, "preprocess_sec": prep,
                    "seed": seed,
                })
                print(f"    [seed={seed}] AAC K0={K0} m={m}: {reduction:.1f}%")
            except Exception as e:
                print(f"    [seed={seed}] AAC K0={K0} m={m}: FAILED ({e})")

    # ALT sweep
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    for K in ALT_K_VALUES:
        try:
            t0 = time.perf_counter()
            teacher = alt_preprocess(
                graph, K, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
            )
            h = make_alt_heuristic(teacher)
            prep = time.perf_counter() - t0
            exps = run_queries(graph, queries, h)
            mean_exp = np.mean(exps)
            reduction = 100.0 * (1.0 - mean_exp / dij_mean) if dij_mean > 0 else 0.0
            results.append({
                "method": "ALT", "K0": 0, "m": 0,
                "num_landmarks_or_dims": K, "bytes_per_vertex": 2 * K * 4,
                "mean_expansions": mean_exp, "median_expansions": np.median(exps),
                "expansion_reduction_pct": reduction, "preprocess_sec": prep,
                "seed": seed,
            })
            print(f"    [seed={seed}] ALT K={K}: {reduction:.1f}%")
        except Exception as e:
            print(f"    [seed={seed}] ALT K={K}: FAILED ({e})")

    # FastMap sweep
    for d in FASTMAP_D_VALUES:
        try:
            t0 = time.perf_counter()
            coords = fastmap_preprocess(graph, d)
            h = make_fastmap_heuristic(coords)
            prep = time.perf_counter() - t0
            exps, costs = run_queries_with_admissibility(graph, queries, h)
            # Check admissibility
            dij_costs = [dijkstra(graph, s, t).cost for s, t in queries]
            n_violations = sum(1 for c, dc in zip(costs, dij_costs) if c - dc > 1e-6)
            mean_exp = np.mean(exps)
            reduction = 100.0 * (1.0 - mean_exp / dij_mean) if dij_mean > 0 else 0.0
            results.append({
                "method": "FastMap", "K0": 0, "m": 0,
                "num_landmarks_or_dims": d, "bytes_per_vertex": d * 4,
                "mean_expansions": mean_exp, "median_expansions": np.median(exps),
                "expansion_reduction_pct": reduction, "preprocess_sec": prep,
                "seed": seed, "admissibility_violations": n_violations,
                "violation_pct": 100.0 * n_violations / len(queries),
            })
            print(f"    [seed={seed}] FastMap d={d}: {reduction:.1f}%, violations={n_violations}/{len(queries)}")
        except Exception as e:
            print(f"    [seed={seed}] FastMap d={d}: FAILED ({e})")

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for graph_name, (gr_file, co_file) in GRAPHS.items():
        gr_path = DATA_DIR / gr_file
        co_path = DATA_DIR / co_file
        if not gr_path.exists():
            print(f"WARNING: {gr_path} not found, skipping {graph_name}")
            continue

        print(f"\n{'='*60}")
        print(f"  Multi-seed Pareto sweep: {graph_name} ({len(SEEDS)} seeds)")
        print(f"{'='*60}")

        graph = load_dimacs(str(gr_path), str(co_path) if co_path.exists() else None)
        print(f"  Loaded: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")
        lcc_nodes, lcc_seed = compute_strong_lcc(graph)

        # Use seed 42 for query generation (same queries across seeds)
        queries = generate_queries(graph, NUM_QUERIES, seed=42)

        # Dijkstra baseline (deterministic, run once)
        print("  Running Dijkstra baseline...")
        dij_exps = [dijkstra(graph, s, t).expansions for s, t in queries]
        dij_mean = np.mean(dij_exps)
        print(f"  Dijkstra: mean={dij_mean:.0f}")

        all_results = []
        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            seed_results = run_single_seed(
                graph_name, graph, queries, lcc_nodes, lcc_seed, dij_mean, seed
            )
            all_results.extend(seed_results)

        # Write per-seed results
        per_seed_path = OUTPUT_DIR / f"pareto_sweep_{graph_name}_perseed.csv"
        with open(per_seed_path, "w", newline="") as f:
            cols = ["method", "K0", "m", "num_landmarks_or_dims", "bytes_per_vertex",
                    "mean_expansions", "median_expansions", "expansion_reduction_pct",
                    "preprocess_sec", "seed", "admissibility_violations", "violation_pct"]
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

        # Aggregate: compute mean and std across seeds
        import pandas as pd
        df = pd.DataFrame(all_results)
        group_cols = ["method", "K0", "m", "num_landmarks_or_dims", "bytes_per_vertex"]
        agg = df.groupby(group_cols, as_index=False).agg(
            mean_expansions_mean=("mean_expansions", "mean"),
            mean_expansions_std=("mean_expansions", "std"),
            expansion_reduction_mean=("expansion_reduction_pct", "mean"),
            expansion_reduction_std=("expansion_reduction_pct", "std"),
            preprocess_sec_mean=("preprocess_sec", "mean"),
            num_seeds=("seed", "count"),
        )

        agg_path = OUTPUT_DIR / f"pareto_sweep_{graph_name}_multiseed.csv"
        agg.to_csv(agg_path, index=False, float_format="%.2f")
        print(f"\n  Results written to {per_seed_path} and {agg_path}")

        # Also write the old-format single-seed CSV using mean values for compatibility
        compat_path = OUTPUT_DIR / f"pareto_sweep_{graph_name}.csv"
        with open(compat_path, "w", newline="") as f:
            from experiments.reporting.csv_writer import write_csv_metadata
            write_csv_metadata(f)
            cols_compat = ["method", "K0", "m", "num_landmarks_or_dims", "bytes_per_vertex",
                           "mean_expansions", "median_expansions", "expansion_reduction_pct",
                           "preprocess_sec", "num_queries"]
            writer = csv.DictWriter(f, fieldnames=cols_compat)
            writer.writeheader()
            # Dijkstra row
            writer.writerow({
                "method": "Dijkstra", "K0": 0, "m": 0, "num_landmarks_or_dims": 0,
                "bytes_per_vertex": 0, "mean_expansions": f"{dij_mean:.1f}",
                "median_expansions": f"{np.median(dij_exps):.1f}",
                "expansion_reduction_pct": "0.00", "preprocess_sec": "0.0",
                "num_queries": NUM_QUERIES,
            })
            for _, row in agg.iterrows():
                writer.writerow({
                    "method": row["method"], "K0": int(row["K0"]), "m": int(row["m"]),
                    "num_landmarks_or_dims": int(row["num_landmarks_or_dims"]),
                    "bytes_per_vertex": int(row["bytes_per_vertex"]),
                    "mean_expansions": f"{row['mean_expansions_mean']:.1f}",
                    "median_expansions": "",
                    "expansion_reduction_pct": f"{row['expansion_reduction_mean']:.2f}",
                    "preprocess_sec": f"{row['preprocess_sec_mean']:.1f}",
                    "num_queries": NUM_QUERIES,
                })

    print("\nDone!")


if __name__ == "__main__":
    main()
