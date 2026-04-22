#!/usr/bin/env python
"""Evaluate AAC+ALT hybrid heuristic on OSMnx and DIMACS graphs.

Tests the hypothesis that max(AAC, ALT_K=4) at combined memory budget
outperforms AAC alone at equal memory. Since both AAC and ALT produce
admissible heuristics, their pointwise max is also admissible.

Output: results/hybrid/hybrid_evaluation.csv
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.io import load_graph_npz
from aac.graphs.loaders.dimacs import load_dimacs
from aac.heuristics import make_hybrid_heuristic
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.metrics.admissibility import check_admissibility
from experiments.utils import compute_strong_lcc, generate_queries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
OUTPUT_DIR = Path("results/hybrid")
SEEDS = [42, 123, 456, 789, 1024]
NUM_QUERIES = 100
NUM_EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3

# Test graphs (representative)
TEST_GRAPHS = {
    "modena": {"path": Path("data/osmnx/modena.npz"), "loader": "npz"},
    "manhattan": {"path": Path("data/osmnx/manhattan.npz"), "loader": "npz"},
    "NY": {
        "path": (Path("data/dimacs/USA-road-d.NY.gr"), Path("data/dimacs/USA-road-d.NY.co")),
        "loader": "dimacs",
    },
}

# AAC configs to test in hybrid
# Format: (K0, m, alt_K)
# alt_K=0 means AAC-only baseline; alt_K>0 means hybrid
CONFIGS = [
    # Baseline AAC-only
    (16, 8, 0),    # 32 B/v
    (32, 16, 0),   # 64 B/v
    (64, 32, 0),   # 128 B/v
    # Hybrid: AAC + ALT_K=4 (adds 32 B/v from ALT)
    (16, 8, 4),    # 32 + 32 = 64 B/v total
    (32, 16, 4),   # 64 + 32 = 96 B/v total
]

# Pure-ALT baselines at each budget for matched-budget comparison.
# Bytes per vertex = 2*K*4 for directed graphs.
ALT_ONLY_K_VALUES = [4, 8, 12, 16, 32]
#  K=4 -> 32 B/v, K=8 -> 64 B/v, K=12 -> 96 B/v, K=16 -> 128 B/v, K=32 -> 256 B/v


def load_graph(name, info):
    if info["loader"] == "npz":
        return load_graph_npz(info["path"])
    else:
        gr, co = info["path"]
        return load_dimacs(gr, co)


def run_experiment(graph_name, graph, seed):
    """Run all configs on one graph with one seed."""
    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    queries = generate_queries(graph, NUM_QUERIES, seed=42)

    # Dijkstra baseline
    dij_results = [dijkstra(graph, s, t) for s, t in queries]
    dij_costs = [r.cost for r in dij_results]
    dij_mean_exp = np.mean([r.expansions for r in dij_results])

    results = []
    torch.manual_seed(seed)

    for K0, m, alt_K in CONFIGS:
        try:
            t0 = time.perf_counter()

            # AAC preprocessing
            anchors = farthest_point_sampling(
                graph, K0, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
            )
            teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)
            compressor = LinearCompressor(K=K0, m=m, is_directed=graph.is_directed)
            train_cfg = TrainConfig(
                num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=seed,
            )
            train_linear_compressor(
                compressor, teacher_labels, train_cfg, valid_vertices=lcc_tensor,
            )
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
            h_aac = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)

            if alt_K > 0:
                # ALT preprocessing for hybrid
                alt_teacher = alt_preprocess(
                    graph, alt_K, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
                )
                h_alt = make_alt_heuristic(alt_teacher)
                heuristic = make_hybrid_heuristic(h_aac, h_alt)
                method_name = f"AAC+ALT"
                bytes_per_v = m * 4 + 2 * alt_K * 4  # AAC + ALT memory
            else:
                heuristic = h_aac
                method_name = "AAC"
                bytes_per_v = m * 4

            prep_sec = time.perf_counter() - t0

            # Run queries
            astar_results = []
            latencies = []
            for s, t in queries:
                qt0 = time.perf_counter()
                r = astar(graph, s, t, heuristic=heuristic)
                latencies.append((time.perf_counter() - qt0) * 1000)
                astar_results.append(r)

            exps = [r.expansions for r in astar_results]
            mean_exp = np.mean(exps)
            reduction = 100.0 * (1.0 - mean_exp / dij_mean_exp) if dij_mean_exp > 0 else 0.0

            # Admissibility check
            adm = check_admissibility(astar_results, dij_costs)

            results.append({
                "graph": graph_name,
                "method": method_name,
                "K0": K0, "m": m, "alt_K": alt_K,
                "bytes_per_vertex": bytes_per_v,
                "mean_expansions": mean_exp,
                "expansion_reduction_pct": reduction,
                "preprocess_sec": prep_sec,
                "mean_query_latency_ms": np.mean(latencies),
                "admissibility_violations": adm.num_violations,
                "seed": seed,
            })
            logger.info(
                "  %s K0=%d m=%d alt_K=%d: %.1f%% reduction, %d B/v, adm=%s",
                method_name, K0, m, alt_K, reduction, bytes_per_v,
                "OK" if adm.num_violations == 0 else f"FAIL({adm.num_violations})",
            )

        except Exception as e:
            logger.error("  %s K0=%d m=%d alt_K=%d FAILED: %s", method_name, K0, m, alt_K, e)

    # Pure-ALT baselines at all budget points for matched-budget comparison
    for K in ALT_ONLY_K_VALUES:
        try:
            t0 = time.perf_counter()
            alt_teacher = alt_preprocess(
                graph, K, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
            )
            h_alt = make_alt_heuristic(alt_teacher)
            prep_sec = time.perf_counter() - t0

            astar_results = []
            latencies = []
            for s, t in queries:
                qt0 = time.perf_counter()
                r = astar(graph, s, t, heuristic=h_alt)
                latencies.append((time.perf_counter() - qt0) * 1000)
                astar_results.append(r)

            exps = [r.expansions for r in astar_results]
            mean_exp = np.mean(exps)
            reduction = 100.0 * (1.0 - mean_exp / dij_mean_exp) if dij_mean_exp > 0 else 0.0
            adm = check_admissibility(astar_results, dij_costs)
            bytes_per_v = 2 * K * 4

            results.append({
                "graph": graph_name,
                "method": "ALT",
                "K0": 0, "m": 0, "alt_K": K,
                "bytes_per_vertex": bytes_per_v,
                "mean_expansions": mean_exp,
                "expansion_reduction_pct": reduction,
                "preprocess_sec": prep_sec,
                "mean_query_latency_ms": np.mean(latencies),
                "admissibility_violations": adm.num_violations,
                "seed": seed,
            })
            logger.info(
                "  ALT K=%d: %.1f%% reduction, %d B/v, adm=%s",
                K, reduction, bytes_per_v,
                "OK" if adm.num_violations == 0 else f"FAIL({adm.num_violations})",
            )
        except Exception as e:
            logger.error("  ALT K=%d FAILED: %s", K, e)

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for graph_name, info in TEST_GRAPHS.items():
        if info["loader"] == "npz" and not info["path"].exists():
            logger.warning("Skipping %s: file not found", graph_name)
            continue
        if info["loader"] == "dimacs" and not info["path"][0].exists():
            logger.warning("Skipping %s: file not found", graph_name)
            continue

        logger.info("=== %s ===", graph_name)
        graph = load_graph(graph_name, info)
        logger.info("Loaded: %d nodes, %d edges, directed=%s",
                     graph.num_nodes, graph.num_edges, graph.is_directed)

        for seed in SEEDS:
            logger.info("--- seed=%d ---", seed)
            results = run_experiment(graph_name, graph, seed)
            all_results.extend(results)

    if all_results:
        csv_path = OUTPUT_DIR / "hybrid_evaluation.csv"
        cols = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(all_results)
        logger.info("Results: %s", csv_path)

        # Summary
        import pandas as pd
        df = pd.DataFrame(all_results)
        group_cols = ["graph", "method", "K0", "m", "alt_K", "bytes_per_vertex"]
        agg = df.groupby(group_cols, as_index=False).agg(
            reduction_mean=("expansion_reduction_pct", "mean"),
            reduction_std=("expansion_reduction_pct", "std"),
        )
        print("\n" + "="*70)
        print("  Hybrid Evaluation Summary (mean +/- std across seeds):")
        for g in agg["graph"].unique():
            print(f"\n  {g}:")
            g_data = agg[agg["graph"] == g].sort_values("bytes_per_vertex")
            for _, row in g_data.iterrows():
                std = f"+/-{row['reduction_std']:.1f}" if row["reduction_std"] > 0 else ""
                print(f"    {row['method']:12s} K0={int(row['K0']):3d} m={int(row['m']):3d} "
                      f"alt_K={int(row['alt_K']):2d}  {int(row['bytes_per_vertex']):4d}B/v: "
                      f"{row['reduction_mean']:.1f}%{std}")
        print("="*70)


if __name__ == "__main__":
    main()
