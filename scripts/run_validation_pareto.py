#!/usr/bin/env python
"""Validation-split Pareto sweep: select K0 on validation queries, evaluate on test queries.

Addresses reviewer concern about retrospective K0 selection (cherry-picking).
Generates 200 queries: 100 for validation (K0 selection), 100 for test (reporting).

Output: results/dimacs/pareto_valsplit_{GRAPH}.csv
        results/osmnx/pareto_valsplit_{GRAPH}.csv
"""

from __future__ import annotations

import csv
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
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DIMACS_GRAPHS = {
    "NY": ("USA-road-d.NY.gr", "USA-road-d.NY.co"),
    "FLA": ("USA-road-d.FLA.gr", "USA-road-d.FLA.co"),
}
OSMNX_GRAPHS = {
    "modena": "data/osmnx/modena.npz",
    "manhattan": "data/osmnx/manhattan.npz",
}
DIMACS_DIR = Path("data/dimacs")
DIMACS_OUT = Path("results/dimacs")
OSMNX_OUT = Path("results/osmnx")

SEEDS = [42, 123, 456, 789, 1024]
SWEEP_K0 = [32, 64, 128]
SWEEP_M = [8, 16, 32, 64]
ALT_K_VALUES = [4, 8, 16, 32]

NUM_VAL_QUERIES = 100
NUM_TEST_QUERIES = 100
NUM_EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3

# Memory budget groups: bytes_per_vertex -> desired budgets to report
BUDGET_GROUPS = [32, 64, 128]


def generate_val_test_queries(graph, num_val, num_test, seed=42):
    """Generate disjoint validation and test query sets."""
    total = num_val + num_test
    all_queries = generate_queries(graph, total, seed=seed)
    return all_queries[:num_val], all_queries[num_val:]


def eval_queries(graph, queries, heuristic):
    """Run A* on queries, return per-query expansions."""
    return [astar(graph, s, t, heuristic=heuristic).expansions for s, t in queries]


def run_aac_config(graph, K0, m, lcc_nodes, lcc_seed, seed):
    """Train AAC compressor and return heuristic + preprocessing time."""
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    anchors = farthest_point_sampling(graph, K0, seed_vertex=lcc_seed, valid_vertices=lcc_tensor)
    teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)
    compressor = LinearCompressor(K=K0, m=m, is_directed=graph.is_directed)
    cfg = TrainConfig(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=seed)
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


def run_graph(graph_name, graph, output_dir):
    """Run validation-split experiment for one graph."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)

    # Generate disjoint val/test queries
    val_queries, test_queries = generate_val_test_queries(
        graph, NUM_VAL_QUERIES, NUM_TEST_QUERIES, seed=42
    )
    print(f"  Queries: {len(val_queries)} val, {len(test_queries)} test")

    # Dijkstra baselines
    dij_val_exps = [dijkstra(graph, s, t).expansions for s, t in val_queries]
    dij_test_exps = [dijkstra(graph, s, t).expansions for s, t in test_queries]
    dij_val_mean = np.mean(dij_val_exps)
    dij_test_mean = np.mean(dij_test_exps)
    print(f"  Dijkstra val mean={dij_val_mean:.0f}, test mean={dij_test_mean:.0f}")

    # Collect all AAC results: {seed -> {(K0, m) -> (heuristic, val_reduction, test_reduction)}}
    all_results = []

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")

        # AAC sweep: train all K0/m combos, evaluate on both val and test
        aac_configs = {}  # (K0, m) -> (val_reduction, test_reduction, prep_time)
        for K0 in SWEEP_K0:
            for m in SWEEP_M:
                if m > K0:
                    continue
                try:
                    h, prep = run_aac_config(graph, K0, m, lcc_nodes, lcc_seed, seed)
                    val_exps = eval_queries(graph, val_queries, h)
                    test_exps = eval_queries(graph, test_queries, h)
                    val_red = 100.0 * (1.0 - np.mean(val_exps) / dij_val_mean)
                    test_red = 100.0 * (1.0 - np.mean(test_exps) / dij_test_mean)
                    bpv = m * 4
                    aac_configs[(K0, m)] = {
                        "val_reduction": val_red,
                        "test_reduction": test_red,
                        "val_mean_exp": np.mean(val_exps),
                        "test_mean_exp": np.mean(test_exps),
                        "preprocess_sec": prep,
                        "bytes_per_vertex": bpv,
                    }
                    print(f"    AAC K0={K0} m={m}: val={val_red:.1f}% test={test_red:.1f}%")
                except Exception as e:
                    print(f"    AAC K0={K0} m={m}: FAILED ({e})")

        # ALT sweep
        alt_configs = {}
        for K in ALT_K_VALUES:
            try:
                t0 = time.perf_counter()
                teacher = alt_preprocess(graph, K, seed_vertex=lcc_seed, valid_vertices=lcc_tensor)
                h = make_alt_heuristic(teacher)
                prep = time.perf_counter() - t0
                val_exps = eval_queries(graph, val_queries, h)
                test_exps = eval_queries(graph, test_queries, h)
                val_red = 100.0 * (1.0 - np.mean(val_exps) / dij_val_mean)
                test_red = 100.0 * (1.0 - np.mean(test_exps) / dij_test_mean)
                bpv = 2 * K * 4
                alt_configs[K] = {
                    "val_reduction": val_red,
                    "test_reduction": test_red,
                    "val_mean_exp": np.mean(val_exps),
                    "test_mean_exp": np.mean(test_exps),
                    "preprocess_sec": prep,
                    "bytes_per_vertex": bpv,
                }
                print(f"    ALT K={K}: val={val_red:.1f}% test={test_red:.1f}%")
            except Exception as e:
                print(f"    ALT K={K}: FAILED ({e})")

        # Validation-based K0 selection: for each budget, pick the K0
        # that gives best val_reduction
        for budget_bpv in BUDGET_GROUPS:
            # Find best AAC config for this budget
            candidates = {
                (K0, m): info for (K0, m), info in aac_configs.items()
                if info["bytes_per_vertex"] == budget_bpv
            }
            if not candidates:
                continue

            best_key = max(candidates, key=lambda k: candidates[k]["val_reduction"])
            best_info = candidates[best_key]
            K0_sel, m_sel = best_key

            # Record result
            all_results.append({
                "graph": graph_name,
                "seed": seed,
                "method": "AAC",
                "K0": K0_sel,
                "m": m_sel,
                "bytes_per_vertex": budget_bpv,
                "val_reduction_pct": best_info["val_reduction"],
                "test_reduction_pct": best_info["test_reduction"],
                "val_mean_exp": best_info["val_mean_exp"],
                "test_mean_exp": best_info["test_mean_exp"],
                "preprocess_sec": best_info["preprocess_sec"],
                "selection": "val-selected",
            })
            print(f"    Val-selected AAC @ {budget_bpv}B/v: K0={K0_sel} "
                  f"(val={best_info['val_reduction']:.1f}% -> test={best_info['test_reduction']:.1f}%)")

        # Also record ALT results (deterministic, no selection needed)
        for K, info in alt_configs.items():
            all_results.append({
                "graph": graph_name,
                "seed": seed,
                "method": "ALT",
                "K0": 0,
                "m": 0,
                "bytes_per_vertex": info["bytes_per_vertex"],
                "val_reduction_pct": info["val_reduction"],
                "test_reduction_pct": info["test_reduction"],
                "val_mean_exp": info["val_mean_exp"],
                "test_mean_exp": info["test_mean_exp"],
                "preprocess_sec": info["preprocess_sec"],
                "selection": "N/A",
            })

    # Write results
    csv_path = output_dir / f"pareto_valsplit_{graph_name}.csv"
    cols = ["graph", "seed", "method", "K0", "m", "bytes_per_vertex",
            "val_reduction_pct", "test_reduction_pct", "val_mean_exp",
            "test_mean_exp", "preprocess_sec", "selection"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Written: {csv_path}")

    # Summary: mean/std across seeds for val-selected configs
    import pandas as pd
    df = pd.DataFrame(all_results)
    for bpv in BUDGET_GROUPS:
        aac_rows = df[(df["method"] == "AAC") & (df["bytes_per_vertex"] == bpv)]
        alt_rows = df[(df["method"] == "ALT") & (df["bytes_per_vertex"] == bpv)]
        if not aac_rows.empty:
            print(f"\n  {graph_name} @ {bpv}B/v AAC (val-selected): "
                  f"test={aac_rows['test_reduction_pct'].mean():.1f}+/-{aac_rows['test_reduction_pct'].std():.1f}%")
        if not alt_rows.empty:
            print(f"  {graph_name} @ {bpv}B/v ALT: "
                  f"test={alt_rows['test_reduction_pct'].mean():.1f}+/-{alt_rows['test_reduction_pct'].std():.1f}%")


def main():
    # DIMACS graphs
    for name, (gr_file, co_file) in DIMACS_GRAPHS.items():
        gr_path = DIMACS_DIR / gr_file
        co_path = DIMACS_DIR / co_file
        if not gr_path.exists():
            print(f"Skipping {name}: {gr_path} not found")
            continue
        print(f"\n{'='*60}")
        print(f"  Validation-split Pareto: {name}")
        print(f"{'='*60}")
        graph = load_dimacs(str(gr_path), str(co_path) if co_path.exists() else None)
        print(f"  Loaded: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")
        run_graph(name, graph, DIMACS_OUT)

    # OSMnx graphs
    for name, npz_path in OSMNX_GRAPHS.items():
        npz = Path(npz_path)
        if not npz.exists():
            print(f"Skipping {name}: {npz} not found")
            continue
        print(f"\n{'='*60}")
        print(f"  Validation-split Pareto: {name}")
        print(f"{'='*60}")
        graph = load_graph_npz(npz)
        print(f"  Loaded: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")
        run_graph(name, graph, OSMNX_OUT)

    print("\nDone!")


if __name__ == "__main__":
    main()
