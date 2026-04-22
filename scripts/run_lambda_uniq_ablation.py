#!/usr/bin/env python
"""Ablation over lambda_uniq (uniqueness regularizer weight) on OSMnx Modena.

We verify whether the uniqueness penalty
R_uniq is load-bearing. We sweep lambda_uniq in {0, 0.01, 0.1} at the
canonical Modena configuration (K0=64, m=16, 5 seeds, 100 uniform queries)
and report:
  - effective_unique_ratio (fraction of selected landmarks that are distinct)
  - expansion reduction vs Dijkstra
  - admissibility violations (should be zero regardless of lambda)

Output:
    results/lambda_uniq_ablation/modena_results.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.io import load_graph_npz
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

GRAPH_PATH = _PROJECT_ROOT / "data" / "osmnx" / "modena.npz"
OUTPUT_DIR = _PROJECT_ROOT / "results" / "lambda_uniq_ablation"

K0 = 64
M = 16
SEEDS = [42, 123, 456, 789, 1024]
NUM_QUERIES = 100
QUERY_SEED = 42
LAMBDAS = [0.0, 0.01, 0.1]


def run_one(graph, lcc_tensor, lcc_seed, queries, dij_mean: float,
            seed: int, uniq_lambda: float) -> dict:
    torch.manual_seed(seed)
    t0 = time.perf_counter()

    rng = torch.Generator().manual_seed(seed)
    anchors = farthest_point_sampling(
        graph, K0, seed_vertex=lcc_seed, rng=rng, valid_vertices=lcc_tensor,
    )
    teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)

    compressor = LinearCompressor(K=K0, m=M, is_directed=graph.is_directed)
    cfg = TrainConfig(
        num_epochs=200, batch_size=256, lr=1e-3, cond_lambda=0.01,
        uniq_lambda=uniq_lambda, T_init=1.0, gamma=1.05,
        seed=seed, patience=20,
    )
    train_linear_compressor(compressor, teacher_labels, cfg, valid_vertices=lcc_tensor)

    compressor.eval()
    d_out_t = teacher_labels.d_out.t()
    d_in_t = teacher_labels.d_in.t()
    with torch.no_grad():
        if graph.is_directed:
            y_fwd, y_bwd = compressor(d_out_t, d_in_t)
            y_fwd, y_bwd = y_fwd.detach(), y_bwd.detach()
        else:
            y = compressor(d_out_t)
            y_fwd = y_bwd = y.detach()
    heuristic = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)
    prep_time = time.perf_counter() - t0

    stats = compressor.selection_stats()

    # A* evaluation
    exps = []
    all_optimal = True
    admis_violations = 0
    for s, t in queries:
        r = astar(graph, s, t, heuristic=heuristic)
        exps.append(r.expansions)
        if not r.optimal:
            all_optimal = False
        r_dij = dijkstra(graph, s, t)
        h_val = heuristic(s, t)
        if h_val > r_dij.cost + 1e-6:
            admis_violations += 1

    mean_exp = float(np.mean(exps))
    reduction = 100.0 * (1.0 - mean_exp / dij_mean) if dij_mean > 0 else 0.0

    print(
        f"    lambda={uniq_lambda:.3f} seed={seed}: "
        f"mean_exp={mean_exp:.0f} ({reduction:.1f}% red.) "
        f"unique_ratio={stats['effective_unique_ratio']:.3f} "
        f"(fwd={stats['unique_fwd']}/{stats['nominal_fwd']}, "
        f"bwd={stats['unique_bwd']}/{stats['nominal_bwd']}) "
        f"adm_viol={admis_violations} opt={all_optimal} [{prep_time:.1f}s]"
    )

    return {
        "graph": "modena",
        "K0": K0,
        "m": M,
        "uniq_lambda": uniq_lambda,
        "seed": seed,
        "mean_exp": f"{mean_exp:.2f}",
        "reduction_pct": f"{reduction:.2f}",
        "unique_fwd": stats["unique_fwd"],
        "nominal_fwd": stats["nominal_fwd"],
        "unique_bwd": stats["unique_bwd"],
        "nominal_bwd": stats["nominal_bwd"],
        "effective_unique_ratio": f"{stats['effective_unique_ratio']:.4f}",
        "admissibility_violations": admis_violations,
        "all_optimal": all_optimal,
        "prep_time_s": f"{prep_time:.2f}",
    }


CSV_COLUMNS = [
    "graph", "K0", "m", "uniq_lambda", "seed",
    "mean_exp", "reduction_pct",
    "unique_fwd", "nominal_fwd", "unique_bwd", "nominal_bwd",
    "effective_unique_ratio",
    "admissibility_violations", "all_optimal", "prep_time_s",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Modena from {GRAPH_PATH}...")
    graph = load_graph_npz(GRAPH_PATH)
    print(f"  {graph.num_nodes:,} nodes, {graph.num_edges:,} edges (directed={graph.is_directed})")

    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    print(f"  LCC: {len(lcc_nodes):,} nodes, seed={lcc_seed}")

    queries = generate_queries(graph, NUM_QUERIES, seed=QUERY_SEED)

    # Baseline Dijkstra (deterministic)
    print("Dijkstra baseline...")
    dij_exps = np.array([dijkstra(graph, s, t).expansions for s, t in queries])
    dij_mean = float(np.mean(dij_exps))
    print(f"  Dijkstra mean expansions: {dij_mean:.0f}")

    rows: list[dict] = []
    for lam in LAMBDAS:
        print(f"\n--- lambda_uniq = {lam} ---")
        for seed in SEEDS:
            rows.append(run_one(graph, lcc_tensor, lcc_seed, queries, dij_mean, seed, lam))

    out_path = OUTPUT_DIR / "modena_results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
