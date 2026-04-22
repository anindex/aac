#!/usr/bin/env python
"""Random-restart FPS baseline for the selection-strategy ablation.

The selection ablation in Section 5.7 (Table 9) compares AAC (default) to
Random-Subset, FPS-Subset (canonical, seed-vertex from LCC), and Greedy-Max.
This script adds a defensive baseline: ``FPS with random-restart'' -- run
plain FPS-ALT with R different random seed vertices, validate each on a
held-out 100-query split, and report the best. If random-restart FPS still
loses to Greedy-Max by less than a few percent, then the FPS baseline
cannot be dismissed as ``unlucky.''

For each (graph, K, seed) we generate 200 queries (100 val, 100 test), build
R independent FPS-ALT heuristics with random seed vertices drawn from the
graph's largest SCC, evaluate each on the val set, pick the best, and report
its test-set expansion reduction. Output is a tidy CSV that feeds an
additional column in the selection-ablation table.

Output: results/random_restart_fps/<graph>.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.graphs.io import load_graph_npz
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from experiments.utils import compute_strong_lcc, generate_queries

OUTPUT_DIR = _ROOT / "results" / "random_restart_fps"
SEEDS = [42, 123, 456, 789, 1024]
QUERY_SEED_BASE = 42

GRAPHS = {
    "modena":    "data/osmnx/modena.npz",
    "manhattan": "data/osmnx/manhattan.npz",
}


def run_graph(graph_key: str, K: int, R: int, seeds: list[int],
              num_val: int, num_test: int) -> list[dict]:
    graph = load_graph_npz(_ROOT / GRAPHS[graph_key])
    lcc_nodes, _ = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    n_total = num_val + num_test
    queries = generate_queries(graph, n_total, seed=QUERY_SEED_BASE)
    val_q, test_q = queries[:num_val], queries[num_val:]
    print(f"[{graph_key}] V={graph.num_nodes:,} LCC={len(lcc_nodes):,} "
          f"K={K} R={R}", flush=True)
    dij_test = np.array([dijkstra(graph, s, t).expansions for s, t in test_q])
    dij_test_mean = float(dij_test.mean())
    dij_val = np.array([dijkstra(graph, s, t).expansions for s, t in val_q])
    dij_val_mean = float(dij_val.mean())

    rows: list[dict] = []
    lcc_arr = np.array(lcc_nodes)
    for seed in seeds:
        rng_seed = np.random.default_rng(seed)
        starts = rng_seed.choice(lcc_arr, size=R, replace=False).tolist()
        best_val_red = -1.0
        best_test_red = None
        best_start = None
        for start in starts:
            labels = alt_preprocess(graph, K, seed_vertex=int(start),
                                    valid_vertices=lcc_tensor)
            h = make_alt_heuristic(labels)
            val_exp = float(np.array([
                astar(graph, s, t, heuristic=h).expansions for s, t in val_q
            ]).mean())
            val_red = 100.0 * (1.0 - val_exp / dij_val_mean)
            if val_red > best_val_red:
                best_val_red = val_red
                best_start = int(start)
                test_exp = float(np.array([
                    astar(graph, s, t, heuristic=h).expansions for s, t in test_q
                ]).mean())
                best_test_red = 100.0 * (1.0 - test_exp / dij_test_mean)
        # Canonical FPS reference (seed_vertex=lcc_nodes[0]) for context.
        rows.append({
            "graph": graph_key,
            "K": K,
            "R": R,
            "seed": seed,
            "best_seed_vertex": best_start,
            "val_red_pct": best_val_red,
            "test_red_pct": best_test_red,
        })
        print(f"  seed={seed} best_start={best_start} "
              f"val={best_val_red:.2f}% test={best_test_red:.2f}%",
              flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=list(GRAPHS.keys()))
    ap.add_argument("--K", type=int, default=8,
                    help="ALT K value (default 8 = 64 B/v on directed graphs)")
    ap.add_argument("--R", type=int, default=10,
                    help="Random restarts per seed (default 10).")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--num-val", type=int, default=100)
    ap.add_argument("--num-test", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for g in args.graphs:
        rows = run_graph(g, args.K, args.R, args.seeds,
                         args.num_val, args.num_test)
        out = args.out_dir / f"{g}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[{g}] wrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
