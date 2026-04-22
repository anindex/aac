#!/usr/bin/env python
"""CDH vs ALT matched-memory head-to-head (matched-memory head-to-head).

Runs ALT and CDH at matched bytes/vertex on one synthetic SBM instance and
emits a CSV consumable by the paper's CDH table. Keeps the protocol identical
to ``run_synthetic_experiments.py``:
- LCC restriction via ``compute_strong_lcc``
- FPS anchors seeded from the LCC
- Closed-set A* without node reopenings
- 5 seeds, 100 queries
- Dijkstra reference computed once per graph

CDH memory at P pivots, r stored, float32, undirected:
    bytes/vertex = r * (4 + ceil(log2(P)/8))
At P = 64 the index is 1 byte, so the matched rule is r = floor(B / 5). We
use P = 64 as the fixed pivot-pool size (Goldenberg et al. 2011 report
diminishing returns beyond ~64 pivots on road networks).

Output: results/cdh_baseline/sbm_cdh.csv

To extend to Modena / Manhattan / DIMACS graphs, import the respective loader
from ``aac.graphs.loaders`` and reuse the ``run_comparison`` function below.
The full paper sweep is documented in paper/main.tex (Section 5.9.4 / Table
``tab:cdh-reference``); for project context see AGENTS.md and for revision
archaeology run ``git log paper/main.tex``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import networkx as nx
import numpy as np
import torch

from aac.baselines import (
    alt_memory_bytes,
    alt_preprocess,
    cdh_memory_bytes,
    cdh_preprocess,
    make_alt_heuristic,
    make_cdh_heuristic,
)
from aac.graphs.convert import edges_to_graph
from aac.graphs.types import Graph
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from experiments.utils import compute_strong_lcc, generate_queries

OUTPUT_DIR = _PROJECT_ROOT / "results" / "cdh_baseline"
GRAPH_SEED = 42
NUM_QUERIES = 100
QUERY_SEED = 42
SEEDS = [42, 123, 456, 789, 1024]

# Matched-memory schedule. At float32 on undirected graphs:
#   ALT: K floats = 4K bytes, so K = B/4.
#   CDH at P=64: r * (4 + 1) bytes = 5r, so r = B/5 rounded down.
BUDGETS = [
    {"label": "32", "alt_K": 8, "cdh_P": 64, "cdh_r": 6},
    {"label": "64", "alt_K": 16, "cdh_P": 64, "cdh_r": 12},
    {"label": "128", "alt_K": 32, "cdh_P": 64, "cdh_r": 25},
]


def _generate_sbm() -> Graph:
    """5-community SBM with 10k vertices. Matches run_synthetic_experiments.py."""
    sizes = [2000] * 5
    p = [
        [0.05, 0.001, 0.001, 0.001, 0.001],
        [0.001, 0.05, 0.001, 0.001, 0.001],
        [0.001, 0.001, 0.05, 0.001, 0.001],
        [0.001, 0.001, 0.001, 0.05, 0.001],
        [0.001, 0.001, 0.001, 0.001, 0.05],
    ]
    G = nx.stochastic_block_model(sizes, p, seed=GRAPH_SEED)
    rng = np.random.default_rng(GRAPH_SEED)
    edges = list(G.edges())
    src = torch.tensor([u for u, _ in edges], dtype=torch.int64)
    tgt = torch.tensor([v for _, v in edges], dtype=torch.int64)
    wts = torch.tensor(rng.uniform(1.0, 10.0, size=len(edges)), dtype=torch.float64)
    return edges_to_graph(src, tgt, wts, num_nodes=G.number_of_nodes(), is_directed=False)


def _run_seeded(graph: Graph, lcc_seed: int, lcc_tensor: torch.Tensor,
                budget: dict, seed: int, queries: list, dij_exps: np.ndarray) -> list[dict]:
    """Run ALT and CDH once for a (budget, seed) cell and return two rows."""
    rows = []
    alt_K = budget["alt_K"]
    P = budget["cdh_P"]
    r = budget["cdh_r"]
    bpv = budget["label"]

    # --- ALT arm ---
    gen = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    alt_labels = alt_preprocess(graph, alt_K, seed_vertex=lcc_seed,
                                rng=gen, valid_vertices=lcc_tensor)
    alt_h = make_alt_heuristic(alt_labels)
    alt_prep = time.perf_counter() - t0
    alt_exps = np.array([astar(graph, s, t, heuristic=alt_h).expansions
                         for s, t in queries])
    rows.append({
        "method": "ALT",
        "budget_bpv": bpv,
        "seed": seed,
        "config": f"K={alt_K}",
        "memory_bytes_per_vertex": alt_memory_bytes(alt_K, is_directed=False),
        "preprocess_time_s": alt_prep,
        "mean_expansions": float(alt_exps.mean()),
        "median_expansions": float(np.median(alt_exps)),
        "mean_reduction_vs_dijkstra": float(1.0 - alt_exps.mean() / dij_exps.mean()),
    })

    # --- CDH arm ---
    gen = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    cdh_labels = cdh_preprocess(
        graph, num_pivots=P, num_stored=r,
        seed_vertex=lcc_seed, rng=gen, valid_vertices=lcc_tensor,
        selection_rule="top_r_farthest", selection_seed=seed,
    )
    cdh_h = make_cdh_heuristic(cdh_labels)
    cdh_prep = time.perf_counter() - t0
    cdh_exps = np.array([astar(graph, s, t, heuristic=cdh_h).expansions
                         for s, t in queries])
    rows.append({
        "method": "CDH",
        "budget_bpv": bpv,
        "seed": seed,
        "config": f"P={P},r={r}",
        "memory_bytes_per_vertex": cdh_memory_bytes(P, r, is_directed=False),
        "preprocess_time_s": cdh_prep,
        "mean_expansions": float(cdh_exps.mean()),
        "median_expansions": float(np.median(cdh_exps)),
        "mean_reduction_vs_dijkstra": float(1.0 - cdh_exps.mean() / dij_exps.mean()),
    })
    return rows


def run_comparison(graph: Graph, graph_name: str, seeds: list[int],
                   budgets: list[dict], num_queries: int) -> list[dict]:
    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    queries = generate_queries(graph, num_queries, seed=QUERY_SEED)

    print(f"[{graph_name}] V={graph.num_nodes:,} E={graph.num_edges:,} "
          f"LCC={len(lcc_nodes):,}")
    print(f"[{graph_name}] Dijkstra reference ({num_queries} queries) ...")
    dij_exps = np.array([dijkstra(graph, s, t).expansions for s, t in queries])
    print(f"[{graph_name}] Dijkstra mean expansions = {dij_exps.mean():.0f}")

    all_rows = []
    for budget in budgets:
        print(f"[{graph_name}] budget {budget['label']} B/v "
              f"(ALT K={budget['alt_K']}, CDH P={budget['cdh_P']} r={budget['cdh_r']})")
        for seed in seeds:
            rows = _run_seeded(graph, lcc_seed, lcc_tensor, budget, seed,
                               queries, dij_exps)
            for row in rows:
                row["graph"] = graph_name
                print(f"    seed={seed} {row['method']:3s} "
                      f"mean_exp={row['mean_expansions']:.0f} "
                      f"reduction={row['mean_reduction_vs_dijkstra']*100:.1f}%")
                all_rows.append(row)
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="sbm", choices=["sbm"],
                    help="Currently only 'sbm' is supported; DIMACS/OSMnx "
                         "runs are documented in paper/main.tex (Section "
                         "5.9.4, Table tab:cdh-reference).")
    ap.add_argument("--smoke", action="store_true",
                    help="Run a single seed x first budget x 20 queries for a "
                         "quick smoke test (~1 min).")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = _generate_sbm()

    if args.smoke:
        rows = run_comparison(
            graph, "sbm_smoke",
            seeds=SEEDS[:1], budgets=BUDGETS[:1], num_queries=20,
        )
        out = OUTPUT_DIR / "sbm_cdh_smoke.csv"
    else:
        rows = run_comparison(graph, "sbm", SEEDS, BUDGETS, NUM_QUERIES)
        out = OUTPUT_DIR / "sbm_cdh.csv"

    if not rows:
        print("No rows produced; aborting.")
        return 1
    fieldnames = ["graph", "method", "budget_bpv", "seed", "config",
                  "memory_bytes_per_vertex", "preprocess_time_s",
                  "mean_expansions", "median_expansions",
                  "mean_reduction_vs_dijkstra"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
