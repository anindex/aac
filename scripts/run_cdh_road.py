#!/usr/bin/env python
r"""CDH vs ALT matched-memory head-to-head on road networks.

Extends scripts/run_cdh_baseline.py to four directed road graphs (OSMnx Modena,
OSMnx Manhattan, DIMACS NY, DIMACS FLA). Same protocol as the SBM benchmark:
matched bytes/vertex with P=64 fixed CDH pivot pool, 5 seeds, 100 queries, and
*three* CDH variants --- the closed-set "intersection" heuristic, the
closed-set "+sub" bound-substitution heuristic that uses the ``P*P`` pivot-
pivot side-table, and "+sub+BPMX" which additionally enables Felner-style
one-step Bidirectional Pathmax during A* expansion (sound under closed-set A*
without reopenings; see ``aac.search.astar(use_bpmx=True)``). The BPMX arm
brings CDH to the "original protocol" CDH baseline as evaluated by Goldenberg
et al.\ (2017, AI Communications).

Matched-memory rule (directed graphs, float32):
  ALT:  K = B / 8   (2K floats, since both forward and backward stored)
  CDH:  bytes = r*(2*4 + ceil(log2(P)/8)) = r*(8+1) = 9r at P=64
        => r = floor(B / 9)
  The ``P*P`` pivot-pivot side-table is fixed off-heap preprocessing and is
  NOT charged to the per-vertex budget (see cdh.py docstring).

Output: results/cdh_baseline/<graph>_cdh.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

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
from aac.graphs.io import load_graph_npz
from aac.graphs.loaders.dimacs import load_dimacs
from aac.graphs.types import Graph
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from experiments.utils import compute_strong_lcc, generate_queries

OUTPUT_DIR = _PROJECT_ROOT / "results" / "cdh_baseline"
NUM_QUERIES = 100
QUERY_SEED = 42
SEEDS = [42, 123, 456, 789, 1024]

# Directed-graph matched memory at float32, P=64 (1-byte index):
#   ALT bytes = 8K  -> K = B/8
#   CDH bytes = r*(2*4 + 1) = 9r  -> r = floor(B/9)
BUDGETS_DIRECTED = [
    {"label": "32",  "alt_K": 4,  "cdh_P": 64, "cdh_r": 3},
    {"label": "64",  "alt_K": 8,  "cdh_P": 64, "cdh_r": 7},
    {"label": "128", "alt_K": 16, "cdh_P": 64, "cdh_r": 14},
]

# Paper Table tab:cdh-reference reports CDH on SBM (handled by
# run_cdh_baseline.py), OSMnx Modena (directed), and DIMACS NY (directed).
# Manhattan and FLA were exploratory and are not in the paper; the pipeline
# limits this script's outputs to the paper-cited graph set.
GRAPHS = {
    "modena":    {"loader": "osmnx", "path": "data/osmnx/modena.npz"},
    "ny":        {"loader": "dimacs", "gr": "data/dimacs/USA-road-d.NY.gr",
                  "co": "data/dimacs/USA-road-d.NY.co"},
}


def _load(graph_key: str) -> Graph:
    spec = GRAPHS[graph_key]
    if spec["loader"] == "osmnx":
        return load_graph_npz(_PROJECT_ROOT / spec["path"])
    return load_dimacs(_PROJECT_ROOT / spec["gr"], _PROJECT_ROOT / spec["co"])


def _run_seeded(graph: Graph, lcc_seed: int, lcc_tensor: torch.Tensor,
                budget: dict, seed: int, queries: list,
                dij_exps: np.ndarray) -> list[dict]:
    rows = []
    alt_K = budget["alt_K"]
    P = budget["cdh_P"]
    r = budget["cdh_r"]
    bpv = budget["label"]

    # ALT arm
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
        "memory_bytes_per_vertex": alt_memory_bytes(alt_K, is_directed=True),
        "preprocess_time_s": alt_prep,
        "query_time_s": 0.0,
        "mean_expansions": float(alt_exps.mean()),
        "median_expansions": float(np.median(alt_exps)),
        "mean_reduction_vs_dijkstra": float(1.0 - alt_exps.mean() / dij_exps.mean()),
    })

    # CDH preprocessing (shared between "intersection" and "+sub" variants:
    # the P*P pivot-pivot side-table is populated once and reused).
    gen = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    cdh_labels = cdh_preprocess(
        graph, num_pivots=P, num_stored=r,
        seed_vertex=lcc_seed, rng=gen, valid_vertices=lcc_tensor,
        selection_rule="top_r_farthest", selection_seed=seed,
        compute_pivot_pivot=True,
    )
    cdh_prep = time.perf_counter() - t0

    cdh_variants = (
        ("CDH",          False, False),
        ("CDH+sub",      True,  False),
        ("CDH+sub+BPMX", True,  True),
    )
    for variant_label, use_sub, use_bpmx in cdh_variants:
        cdh_h = make_cdh_heuristic(cdh_labels, use_bound_substitution=use_sub)
        t1 = time.perf_counter()
        cdh_exps = np.array([
            astar(graph, s, t, heuristic=cdh_h, use_bpmx=use_bpmx).expansions
            for s, t in queries
        ])
        query_time = time.perf_counter() - t1
        rows.append({
            "method": variant_label,
            "budget_bpv": bpv,
            "seed": seed,
            "config": f"P={P},r={r}",
            "memory_bytes_per_vertex": cdh_memory_bytes(P, r, is_directed=True),
            "preprocess_time_s": cdh_prep if (not use_sub and not use_bpmx) else 0.0,
            "query_time_s": query_time,
            "mean_expansions": float(cdh_exps.mean()),
            "median_expansions": float(np.median(cdh_exps)),
            "mean_reduction_vs_dijkstra": float(
                1.0 - cdh_exps.mean() / dij_exps.mean()
            ),
        })
    return rows


def run_comparison(graph: Graph, graph_name: str, seeds: list[int],
                   budgets: list[dict], num_queries: int) -> list[dict]:
    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    queries = generate_queries(graph, num_queries, seed=QUERY_SEED)

    print(f"[{graph_name}] V={graph.num_nodes:,} E={graph.num_edges:,} "
          f"LCC={len(lcc_nodes):,}", flush=True)
    print(f"[{graph_name}] Dijkstra reference ({num_queries} queries) ...",
          flush=True)
    t0 = time.perf_counter()
    dij_exps = np.array([dijkstra(graph, s, t).expansions for s, t in queries])
    print(f"[{graph_name}] Dijkstra mean expansions = {dij_exps.mean():.0f} "
          f"(took {time.perf_counter()-t0:.1f}s)", flush=True)

    all_rows = []
    for budget in budgets:
        print(f"[{graph_name}] budget {budget['label']} B/v "
              f"(ALT K={budget['alt_K']}, CDH P={budget['cdh_P']} "
              f"r={budget['cdh_r']})", flush=True)
        for seed in seeds:
            rows = _run_seeded(graph, lcc_seed, lcc_tensor, budget, seed,
                               queries, dij_exps)
            for row in rows:
                row["graph"] = graph_name
                print(f"    seed={seed} {row['method']:3s} "
                      f"mean_exp={row['mean_expansions']:.0f} "
                      f"reduction={row['mean_reduction_vs_dijkstra']*100:.1f}%",
                      flush=True)
                all_rows.append(row)
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, choices=sorted(GRAPHS.keys()))
    ap.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = _load(args.graph)
    rows = run_comparison(graph, args.graph, args.seeds, BUDGETS_DIRECTED,
                          args.num_queries)
    out = OUTPUT_DIR / f"{args.graph}_cdh.csv"
    fieldnames = ["graph", "method", "budget_bpv", "seed", "config",
                  "memory_bytes_per_vertex", "preprocess_time_s",
                  "query_time_s",
                  "mean_expansions", "median_expansions",
                  "mean_reduction_vs_dijkstra"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
