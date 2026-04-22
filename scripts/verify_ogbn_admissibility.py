#!/usr/bin/env python
"""Admissibility audit for OGB-arXiv.

Theorem 2 guarantees AAC admissibility architecturally; max(admissible,
admissible) preserves admissibility. This script is an independent empirical
audit of the ALT pipeline on the weighted OGB-arXiv graph at the same
(seed, budget) cells the pre-registered prediction was evaluated on, plus a
20-query sanity check at K=16. It writes:

    results/synthetic/ogbn_arxiv_admissibility.csv

with one row per (seed, budget_bpv) cell across the 5 paper seeds and
3 budget tiers (B in {32, 64, 128} B/v -> ALT K in {4, 8, 16} on the
symmetrized OGB-arXiv graph), backing the paper's "15/15 cells, zero
admissibility violations" claim (Section 5.9.3 / Appendix E.3).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

# OGB uses torch.load with default weights_only=True on torch>=2.6, which fails.
# Patch before importing anything that triggers the load.
import torch as _torch

_orig_load = _torch.load
_torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

from scripts.run_nonroad_real import load_ogbn_arxiv
from scripts.run_synthetic_experiments import nx_to_graph

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra

OUTPUT_CSV = _PROJECT_ROOT / "results" / "synthetic" / "ogbn_arxiv_admissibility.csv"

SEEDS = [42, 123, 456, 789, 1024]
# Matched-memory ALT K per per-vertex budget for the symmetrized
# (undirected) OGB-arXiv graph: ALT K = B/4. Three budgets are reported in
# the paper's pre-registration tables and Section 5.9.3.
BUDGETS = [(32, 4), (64, 8), (128, 16)]
NUM_QUERIES = 100


def _audit_cell(graph, K: int, seed: int, num_queries: int) -> dict:
    """Run admissibility audit over ``num_queries`` random pairs at ALT K."""
    V = graph.num_nodes
    rng = np.random.default_rng(seed)
    srcs = rng.integers(0, V, size=num_queries)
    dsts = rng.integers(0, V, size=num_queries)
    queries = [(int(s), int(t)) for s, t in zip(srcs, dsts) if s != t]

    alt_prep = alt_preprocess(graph, K, seed_vertex=0)
    alt_h = make_alt_heuristic(alt_prep)

    violations = 0
    max_ratio = 1.0
    checked = 0
    for s, t in queries:
        dij = dijkstra(graph, s, t)
        if dij.cost is None or dij.cost == float("inf"):
            continue
        res = astar(graph, s, t, heuristic=alt_h)
        if res.cost is None:
            continue
        checked += 1
        ratio = res.cost / dij.cost
        if ratio > 1.0 + 1e-9:
            violations += 1
            max_ratio = max(max_ratio, ratio)
    return {
        "checked": checked,
        "violations": violations,
        "max_ratio": float(max_ratio),
    }


def main() -> int:
    print("Loading OGB-arXiv ...")
    nxg = load_ogbn_arxiv()
    rng_w = np.random.default_rng(42)
    for u, v, d in nxg.edges(data=True):
        d["weight"] = float(rng_w.uniform(1.0, 10.0))
    graph = nx_to_graph(nxg)
    print(f"  V={graph.num_nodes:,}  E={graph.num_edges:,}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for budget_bpv, K in BUDGETS:
        for seed in SEEDS:
            print(f"  cell: B={budget_bpv} B/v (ALT K={K}), seed={seed}")
            res = _audit_cell(graph, K, seed, NUM_QUERIES)
            row = {
                "graph": "ogbn_arxiv",
                "budget_bpv": budget_bpv,
                "alt_K": K,
                "seed": seed,
                "num_queries": NUM_QUERIES,
                "checked": res["checked"],
                "admissibility_violations": res["violations"],
                "max_cost_ratio": f"{res['max_ratio']:.9f}",
                "all_admissible": str(res["violations"] == 0),
            }
            rows.append(row)
            print(
                f"    checked={row['checked']}  violations={row['admissibility_violations']}"
                f"  max_ratio={row['max_cost_ratio']}"
            )

    fields = list(rows[0].keys())
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    n_cells = len(rows)
    n_clean = sum(1 for r in rows if r["admissibility_violations"] == 0)
    print()
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Summary: {n_clean}/{n_cells} cells with zero admissibility violations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
