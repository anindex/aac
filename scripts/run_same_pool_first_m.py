#!/usr/bin/env python
"""Same-pool first-m wrapper diagnostic.

For each (graph, K0, m, seed) cell used by the AAC experiments, build
the teacher FPS pool of K0 landmarks, take the *first m* of those landmarks
(farthest-first ordering), and evaluate plain ALT on that m-subset. This is
the "ALT-pool (first m)" diagnostic that isolates *pool access* from
*training signal*:

    AAC learns m landmarks out of K0 from the same pool via a Gumbel-softmax
    selector; forcing the selector to the first m of the pool is the
    no-training baseline drawn from exactly the same candidate set. The gap
    between "ALT-pool (first m)" and "plain ALT with K=m" isolates whether a
    larger pool *by itself* helps ALT, independent of any learning; the gap
    between "AAC" and "ALT-pool (first m)" isolates the learning signal.

We sweep the same (graph, K0, m) configurations that Tables 3 (DIMACS main),
6 (OSMnx), and 14 (matched hybrid non-road) report. The per-seed query set is
generated with the same seed=42 convention as the rest of the paper, so the
numbers plug into those tables by seed and (K0, m) key.

Output: results/same_pool_first_m/<graph>.csv

Columns: graph, K0, m, budget_bpv, seed, mean_expansions, median_expansions,
    expansion_reduction_pct, dijkstra_mean.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from aac.baselines.alt import make_alt_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.io import load_graph_npz
from aac.graphs.loaders.dimacs import load_dimacs
from aac.graphs.types import Graph
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from experiments.utils import compute_strong_lcc, generate_queries

OUTPUT_DIR = _ROOT / "results" / "same_pool_first_m"
SEEDS = [42, 123, 456, 789, 1024]
NUM_QUERIES = 100
QUERY_SEED = 42

# ---- Per-graph configurations matched to the tables we feed ----
#
# DIMACS main table (Table 3/main_results): directed graphs, 3 budgets.
# AAC uses K0 in {32, 64, 128} and m from the best-K0 selection. For the
# diagnostic we use the K0 that the table reports; the first-m column picks
# the first m = ALT-equivalent bytes/budget anchors out of that pool.
#
# We key by (budget_bpv, K0, m) -> the concrete AAC config reported in the
# corresponding table.
DIMACS_CONFIGS = [
    # Values follow paper/table_main_dimacs.tex best-K0 entries.
    {"budget_bpv": 32,  "K0": 32,  "m": 4,  "directed": True},
    {"budget_bpv": 64,  "K0": 64,  "m": 8,  "directed": True},
    {"budget_bpv": 128, "K0": 128, "m": 16, "directed": True},
]
OSMNX_CONFIGS = [
    {"budget_bpv": 32,  "K0": 32,  "m": 4,  "directed": True},
    {"budget_bpv": 64,  "K0": 64,  "m": 8,  "directed": True},
    {"budget_bpv": 128, "K0": 128, "m": 16, "directed": True},
]
# Non-road graphs (matched_hybrid table): undirected; ALT K = B/4 = m.
NONROAD_CONFIGS = [
    {"budget_bpv": 32,  "K0": 32,  "m": 8,  "directed": False},
    {"budget_bpv": 64,  "K0": 64,  "m": 16, "directed": False},
    {"budget_bpv": 128, "K0": 128, "m": 32, "directed": False},
]

GRAPHS = {
    # DIMACS (paper Table tab:same-pool-firstm reports NY only on the
    # DIMACS side; FLA was exploratory and is not cited).
    "NY":  {"loader": "dimacs",
            "gr": "data/dimacs/USA-road-d.NY.gr",
            "co": "data/dimacs/USA-road-d.NY.co",
            "configs": DIMACS_CONFIGS},
    # OSMnx
    "modena":    {"loader": "osmnx", "path": "data/osmnx/modena.npz",
                  "configs": OSMNX_CONFIGS},
    "manhattan": {"loader": "osmnx", "path": "data/osmnx/manhattan.npz",
                  "configs": OSMNX_CONFIGS},
    "berlin":    {"loader": "osmnx", "path": "data/osmnx/berlin.npz",
                  "configs": OSMNX_CONFIGS},
    "los_angeles": {"loader": "osmnx", "path": "data/osmnx/los_angeles.npz",
                    "configs": OSMNX_CONFIGS},
    # Non-road (for matched_hybrid table). Use the synthetic generators.
    "sbm":       {"loader": "synth", "kind": "sbm",
                  "configs": NONROAD_CONFIGS},
    "ba":        {"loader": "synth", "kind": "ba",
                  "configs": NONROAD_CONFIGS},
    "ogb_arxiv": {"loader": "ogb_arxiv", "configs": NONROAD_CONFIGS},
}


def _load(graph_key: str) -> Graph:
    spec = GRAPHS[graph_key]
    kind = spec["loader"]
    if kind == "osmnx":
        return load_graph_npz(_ROOT / spec["path"])
    if kind == "dimacs":
        return load_dimacs(str(_ROOT / spec["gr"]),
                           str(_ROOT / spec["co"]))
    if kind == "synth":
        from run_synthetic_experiments import (
            generate_community_graph,
            generate_powerlaw_graph,
            nx_to_graph,
        )
        gen = (generate_community_graph if spec["kind"] == "sbm"
               else generate_powerlaw_graph)
        G_nx = gen(seed=42)
        return nx_to_graph(G_nx, weight_seed=42)
    if kind == "ogb_arxiv":
        # Lazy import to avoid torch_geometric dependency at module load.
        from aac.graphs.loaders.ogb import load_ogb_arxiv
        return load_ogb_arxiv(_ROOT / "data" / "ogb")
    raise ValueError(f"Unknown loader for graph {graph_key}: {kind}")


def _first_m_heuristic(graph: Graph, pool_K: int, m: int,
                       seed_vertex: int,
                       valid_vertices: torch.Tensor | None,
                       rng: torch.Generator):
    """Build ALT on the first m anchors of a K0-FPS pool.

    Since FPS is deterministic given (seed_vertex, valid_vertices), the first
    m anchors of a K0-pool are exactly the m anchors FPS would select if
    asked for m landmarks directly, starting from the same seed vertex; the
    point of this diagnostic is that the *teacher cost* was already paid for
    K0 but the *deployed* heuristic uses only m of them.
    """
    pool = farthest_point_sampling(
        graph, pool_K, seed_vertex=seed_vertex,
        rng=rng, valid_vertices=valid_vertices,
    )
    first_m = pool[:m]
    teacher = compute_teacher_labels(graph, first_m, use_gpu=False)
    return make_alt_heuristic(teacher)


def run_graph(graph_key: str, seeds: Iterable[int], num_queries: int) -> list[dict]:
    spec = GRAPHS[graph_key]
    graph = _load(graph_key)
    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    queries = generate_queries(graph, num_queries, seed=QUERY_SEED)

    print(f"[{graph_key}] V={graph.num_nodes:,} E={graph.num_edges:,} "
          f"LCC={len(lcc_nodes):,} directed={graph.is_directed}", flush=True)
    t0 = time.perf_counter()
    dij_exps = np.array([dijkstra(graph, s, t).expansions for s, t in queries])
    dij_mean = float(dij_exps.mean())
    print(f"[{graph_key}] Dijkstra mean={dij_mean:.0f} "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    rows = []
    for cfg in spec["configs"]:
        K0, m, bpv = cfg["K0"], cfg["m"], cfg["budget_bpv"]
        for seed in seeds:
            gen = torch.Generator().manual_seed(seed)
            t1 = time.perf_counter()
            h = _first_m_heuristic(graph, K0, m, lcc_seed, lcc_tensor, gen)
            prep = time.perf_counter() - t1
            exps = np.array([astar(graph, s, t, heuristic=h).expansions
                             for s, t in queries])
            mean_exp = float(exps.mean())
            reduction = 100.0 * (1.0 - mean_exp / dij_mean) if dij_mean > 0 else 0.0
            rows.append({
                "graph": graph_key,
                "K0": K0,
                "m": m,
                "budget_bpv": bpv,
                "seed": seed,
                "preprocess_s": prep,
                "mean_expansions": mean_exp,
                "median_expansions": float(np.median(exps)),
                "expansion_reduction_pct": reduction,
                "dijkstra_mean": dij_mean,
            })
            print(f"  K0={K0} m={m} B={bpv} seed={seed}: "
                  f"red={reduction:.2f}%  mean_exp={mean_exp:.0f}",
                  flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=sorted(GRAPHS.keys()))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for graph_key in args.graphs:
        rows = run_graph(graph_key, args.seeds, args.num_queries)
        out = OUTPUT_DIR / f"{graph_key}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[{graph_key}] wrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
