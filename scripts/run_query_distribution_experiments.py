#!/usr/bin/env python
"""Evaluate AAC vs ALT under non-uniform query distributions.

This script tests the reviewer's most important missing experiment: whether
learned landmark compression gains anything when endpoint demand is no longer
uniform. It reuses the existing AAC/ALT preprocessing helpers and compares
three demand patterns:

- uniform: baseline random queries over the largest connected component
- hotspot: endpoints concentrated on a small hotspot set
- powerlaw: endpoints sampled from a degree-weighted distribution

Outputs:
  results/query_distributions/query_mode_results.csv
  results/query_distributions/query_mode_summary.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
# Make src/ importable so `experiments` resolves to src/experiments/. The
# repo root is also on the path so `scripts.*` self-imports continue to
# resolve.
_SRC_DIR = str(Path(_PROJECT_ROOT) / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.graphs.io import load_graph_npz
from aac.graphs.loaders.dimacs import load_dimacs
from experiments.utils import generate_queries
from scripts.run_ablation_selection import (
    dijkstra_baseline,
    get_lcc,
    run_astar_queries,
    train_aac,
)

GRAPHS = [
    {
        "name": "NY",
        "kind": "dimacs",
        "gr_path": Path("data/dimacs/USA-road-d.NY.gr"),
        "co_path": Path("data/dimacs/USA-road-d.NY.co"),
    },
    {
        "name": "manhattan",
        "kind": "osmnx",
        "path": Path("data/osmnx/manhattan.npz"),
    },
]
QUERY_MODES = {
    "uniform": {},
    "hotspot": {"hotspot_nodes": 8, "hotspot_mix": 0.90},
    "powerlaw": {"powerlaw_alpha": 1.5},
}
SEEDS = [42, 123, 456, 789, 1024]
NUM_QUERIES = 100
K0 = 64
M = 16
OUTPUT_DIR = Path("results/query_distributions")


def load_graph(spec: dict):
    """Load a graph from a graph specification entry."""
    if spec["kind"] == "dimacs":
        return load_dimacs(str(spec["gr_path"]), str(spec["co_path"]))
    if spec["kind"] == "osmnx":
        return load_graph_npz(str(spec["path"]))
    raise ValueError(f"Unsupported graph kind: {spec['kind']}")


def matched_alt_landmarks(graph, m: int) -> int:
    """Return ALT landmark count that matches an m-float AAC budget."""
    directional_factor = 2 if getattr(graph, "is_directed", False) else 1
    return max(1, m // directional_factor)


def make_hybrid_heuristic(aac_h, alt_h):
    """Pointwise max of two admissible heuristics."""
    def hybrid(node: int, target: int) -> float:
        return max(aac_h(node, target), alt_h(node, target))

    return hybrid


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print("=" * 72)
    print("Query-Distribution Sensitivity: AAC vs ALT vs Hybrid")
    print("=" * 72)
    print(f"Graphs: {[g['name'] for g in GRAPHS]}")
    print(f"Modes: {list(QUERY_MODES)} | Seeds: {SEEDS} | Queries/mode: {NUM_QUERIES}")
    print(f"AAC: K0={K0}, m={M}")
    print("=" * 72)

    for graph_spec in GRAPHS:
        graph_name = graph_spec["name"]
        print(f"\n--- Loading {graph_name} ---")
        graph = load_graph(graph_spec)
        lcc_nodes, lcc_seed = get_lcc(graph)
        alt_k = matched_alt_landmarks(graph, M)
        alt_bpv = alt_k * (8 if graph.is_directed else 4)

        print(
            f"Graph: {graph_name} | directed={graph.is_directed} | "
            f"AAC bytes/v={M * 4} | ALT bytes/v={alt_bpv} (K={alt_k})"
        )

        for seed in SEEDS:
            print(f"  seed={seed}: preprocessing AAC...", end=" ", flush=True)
            t0 = time.perf_counter()
            aac_h, _teacher_labels, _anchors, _compressor, _metrics = train_aac(
                graph, K0, M, seed, lcc_nodes, lcc_seed
            )
            aac_preprocess_sec = time.perf_counter() - t0
            print(f"done ({aac_preprocess_sec:.1f}s)")

            print(f"  seed={seed}: preprocessing ALT...", end=" ", flush=True)
            t0 = time.perf_counter()
            rng = torch.Generator().manual_seed(seed)
            alt_teacher = alt_preprocess(
                graph,
                alt_k,
                seed_vertex=lcc_seed,
                rng=rng,
                valid_vertices=lcc_nodes,
            )
            alt_h = make_alt_heuristic(alt_teacher)
            alt_preprocess_sec = time.perf_counter() - t0
            print(f"done ({alt_preprocess_sec:.1f}s)")

            hybrid_h = make_hybrid_heuristic(aac_h, alt_h)

            for query_mode, mode_kwargs in QUERY_MODES.items():
                queries = generate_queries(
                    graph,
                    NUM_QUERIES,
                    seed=seed,
                    mode=query_mode,
                    **mode_kwargs,
                )
                dij_exps = dijkstra_baseline(graph, queries)
                dij_mean = dij_exps.mean()

                for method_name, heuristic, preprocess_sec, bytes_per_vertex in [
                    ("AAC", aac_h, aac_preprocess_sec, M * 4),
                    ("ALT", alt_h, alt_preprocess_sec, alt_bpv),
                    ("Hybrid", hybrid_h, aac_preprocess_sec + alt_preprocess_sec, M * 4 + alt_bpv),
                ]:
                    exps, all_optimal = run_astar_queries(graph, queries, heuristic)
                    reduction_pct = 100.0 * (1 - exps.mean() / dij_mean)
                    rows.append(
                        {
                            "graph": graph_name,
                            "graph_kind": graph_spec["kind"],
                            "seed": seed,
                            "query_mode": query_mode,
                            "method": method_name,
                            "K0": K0 if method_name != "ALT" else 0,
                            "m": M if method_name != "ALT" else 0,
                            "alt_K": alt_k,
                            "bytes_per_vertex": bytes_per_vertex,
                            "num_queries": NUM_QUERIES,
                            "mean_expansions": float(exps.mean()),
                            "median_expansions": float(np.median(exps)),
                            "reduction_pct": float(reduction_pct),
                            "all_optimal": bool(all_optimal),
                            "preprocess_sec": float(preprocess_sec),
                        }
                    )

                print(
                    f"    {query_mode:8s}: "
                    f"AAC={rows[-3]['reduction_pct']:.1f}%  "
                    f"ALT={rows[-2]['reduction_pct']:.1f}%  "
                    f"Hybrid={rows[-1]['reduction_pct']:.1f}%"
                )

    results_path = OUTPUT_DIR / "query_mode_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = (
        pd.DataFrame(rows)
        .groupby(["graph", "graph_kind", "query_mode", "method", "bytes_per_vertex"], as_index=False)
        .agg(
            reduction_pct_mean=("reduction_pct", "mean"),
            reduction_pct_std=("reduction_pct", "std"),
            mean_expansions_mean=("mean_expansions", "mean"),
            preprocess_sec_mean=("preprocess_sec", "mean"),
            all_optimal=("all_optimal", "all"),
        )
    )
    summary_path = OUTPUT_DIR / "query_mode_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.4f")

    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
