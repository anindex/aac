#!/usr/bin/env python
"""FastMap admissibility verification: save per-query evidence.

Runs FastMap on NY and FLA, saves per-query (cost, dijkstra_cost, is_violation)
to provide concrete evidence for the "100% suboptimal" claim.

Output: results/dimacs/fastmap_admissibility.csv
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import scipy.sparse.csgraph

from aac.baselines.fastmap import fastmap_preprocess, make_fastmap_heuristic
from aac.graphs.convert import graph_to_scipy
from aac.graphs.loaders.dimacs import load_dimacs
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from experiments.utils import generate_queries

DATA_DIR = Path("data/dimacs")
OUTPUT_DIR = Path("results/dimacs")
GRAPHS = {
    "NY": ("USA-road-d.NY.gr", "USA-road-d.NY.co"),
    "FLA": ("USA-road-d.FLA.gr", "USA-road-d.FLA.co"),
}
NUM_QUERIES = 100
FASTMAP_DIMS = [8, 16, 32, 64]
SEED = 42
ATOL = 1e-6


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for graph_name, (gr_file, co_file) in GRAPHS.items():
        gr_path = DATA_DIR / gr_file
        co_path = DATA_DIR / co_file
        if not gr_path.exists():
            print(f"WARNING: {gr_path} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  FastMap admissibility: {graph_name}")
        print(f"{'='*60}")

        graph = load_dimacs(str(gr_path), str(co_path) if co_path.exists() else None)
        print(f"  {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")

        queries = generate_queries(graph, NUM_QUERIES, seed=SEED)

        # Dijkstra reference (once per graph)
        print("  Computing Dijkstra reference...")
        dij_results = [dijkstra(graph, s, t) for s, t in queries]
        dij_costs = [r.cost for r in dij_results]
        dij_exps = [r.expansions for r in dij_results]

        for d in FASTMAP_DIMS:
            print(f"\n  FastMap d={d}:")
            t0 = time.perf_counter()
            coords = fastmap_preprocess(graph, d)
            h = make_fastmap_heuristic(coords)
            prep_time = time.perf_counter() - t0
            print(f"    Preprocessing: {prep_time:.1f}s")

            violations = 0
            for i, (s, t) in enumerate(queries):
                result = astar(graph, s, t, heuristic=h)
                fm_cost = result.cost
                dij_cost = dij_costs[i]
                is_violation = fm_cost - dij_cost > ATOL
                if is_violation:
                    violations += 1
                cost_ratio = fm_cost / dij_cost if dij_cost > 0 else float("nan")
                all_rows.append({
                    "graph": graph_name,
                    "fastmap_dims": d,
                    "query_idx": i,
                    "source": queries[i][0],
                    "target": queries[i][1],
                    "fastmap_cost": fm_cost,
                    "dijkstra_cost": dij_cost,
                    "cost_diff": fm_cost - dij_cost,
                    "cost_ratio": cost_ratio,
                    "is_violation": is_violation,
                    "fastmap_expansions": result.expansions,
                    "dijkstra_expansions": dij_exps[i],
                })

            pct = 100.0 * violations / NUM_QUERIES
            print(f"    Violations: {violations}/{NUM_QUERIES} ({pct:.1f}%)")
            cost_ratios = [r["cost_ratio"] for r in all_rows[-NUM_QUERIES:] if not np.isnan(r["cost_ratio"])]
            if cost_ratios:
                print(f"    Cost ratio (FM/Dij): mean={np.mean(cost_ratios):.4f}, max={np.max(cost_ratios):.4f}")

    # Write results
    csv_path = OUTPUT_DIR / "fastmap_admissibility.csv"
    cols = ["graph", "fastmap_dims", "query_idx", "source", "target",
            "fastmap_cost", "dijkstra_cost", "cost_diff", "cost_ratio",
            "is_violation", "fastmap_expansions", "dijkstra_expansions"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    import pandas as pd
    df = pd.DataFrame(all_rows)
    summary = df.groupby(["graph", "fastmap_dims"]).agg(
        n_queries=("query_idx", "count"),
        n_violations=("is_violation", "sum"),
        violation_pct=("is_violation", lambda x: 100.0 * x.sum() / len(x)),
        mean_cost_ratio=("cost_ratio", "mean"),
        max_cost_ratio=("cost_ratio", "max"),
    ).reset_index()

    summary_path = OUTPUT_DIR / "fastmap_admissibility_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.4f")

    print(f"\n{'='*60}")
    print("  Summary:")
    for _, row in summary.iterrows():
        print(f"    {row['graph']} d={int(row['fastmap_dims'])}: "
              f"{int(row['n_violations'])}/{int(row['n_queries'])} violations "
              f"({row['violation_pct']:.1f}%), "
              f"mean cost ratio={row['mean_cost_ratio']:.4f}")

    print(f"\nResults: {csv_path}, {summary_path}")


if __name__ == "__main__":
    main()
