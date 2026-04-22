#!/usr/bin/env python
"""Compute FPS covering radii for Modena (OSMnx) and NY (DIMACS).

For each graph and K in {4, 8, 16, 32, 64}:
  - Run FPS to select K anchors (restricted to LCC)
  - Compute teacher labels via compute_teacher_labels
  - Compute covering radius as max_v(min_k(d_out[k,v]))
  - For directed graphs, also compute backward covering radius from d_in

Output: results/covering_radius.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.loaders.dimacs import load_dimacs
from aac.graphs.loaders.osmnx import load_osmnx
from experiments.utils import compute_strong_lcc

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "covering_radius.csv")

K_VALUES = [4, 8, 16, 32, 64]
SENTINEL = 1e18


def covering_radius_from_d(d: np.ndarray) -> float:
    """Compute covering radius: max_v(min_k(d[k,v])).

    Masks sentinel values (>= 0.99 * SENTINEL) as inf before computing
    the per-vertex minimum across anchors, then takes the max over vertices.

    Args:
        d: (K, V) distance array.

    Returns:
        Covering radius (float).
    """
    d_safe = np.where(d < 0.99 * SENTINEL, d, np.inf)
    # min over anchors (axis=0) gives shape (V,)
    min_per_vertex = np.min(d_safe, axis=0)
    # Exclude vertices that are inf (unreachable from all anchors)
    finite_mask = np.isfinite(min_per_vertex)
    if not finite_mask.any():
        return float("inf")
    return float(np.max(min_per_vertex[finite_mask]))


def main() -> None:
    torch.manual_seed(42)

    # ---------------------------------------------------------------
    # Load graphs
    # ---------------------------------------------------------------
    print("Loading Modena (OSMnx)...")
    t0 = time.perf_counter()
    graph_modena = load_osmnx("modena")
    print(f"  {graph_modena.num_nodes:,} nodes, {graph_modena.num_edges:,} edges, "
          f"directed={graph_modena.is_directed}  ({time.perf_counter() - t0:.1f}s)")

    print("Loading NY (DIMACS)...")
    t0 = time.perf_counter()
    gr_path = os.path.join(PROJECT_ROOT, "data", "dimacs", "USA-road-d.NY.gr")
    co_path = os.path.join(PROJECT_ROOT, "data", "dimacs", "USA-road-d.NY.co")
    graph_ny = load_dimacs(gr_path, co_path)
    print(f"  {graph_ny.num_nodes:,} nodes, {graph_ny.num_edges:,} edges, "
          f"directed={graph_ny.is_directed}  ({time.perf_counter() - t0:.1f}s)")

    graphs = [
        ("modena", graph_modena),
        ("NY", graph_ny),
    ]

    # ---------------------------------------------------------------
    # Compute covering radii
    # ---------------------------------------------------------------
    rows: list[dict] = []

    for graph_name, graph in graphs:
        is_directed = getattr(graph, "is_directed", False)
        print(f"\n{'='*60}")
        print(f"Graph: {graph_name}  (directed={is_directed})")
        print(f"{'='*60}")

        # Compute LCC
        lcc_nodes, lcc_seed = compute_strong_lcc(graph)
        lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
        print(f"  LCC: {len(lcc_nodes):,} / {graph.num_nodes:,} nodes")

        for K in K_VALUES:
            print(f"\n  K={K}:")
            t0 = time.perf_counter()

            # FPS anchor selection (restricted to LCC)
            anchors = farthest_point_sampling(
                graph, K, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
            )
            print(f"    FPS done ({time.perf_counter() - t0:.1f}s)")

            # Compute teacher labels
            t1 = time.perf_counter()
            labels = compute_teacher_labels(
                graph, anchors, use_gpu=False, backend="scipy",
            )
            print(f"    SSSP done ({time.perf_counter() - t1:.1f}s)")

            # Forward covering radius: max_v(min_k(d_out[k,v]))
            d_out_np = labels.d_out.numpy()
            cr_fwd = covering_radius_from_d(d_out_np)
            print(f"    Covering radius (fwd): {cr_fwd:,.1f}")

            row = {
                "graph": graph_name,
                "K": K,
                "covering_radius": cr_fwd,
                "is_directed": is_directed,
            }

            # For directed graphs, also compute backward covering radius
            if is_directed:
                d_in_np = labels.d_in.numpy()
                cr_bwd = covering_radius_from_d(d_in_np)
                print(f"    Covering radius (bwd): {cr_bwd:,.1f}")
                row["covering_radius_bwd"] = cr_bwd

            rows.append(row)

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fieldnames = ["graph", "K", "covering_radius", "is_directed"]
    # Add backward column if any directed graph was processed
    if any(r.get("covering_radius_bwd") is not None for r in rows):
        fieldnames.append("covering_radius_bwd")

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {OUTPUT_PATH}")

    # Print summary table
    print(f"\n{'graph':<10} {'K':>4} {'cr_fwd':>14} {'cr_bwd':>14} {'directed':>9}")
    print("-" * 55)
    for r in rows:
        bwd = r.get("covering_radius_bwd", "")
        if bwd != "":
            bwd = f"{bwd:>14,.1f}"
        print(f"{r['graph']:<10} {r['K']:>4} {r['covering_radius']:>14,.1f} {bwd:>14} {str(r['is_directed']):>9}")


if __name__ == "__main__":
    main()
