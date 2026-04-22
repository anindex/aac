#!/usr/bin/env python
"""Timing benchmarks and Wilcoxon significance tests: AAC vs ALT on DIMACS graphs.

Runs AAC (K0=64, m=16) and ALT (16 landmarks) on all 4 DIMACS road
networks (NY, BAY, COL, FLA) with paired queries (same seed). Collects:
    - Offline preprocessing breakdown (anchor selection, SSSP, training, total)
    - Per-query latency (ms) and node expansions
    - Aggregate p50/p95 timing percentiles
    - Wilcoxon signed-rank p-values for expansion comparisons

Outputs:
    results/dimacs/timing_p50_p95.csv    -- per-method latency + offline timing
    results/dimacs/preprocessing_breakdown.csv -- offline timing breakdown only
  results/dimacs/wilcoxon_pvalues.csv  -- statistical significance tests
  results/dimacs/per_query_paired.csv  -- raw paired data for reproducibility
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make src/ importable so `experiments` resolves to src/experiments/.
_project_root = str(Path(__file__).resolve().parent.parent)
_src_dir = str(Path(_project_root) / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration (matches aac.yaml)
# ---------------------------------------------------------------------------
GRAPHS = ["NY", "BAY", "COL", "FLA"]
DATA_DIR = "data/dimacs"
RESULTS_DIR = "results/dimacs"

# AAC parameters
K0 = 64
M = 16

# Training parameters
TRAIN_NUM_EPOCHS = 200
TRAIN_BATCH_SIZE = 256
TRAIN_LR = 1e-3
TRAIN_COND_LAMBDA = 0.01
TRAIN_T_INIT = 1.0
TRAIN_GAMMA = 1.05

# ALT parameters at matched deployed label memory (64 B/v on directed
# DIMACS graphs: AAC stores m=16 floats/vertex = 64 B; ALT stores 2*K
# floats/vertex = 64 B at K=8). The earlier value K=16 corresponds to
# 128 B/v, i.e. a 2x memory advantage for ALT -- not matched memory and
# inconsistent with `paper/table_multi_axis_cost.tex` (B=64 B/v: AAC
# K0=64,m=16; ALT K=8) and Section 5.2 of the paper.
ALT_NUM_LANDMARKS = 8

# Query parameters
NUM_QUERIES = 100
SEED = 42

# ---------------------------------------------------------------------------
# Imports from the AAC library
# ---------------------------------------------------------------------------
from aac.graphs.loaders.dimacs import load_dimacs
from aac.search.astar import astar

# For AAC preprocessing (inline, no Hydra dependency)
import scipy.sparse.csgraph

from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.convert import graph_to_scipy
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.metrics.collector import PreprocessingMetrics

# For ALT preprocessing
from aac.baselines.alt import alt_preprocess, make_alt_heuristic

# For queries
from experiments.utils import generate_queries


def preprocess_aac_standalone(graph, K0: int, m: int) -> tuple:
    """AAC preprocessing: FPS -> SSSP -> LinearCompressor -> train.

    Returns:
        (heuristic_fn, preprocess_metrics)
    """
    # Find LCC for anchor selection
    scipy_csr = graph_to_scipy(graph)
    _, labels = scipy.sparse.csgraph.connected_components(scipy_csr, directed=False)
    component_sizes = np.bincount(labels)
    largest_component = int(np.argmax(component_sizes))
    lcc_nodes = np.where(labels == largest_component)[0]
    lcc_seed = int(lcc_nodes[0])

    t0 = time.perf_counter()

    # Anchor selection via FPS
    anchors = farthest_point_sampling(
        graph, K0, seed_vertex=lcc_seed,
        valid_vertices=torch.tensor(lcc_nodes, dtype=torch.int64),
    )
    t1 = time.perf_counter()

    # SSSP from anchors
    teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)
    t2 = time.perf_counter()

    # LinearCompressor training
    compressor = LinearCompressor(K=K0, m=m, is_directed=graph.is_directed)
    train_cfg = TrainConfig(
        num_epochs=TRAIN_NUM_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        lr=TRAIN_LR,
        cond_lambda=TRAIN_COND_LAMBDA,
        T_init=TRAIN_T_INIT,
        gamma=TRAIN_GAMMA,
    )
    lcc_vertices = torch.tensor(lcc_nodes, dtype=torch.int64)
    train_linear_compressor(compressor, teacher_labels, train_cfg, valid_vertices=lcc_vertices)

    # Build heuristic from compressed labels
    d_out_t = teacher_labels.d_out.t()  # (V, K)
    d_in_t = teacher_labels.d_in.t()    # (V, K)
    compressor.eval()
    with torch.no_grad():
        if graph.is_directed:
            y_fwd, y_bwd = compressor(d_out_t, d_in_t)
            y_fwd, y_bwd = y_fwd.detach(), y_bwd.detach()
        else:
            y = compressor(d_out_t)
            y_fwd = y_bwd = y.detach()
    heuristic = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)

    t3 = time.perf_counter()
    preprocess_metrics = PreprocessingMetrics(
        anchor_selection_sec=t1 - t0,
        sssp_sec=t2 - t1,
        training_sec=t3 - t2,
        total_sec=t3 - t0,
    )
    return heuristic, preprocess_metrics


def preprocess_alt_standalone(graph, num_landmarks: int) -> tuple:
    """ALT preprocessing: FPS -> SSSP -> heuristic.

    Returns:
        (heuristic_fn, preprocess_metrics)
    """
    # Match AAC by restricting landmark selection to the same largest component.
    scipy_csr = graph_to_scipy(graph)
    _, labels = scipy.sparse.csgraph.connected_components(scipy_csr, directed=False)
    component_sizes = np.bincount(labels)
    largest_component = int(np.argmax(component_sizes))
    lcc_nodes = np.where(labels == largest_component)[0]
    lcc_seed = int(lcc_nodes[0])
    lcc_vertices = torch.tensor(lcc_nodes, dtype=torch.int64)

    t0 = time.perf_counter()
    anchors = farthest_point_sampling(
        graph,
        num_landmarks,
        seed_vertex=lcc_seed,
        valid_vertices=lcc_vertices,
    )
    t1 = time.perf_counter()
    teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)
    heuristic = make_alt_heuristic(teacher_labels)
    t2 = time.perf_counter()
    preprocess_metrics = PreprocessingMetrics(
        anchor_selection_sec=t1 - t0,
        sssp_sec=t2 - t1,
        training_sec=0.0,
        total_sec=t2 - t0,
    )
    return heuristic, preprocess_metrics


def run_queries(graph, heuristic, queries: list[tuple[int, int]]) -> tuple:
    """Run A* queries and collect per-query expansions and latency.

    Returns:
        (expansions_list, latency_ms_list)
    """
    expansions_list = []
    latency_ms_list = []

    for s, t in queries:
        start = time.perf_counter()
        result = astar(graph, s, t, heuristic=heuristic)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        expansions_list.append(result.expansions)
        latency_ms_list.append(elapsed_ms)

    return expansions_list, latency_ms_list


def main() -> None:
    """Run timing benchmarks and Wilcoxon tests on all DIMACS graphs."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Accumulators for CSV output
    timing_rows = []       # for timing_p50_p95.csv
    wilcoxon_rows = []     # for wilcoxon_pvalues.csv
    paired_rows = []       # for per_query_paired.csv

    print(f"{'='*70}")
    print("AAC vs ALT Timing & Statistical Significance Benchmarks")
    print(f"{'='*70}")
    print(f"AAC: K0={K0}, m={M} | ALT: {ALT_NUM_LANDMARKS} landmarks")
    print(f"Queries: {NUM_QUERIES} per graph | Seed: {SEED}")
    print(f"{'='*70}\n")

    for graph_name in GRAPHS:
        gr_path = os.path.join(DATA_DIR, f"USA-road-d.{graph_name}.gr")
        co_path = os.path.join(DATA_DIR, f"USA-road-d.{graph_name}.co")

        if not os.path.exists(gr_path) or not os.path.exists(co_path):
            print(f"WARNING: Data files for {graph_name} not found, skipping.")
            continue

        print(f"\n{'─'*70}")
        print(f"Graph: {graph_name}")
        print(f"{'─'*70}")

        # Load graph
        print(f"  Loading {graph_name}...", end=" ", flush=True)
        t_load_start = time.perf_counter()
        graph = load_dimacs(gr_path, co_path)
        t_load = time.perf_counter() - t_load_start
        print(f"done ({graph.num_nodes:,} nodes, {graph.num_edges:,} edges, {t_load:.1f}s)")

        # Generate queries (same seed for both methods)
        queries = generate_queries(graph, NUM_QUERIES, seed=SEED)
        print(f"  Generated {len(queries)} queries (seed={SEED})")

        # ---- AAC preprocessing ----
        print(f"  AAC preprocessing (K0={K0}, m={M})...", end=" ", flush=True)
        aac_heuristic, aac_preprocess = preprocess_aac_standalone(graph, K0, M)
        print(
            "done "
            f"({aac_preprocess.total_sec:.1f}s total = "
            f"{aac_preprocess.anchor_selection_sec:.1f}s anchor + "
            f"{aac_preprocess.sssp_sec:.1f}s SSSP + "
            f"{aac_preprocess.training_sec:.1f}s train)"
        )

        # ---- ALT preprocessing ----
        print(f"  ALT preprocessing ({ALT_NUM_LANDMARKS} landmarks)...", end=" ", flush=True)
        alt_heuristic, alt_preprocess = preprocess_alt_standalone(graph, ALT_NUM_LANDMARKS)
        print(
            "done "
            f"({alt_preprocess.total_sec:.1f}s total = "
            f"{alt_preprocess.anchor_selection_sec:.1f}s anchor + "
            f"{alt_preprocess.sssp_sec:.1f}s SSSP)"
        )

        # ---- Run queries ----
        print(f"  Running {NUM_QUERIES} AAC queries...", end=" ", flush=True)
        aac_expansions, aac_latencies = run_queries(graph, aac_heuristic, queries)
        print(f"done (total {sum(aac_latencies)/1000:.1f}s)")

        print(f"  Running {NUM_QUERIES} ALT queries...", end=" ", flush=True)
        alt_expansions, alt_latencies = run_queries(graph, alt_heuristic, queries)
        print(f"done (total {sum(alt_latencies)/1000:.1f}s)")

        # ---- Aggregate timing ----
        aac_lat = np.array(aac_latencies)
        alt_lat = np.array(alt_latencies)
        aac_exp = np.array(aac_expansions)
        alt_exp = np.array(alt_expansions)

        # Timing rows
        for method, lat_arr, preprocess in [
            ("aac", aac_lat, aac_preprocess),
            ("alt", alt_lat, alt_preprocess),
        ]:
            timing_rows.append({
                "graph": graph_name,
                "method": method,
                "p50_ms": float(np.percentile(lat_arr, 50)),
                "p95_ms": float(np.percentile(lat_arr, 95)),
                "median_ms": float(np.median(lat_arr)),
                "mean_ms": float(np.mean(lat_arr)),
                "min_ms": float(np.min(lat_arr)),
                "max_ms": float(np.max(lat_arr)),
                "num_queries": len(lat_arr),
                "anchor_selection_sec": float(preprocess.anchor_selection_sec),
                "sssp_sec": float(preprocess.sssp_sec),
                "training_sec": float(preprocess.training_sec),
                "preprocess_total_sec": float(preprocess.total_sec),
            })

        # ---- Wilcoxon test ----
        from scipy.stats import wilcoxon

        # Test if AAC has fewer expansions than ALT
        try:
            stat_less, p_less = wilcoxon(aac_exp, alt_exp, alternative="less")
        except ValueError:
            # All differences are zero (identical results)
            stat_less, p_less = 0.0, 1.0

        try:
            stat_two, p_two = wilcoxon(aac_exp, alt_exp, alternative="two-sided")
        except ValueError:
            stat_two, p_two = 0.0, 1.0

        aac_wins = int(np.sum(aac_exp < alt_exp))
        reduction_pct = float((1.0 - np.median(aac_exp) / np.median(alt_exp)) * 100.0) if np.median(alt_exp) > 0 else 0.0

        wilcoxon_rows.append({
            "graph": graph_name,
            "aac_median_expansions": float(np.median(aac_exp)),
            "alt_median_expansions": float(np.median(alt_exp)),
            "aac_mean_expansions": float(np.mean(aac_exp)),
            "alt_mean_expansions": float(np.mean(alt_exp)),
            "reduction_pct": round(reduction_pct, 2),
            "wilcoxon_statistic": float(stat_less),
            "p_value_less": float(p_less),
            "p_value_twosided": float(p_two),
            "aac_wins": aac_wins,
        })

        # ---- Per-query paired data ----
        for i, (s, t) in enumerate(queries):
            paired_rows.append({
                "graph": graph_name,
                "query_idx": i,
                "source": s,
                "target": t,
                "aac_expansions": int(aac_exp[i]),
                "alt_expansions": int(alt_exp[i]),
                "aac_p50_ms": float(aac_lat[i]),
                "alt_p50_ms": float(alt_lat[i]),
            })

        # Print summary for this graph
        print(f"\n  Results for {graph_name}:")
        print(
            f"    AAC: offline={aac_preprocess.total_sec:.1f}s  "
            f"p50={np.percentile(aac_lat, 50):.1f}ms  "
            f"p95={np.percentile(aac_lat, 95):.1f}ms  "
            f"median_exp={np.median(aac_exp):.0f}"
        )
        print(
            f"    ALT: offline={alt_preprocess.total_sec:.1f}s  "
            f"p50={np.percentile(alt_lat, 50):.1f}ms  "
            f"p95={np.percentile(alt_lat, 95):.1f}ms  "
            f"median_exp={np.median(alt_exp):.0f}"
        )
        print(f"    Reduction: {reduction_pct:.1f}%  Wilcoxon p={p_less:.2e}  AAC wins: {aac_wins}/{len(queries)}")

    # ---- Write CSVs ----
    import csv

    # timing_p50_p95.csv
    timing_path = os.path.join(RESULTS_DIR, "timing_p50_p95.csv")
    with open(timing_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "graph", "method", "p50_ms", "p95_ms", "median_ms", "mean_ms", "min_ms", "max_ms", "num_queries",
            "anchor_selection_sec", "sssp_sec", "training_sec", "preprocess_total_sec",
        ])
        writer.writeheader()
        writer.writerows(timing_rows)
    print(f"\nWrote: {timing_path} ({len(timing_rows)} rows)")

    preprocess_path = os.path.join(RESULTS_DIR, "preprocessing_breakdown.csv")
    with open(preprocess_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "graph", "method", "anchor_selection_sec", "sssp_sec", "training_sec", "preprocess_total_sec"
        ])
        writer.writeheader()
        writer.writerows([
            {
                "graph": row["graph"],
                "method": row["method"],
                "anchor_selection_sec": row["anchor_selection_sec"],
                "sssp_sec": row["sssp_sec"],
                "training_sec": row["training_sec"],
                "preprocess_total_sec": row["preprocess_total_sec"],
            }
            for row in timing_rows
        ])
    print(f"Wrote: {preprocess_path} ({len(timing_rows)} rows)")

    # wilcoxon_pvalues.csv
    wilcoxon_path = os.path.join(RESULTS_DIR, "wilcoxon_pvalues.csv")
    with open(wilcoxon_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "graph", "aac_median_expansions", "alt_median_expansions",
            "aac_mean_expansions", "alt_mean_expansions", "reduction_pct",
            "wilcoxon_statistic", "p_value_less", "p_value_twosided", "aac_wins"
        ])
        writer.writeheader()
        writer.writerows(wilcoxon_rows)
    print(f"Wrote: {wilcoxon_path} ({len(wilcoxon_rows)} rows)")

    # per_query_paired.csv
    paired_path = os.path.join(RESULTS_DIR, "per_query_paired.csv")
    with open(paired_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "graph", "query_idx", "source", "target",
            "aac_expansions", "alt_expansions", "aac_p50_ms", "alt_p50_ms"
        ])
        writer.writeheader()
        writer.writerows(paired_rows)
    print(f"Wrote: {paired_path} ({len(paired_rows)} rows)")

    # ---- Final summary table ----
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Graph':<8} {'AAC p50':>10} {'AAC p95':>10} {'ALT p50':>10} {'ALT p95':>10} {'Red.%':>8} {'p-value':>12}")
    print(f"{'─'*70}")
    for tr in timing_rows:
        if tr["method"] == "aac":
            graph_name = tr["graph"]
            aac_p50 = tr["p50_ms"]
            aac_p95 = tr["p95_ms"]
            # Find matching ALT row
            alt_row = next(r for r in timing_rows if r["graph"] == graph_name and r["method"] == "alt")
            alt_p50 = alt_row["p50_ms"]
            alt_p95 = alt_row["p95_ms"]
            # Find Wilcoxon row
            wrow = next(r for r in wilcoxon_rows if r["graph"] == graph_name)
            print(f"{graph_name:<8} {aac_p50:>9.1f}ms {aac_p95:>9.1f}ms {alt_p50:>9.1f}ms {alt_p95:>9.1f}ms {wrow['reduction_pct']:>7.1f}% {wrow['p_value_less']:>12.2e}")
    print(f"{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
