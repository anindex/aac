#!/usr/bin/env python
"""OSMnx experiments: AAC vs ALT vs Dijkstra on road networks at scale.

Standalone script that runs AAC and ALT on OSMnx road network graphs from
small-scale (Modena, Manhattan) to city-scale (Berlin, LA) to country-scale
(Netherlands) with multiple seeds. Uses the chunked-SSSP scalability path
(chunked SSSP, float32, NetworKit backend) for large-scale graphs.

Metrics: preprocessing time, peak memory (RSS), query latency (ms),
expansion reduction (%), admissibility verification.

Output:
  - results/osmnx/large_scale_results.csv       (per-seed)
  - results/osmnx/large_scale_results_agg.csv   (aggregated)
  - results/osmnx/osmnx_results.csv             (backward-compat, Modena/Manhattan only)

Usage:
  python scripts/run_osmnx_experiments.py                        # all 5 graphs
  python scripts/run_osmnx_experiments.py --graphs berlin la     # subset
  python scripts/run_osmnx_experiments.py --graphs modena manhattan  # small only
"""

from __future__ import annotations

import argparse
import csv
import logging
import resource
import time
from pathlib import Path

import numpy as np
import scipy.sparse.csgraph
import torch

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.convert import graph_to_scipy
from aac.graphs.io import load_graph_npz
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.metrics.admissibility import check_admissibility
from experiments.utils import generate_queries

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/osmnx")
OUTPUT_DIR = Path("results/osmnx")

# All 5 graphs: NPZ paths (produced by scripts/prepare_osmnx_graphs.py)
GRAPHS = {
    "modena": Path("data/osmnx/modena.npz"),
    "manhattan": Path("data/osmnx/manhattan.npz"),
    "berlin": Path("data/osmnx/berlin.npz"),
    "los_angeles": Path("data/osmnx/los_angeles.npz"),
    "netherlands": Path("data/osmnx/netherlands.npz"),
}

SEEDS = [42, 123, 456, 789, 1024]
NUM_QUERIES = 100

# AAC configs: test multiple K0/m combinations
HIT_CONFIGS = [
    (16, 8),   # K0=16, m=8  -> 32 bytes/v
    (32, 8),   # K0=32, m=8  -> 32 bytes/v
    (32, 16),  # K0=32, m=16 -> 64 bytes/v
    (64, 16),  # K0=64, m=16 -> 64 bytes/v
    (64, 32),  # K0=64, m=32 -> 128 bytes/v
]

# ALT configs: match AAC memory budgets
ALT_K_VALUES = [4, 8, 16, 32]

# Training config
NUM_EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3

# Per-graph configuration overrides for the chunked-SSSP path.
# Netherlands uses chunk_size=8 for bounded peak memory on 5M+ nodes.
# All graphs use float64 for SSSP to preserve admissibility guarantees
# (float32 SSSP accumulation can violate triangle inequality by >1e-6).
GRAPH_CONFIG = {
    "netherlands": {"dtype": torch.float64, "chunk_size": 8},
    # All others use defaults below
}
DEFAULT_CONFIG = {"dtype": torch.float64, "chunk_size": 8}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_peak_rss_mb() -> float:
    """Peak RSS in MB (Linux: ru_maxrss is kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def compute_lcc(graph):
    """Compute largest connected component nodes and a seed vertex in LCC.

    Uses strong CC for directed graphs to match AAC/ALT preprocessing.
    """
    scipy_csr = graph_to_scipy(graph)
    is_directed = getattr(graph, "is_directed", False)
    _, labels = scipy.sparse.csgraph.connected_components(
        scipy_csr, directed=is_directed,
        connection="strong" if is_directed else "weak",
    )
    sizes = np.bincount(labels)
    largest = int(np.argmax(sizes))
    lcc_nodes = np.where(labels == largest)[0]
    return lcc_nodes, int(lcc_nodes[0])


def run_queries_fn(graph, queries, heuristic):
    """Run A* queries, returning (expansions_list, latencies_ms_list)."""
    expansions = []
    latencies_ms = []
    for s, t in queries:
        t0 = time.perf_counter()
        result = astar(graph, s, t, heuristic=heuristic)
        t1 = time.perf_counter()
        expansions.append(result.expansions)
        latencies_ms.append((t1 - t0) * 1000)
    return expansions, latencies_ms


def run_queries_full(graph, queries, heuristic):
    """Run A* queries, returning (SearchResult_list, latencies_ms_list)."""
    results = []
    latencies_ms = []
    for s, t in queries:
        t0 = time.perf_counter()
        result = astar(graph, s, t, heuristic=heuristic)
        t1 = time.perf_counter()
        results.append(result)
        latencies_ms.append((t1 - t0) * 1000)
    return results, latencies_ms


def parse_args():
    parser = argparse.ArgumentParser(
        description="OSMnx road network experiments: AAC vs ALT vs Dijkstra at scale.",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=list(GRAPHS.keys()),
        choices=list(GRAPHS.keys()),
        help="Which graphs to run (default: all 5). Example: --graphs berlin los_angeles",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    selected_graphs = args.graphs

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    total_graphs = len(selected_graphs)
    graph_start_time = time.perf_counter()

    for graph_idx, graph_name in enumerate(selected_graphs):
        npz_path = GRAPHS[graph_name]
        if not npz_path.exists():
            logger.warning("NPZ file %s not found, skipping %s", npz_path, graph_name)
            continue

        logger.info(
            "=== Graph %d/%d: %s ===", graph_idx + 1, total_graphs, graph_name
        )

        # Load graph from NPZ (current schema, allow_pickle=False)
        graph = load_graph_npz(npz_path)
        logger.info(
            "Loaded: %s nodes, %s edges, directed=%s",
            f"{graph.num_nodes:,}", f"{graph.num_edges:,}", graph.is_directed,
        )

        # Per-graph SSSP configuration
        cfg = GRAPH_CONFIG.get(graph_name, DEFAULT_CONFIG)
        logger.info(
            "Config: dtype=%s, chunk_size=%d",
            cfg["dtype"], cfg["chunk_size"],
        )

        # LCC
        lcc_nodes, lcc_seed = compute_lcc(graph)
        logger.info("LCC: %s nodes", f"{len(lcc_nodes):,}")

        # Deterministic queries (same across seeds)
        queries = generate_queries(graph, NUM_QUERIES, seed=42)

        # Dijkstra baseline -- collect full SearchResult objects for admissibility checks
        logger.info("Running Dijkstra baseline (%d queries)...", NUM_QUERIES)
        dij_t0 = time.perf_counter()
        dij_results = [dijkstra(graph, s, t) for s, t in queries]
        dij_t1 = time.perf_counter()
        dij_costs = [r.cost for r in dij_results]
        dij_exps = [r.expansions for r in dij_results]
        dij_mean = np.mean(dij_exps)
        dij_latency = (dij_t1 - dij_t0) / NUM_QUERIES * 1000  # ms per query
        logger.info(
            "Dijkstra mean expansions: %.0f, latency: %.1f ms/query",
            dij_mean, dij_latency,
        )

        for seed_idx, seed in enumerate(SEEDS):
            logger.info(
                "--- %s seed %d (%d/%d) ---", graph_name, seed, seed_idx + 1, len(SEEDS),
            )
            torch.manual_seed(seed)
            lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)

            # --- ALT baselines ---
            for K in ALT_K_VALUES:
                try:
                    t0 = time.perf_counter()
                    teacher = alt_preprocess(
                        graph, K, seed_vertex=lcc_seed,
                        valid_vertices=lcc_tensor,
                    )
                    h = make_alt_heuristic(teacher)
                    prep = time.perf_counter() - t0
                    peak_mem = get_peak_rss_mb()

                    # Run queries with full results for admissibility
                    alt_full, alt_latencies = run_queries_full(graph, queries, h)
                    alt_exps = [r.expansions for r in alt_full]
                    mean_exp = np.mean(alt_exps)
                    mean_latency = np.mean(alt_latencies)
                    reduction = 100.0 * (1.0 - mean_exp / dij_mean) if dij_mean > 0 else 0.0

                    # Admissibility verification
                    adm = check_admissibility(alt_full, dij_costs)
                    assert adm.num_violations == 0, (
                        f"ALT admissibility violated on {graph_name} K={K}: "
                        f"{adm.num_violations} queries"
                    )

                    all_results.append({
                        "graph": graph_name,
                        "method": "ALT",
                        "K0": 0,
                        "m": 0,
                        "num_landmarks_or_dims": K,
                        "bytes_per_vertex": 2 * K * 4,
                        "mean_expansions": mean_exp,
                        "expansion_reduction_pct": reduction,
                        "preprocess_sec": prep,
                        "mean_query_latency_ms": mean_latency,
                        "peak_memory_mb": peak_mem,
                        "admissibility_violations": adm.num_violations,
                        "num_nodes": graph.num_nodes,
                        "num_edges": graph.num_edges,
                        "seed": seed,
                    })
                    logger.info(
                        "  ALT K=%d: reduction=%.1f%%, latency=%.1fms, adm_ok",
                        K, reduction, mean_latency,
                    )
                except Exception as e:
                    logger.error("  ALT K=%d: FAILED (%s)", K, e)

            # --- AAC configs ---
            for K0, m in HIT_CONFIGS:
                try:
                    t0 = time.perf_counter()
                    anchors = farthest_point_sampling(
                        graph, K0, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
                    )
                    teacher_labels = compute_teacher_labels(
                        graph, anchors, use_gpu=False,
                        chunk_size=cfg["chunk_size"],
                        dtype=cfg["dtype"],
                        backend="auto",
                    )
                    compressor = LinearCompressor(K=K0, m=m, is_directed=graph.is_directed)
                    train_cfg = TrainConfig(
                        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=seed,
                    )
                    train_linear_compressor(
                        compressor, teacher_labels, train_cfg, valid_vertices=lcc_tensor,
                    )

                    d_out_t = teacher_labels.d_out.t()
                    d_in_t = teacher_labels.d_in.t()
                    compressor.eval()
                    with torch.no_grad():
                        if graph.is_directed:
                            y_fwd, y_bwd = compressor(d_out_t, d_in_t)
                            y_fwd, y_bwd = y_fwd.detach(), y_bwd.detach()
                        else:
                            y = compressor(d_out_t)
                            y_fwd = y_bwd = y.detach()
                    h = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)
                    prep = time.perf_counter() - t0
                    peak_mem = get_peak_rss_mb()

                    # Run queries with full results for admissibility
                    aac_full, aac_latencies = run_queries_full(graph, queries, h)
                    aac_exps = [r.expansions for r in aac_full]
                    mean_exp = np.mean(aac_exps)
                    mean_latency = np.mean(aac_latencies)
                    reduction = 100.0 * (1.0 - mean_exp / dij_mean) if dij_mean > 0 else 0.0

                    # Admissibility verification (EXPR-03)
                    adm = check_admissibility(aac_full, dij_costs)
                    assert adm.num_violations == 0, (
                        f"AAC admissibility violated on {graph_name} K0={K0} m={m}: "
                        f"{adm.num_violations} queries"
                    )

                    all_results.append({
                        "graph": graph_name,
                        "method": "AAC",
                        "K0": K0,
                        "m": m,
                        "num_landmarks_or_dims": m,
                        "bytes_per_vertex": m * 4,
                        "mean_expansions": mean_exp,
                        "expansion_reduction_pct": reduction,
                        "preprocess_sec": prep,
                        "mean_query_latency_ms": mean_latency,
                        "peak_memory_mb": peak_mem,
                        "admissibility_violations": adm.num_violations,
                        "num_nodes": graph.num_nodes,
                        "num_edges": graph.num_edges,
                        "seed": seed,
                    })
                    logger.info(
                        "  AAC K0=%d m=%d: reduction=%.1f%%, latency=%.1fms, adm_ok",
                        K0, m, reduction, mean_latency,
                    )
                except Exception as e:
                    logger.error("  AAC K0=%d m=%d: FAILED (%s)", K0, m, e)

        # Estimated time remaining
        elapsed = time.perf_counter() - graph_start_time
        avg_per_graph = elapsed / (graph_idx + 1)
        remaining = avg_per_graph * (total_graphs - graph_idx - 1)
        logger.info(
            "Graph %d/%d done. Elapsed: %.0fs, Est. remaining: %.0fs",
            graph_idx + 1, total_graphs, elapsed, remaining,
        )

    # -----------------------------------------------------------------------
    # Write results
    # -----------------------------------------------------------------------
    if all_results:
        # Per-seed CSV (all graphs)
        csv_path = OUTPUT_DIR / "large_scale_results.csv"
        cols = [
            "graph", "method", "K0", "m", "num_landmarks_or_dims",
            "bytes_per_vertex", "mean_expansions", "expansion_reduction_pct",
            "preprocess_sec", "mean_query_latency_ms", "peak_memory_mb",
            "admissibility_violations", "num_nodes", "num_edges", "seed",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        logger.info("Per-seed results: %s", csv_path)

        # Backward-compatible CSV (Modena/Manhattan only)
        compat_results = [r for r in all_results if r["graph"] in ("modena", "manhattan")]
        if compat_results:
            compat_path = OUTPUT_DIR / "osmnx_results.csv"
            compat_cols = [
                "city", "method", "K0", "m", "num_landmarks_or_dims",
                "bytes_per_vertex", "mean_expansions", "expansion_reduction_pct",
                "preprocess_sec", "seed",
            ]
            # Remap 'graph' -> 'city' for backward compat
            compat_rows = []
            for r in compat_results:
                row = dict(r)
                row["city"] = row.pop("graph")
                compat_rows.append(row)
            with open(compat_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=compat_cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(compat_rows)
            logger.info("Backward-compatible results: %s", compat_path)

        # Aggregated CSV
        import pandas as pd

        df = pd.DataFrame(all_results)
        group_cols = [
            "graph", "method", "K0", "m", "num_landmarks_or_dims", "bytes_per_vertex",
        ]
        agg = df.groupby(group_cols, as_index=False).agg(
            reduction_mean=("expansion_reduction_pct", "mean"),
            reduction_std=("expansion_reduction_pct", "std"),
            expansions_mean=("mean_expansions", "mean"),
            latency_mean_ms=("mean_query_latency_ms", "mean"),
            peak_memory_mb_max=("peak_memory_mb", "max"),
            n_seeds=("seed", "count"),
        )
        agg_path = OUTPUT_DIR / "large_scale_results_agg.csv"
        agg.to_csv(agg_path, index=False, float_format="%.2f")
        logger.info("Aggregated results: %s", agg_path)

        # Print summary
        print(f"\n{'='*80}")
        print("  Summary (mean +/- std across seeds):")
        for graph_name in agg["graph"].unique():
            print(f"\n  {graph_name}:")
            g_data = agg[agg["graph"] == graph_name].sort_values("bytes_per_vertex")
            for _, row in g_data.iterrows():
                if row["method"] == "AAC":
                    cfg_str = f"K0={int(row['K0'])},m={int(row['m'])}"
                else:
                    cfg_str = f"K={int(row['num_landmarks_or_dims'])}"
                std_str = (
                    f"+/-{row['reduction_std']:.1f}"
                    if not np.isnan(row["reduction_std"])
                    else ""
                )
                print(
                    f"    {row['method']:>4} {cfg_str:<16} "
                    f"{int(row['bytes_per_vertex']):>4}B/v: "
                    f"{row['reduction_mean']:.1f}%{std_str}"
                )

        print(f"\nResults: {csv_path}, {agg_path}")


if __name__ == "__main__":
    main()
