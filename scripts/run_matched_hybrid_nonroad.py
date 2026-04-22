#!/usr/bin/env python
"""Matched-TOTAL-budget hybrid comparison on non-road graphs (matched-budget evaluation).

The existing ``run_synthetic_experiments.py`` / ``run_nonroad_real.py`` pipelines
report AAC, ALT, and Hybrid at each nominal ``budget_bpv`` but the hybrid column
consumes *both* AAC(m) and ALT(K) memory simultaneously (i.e. roughly 2x the
nominal single-method budget). Reviewers asked for a strict matched-TOTAL
budget comparison: at a fixed per-vertex memory B, which arm wins?

Three arms are compared at each total budget B in bytes per vertex (assuming
fp32 labels and the paper's ALT=2*K floats / AAC=m floats accounting):

    Pure AAC   : m = B / 4                           (all B for AAC)
    Pure ALT   : K = B / 8                           (all B for ALT)
    Hybrid 1/2 : m = B / 8,  K = B / 16              (B/2 for each)

Graphs:
    * SBM  (5 x 2000 nodes, p_in=0.05, p_out=0.001)
    * BA   (10k nodes, m=5 preferential attachment)
    * OGB-arXiv  (~170k node citation LCC, symmetrized, uniform[1,10] weights)

Budgets: 32, 64, 128 B/v. Seeds: 5. Queries: 100 uniform-random per graph.

Output:
    results/hybrid_nonroad/matched_budget_hybrid.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import scipy.stats
import torch
from scripts.run_synthetic_experiments import (
    GRAPH_SEED,
    NUM_QUERIES,
    QUERY_SEED,
    SEEDS,
    generate_community_graph,
    generate_powerlaw_graph,
    nx_to_graph,
)

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.heuristics import make_hybrid_heuristic
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

OUTPUT_DIR = _PROJECT_ROOT / "results" / "hybrid_nonroad"

# Total per-vertex memory in bytes (fp32). Three arms compared at each budget.
TOTAL_BUDGETS_BYTES = [32, 64, 128]

# K0 over-sampling factor for AAC anchors (matches synthetic defaults: K0 = 4*m).
K0_FACTOR = 4


def matched_configs(B: int, is_directed: bool) -> dict:
    """Return the three matched arms at a fixed total budget B bytes/vertex.

    Accounting:
        AAC  stores m*dtype_size bytes/vertex (both directed and undirected).
        ALT  stores 2*K*dtype_size bytes/vertex on directed graphs
             and     K*dtype_size bytes/vertex on undirected graphs,
             because compute_teacher_labels sets d_in = d_out when
             graph.is_directed is False (see aac/embeddings/sssp.py).

    Matched rule:
        Directed:   AAC m = B/4,  ALT K = B/8,  Hybrid m = B/8, K = B/16
        Undirected: AAC m = B/4,  ALT K = B/4,  Hybrid m = B/8, K = B/8

    This fixes the original (directed-only) rule silently applied to SBM/BA/
    OGB-arXiv, which gave ALT a 2x memory disadvantage on undirected graphs.
    """
    if is_directed:
        alt_full_K = B // 8
        hybrid_K = B // 16
    else:
        alt_full_K = B // 4
        hybrid_K = B // 8
    return {
        "pure_aac":    {"m": B // 4, "K": 0,          "K0": (B // 4) * K0_FACTOR},
        "pure_alt":    {"m": 0,      "K": alt_full_K, "K0": 0},
        "hybrid_half": {"m": B // 8, "K": hybrid_K,   "K0": (B // 8) * K0_FACTOR},
    }


# ---------------------------------------------------------------------------
# Heuristic builders
# ---------------------------------------------------------------------------
def build_aac(graph, lcc_tensor, lcc_seed, K0: int, m: int, seed: int):
    """Train a AAC linear compressor and return a heuristic callable + prep time."""
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    anchors = farthest_point_sampling(
        graph, K0, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
    )
    teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)
    compressor = LinearCompressor(K=K0, m=m, is_directed=graph.is_directed)
    cfg = TrainConfig(num_epochs=200, batch_size=256, lr=1e-3, seed=seed)
    train_linear_compressor(
        compressor, teacher_labels, cfg, valid_vertices=lcc_tensor,
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
    return h, time.perf_counter() - t0


def build_alt(graph, lcc_tensor, lcc_seed, K: int):
    """Build the ALT heuristic and return a callable + prep time."""
    t0 = time.perf_counter()
    teacher_alt = alt_preprocess(
        graph, K, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
    )
    h = make_alt_heuristic(teacher_alt)
    return h, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Per-graph experiment
# ---------------------------------------------------------------------------
def run_graph(graph_type: str, graph) -> list[dict]:
    """Run all (budget, arm, seed) combinations on a single graph."""
    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    print(f"  LCC: {len(lcc_nodes):,} / {graph.num_nodes:,} nodes")

    queries = generate_queries(graph, NUM_QUERIES, seed=QUERY_SEED)

    print("  Dijkstra baseline...")
    dij_exps = np.array([dijkstra(graph, s, t).expansions for s, t in queries])
    dij_mean = float(np.mean(dij_exps))
    print(f"  Dijkstra: mean={dij_mean:.0f} expansions")

    rows: list[dict] = []

    for B in TOTAL_BUDGETS_BYTES:
        cfgs = matched_configs(B, is_directed=graph.is_directed)
        print(f"\n  === Total budget B={B} B/v (is_directed={graph.is_directed}) ===")
        print(f"    pure_aac:    m={cfgs['pure_aac']['m']}, K0={cfgs['pure_aac']['K0']}")
        print(f"    pure_alt:    K={cfgs['pure_alt']['K']}")
        print(f"    hybrid_half: m={cfgs['hybrid_half']['m']}, K0={cfgs['hybrid_half']['K0']}, K={cfgs['hybrid_half']['K']}")

        # --- Pure ALT at full budget (deterministic, compute once) ---
        alt_K_full = cfgs["pure_alt"]["K"]
        alt_h_full, alt_prep = build_alt(graph, lcc_tensor, lcc_seed, alt_K_full)
        alt_exps_full = np.array([
            astar(graph, s, t, heuristic=alt_h_full).expansions for s, t in queries
        ])
        alt_mean_full = float(np.mean(alt_exps_full))
        print(f"    pure_alt: mean={alt_mean_full:.0f}  [{alt_prep:.1f}s]")

        # --- Pure ALT at HALF budget (reused inside hybrid, deterministic) ---
        alt_K_half = cfgs["hybrid_half"]["K"]
        alt_h_half, _ = build_alt(graph, lcc_tensor, lcc_seed, alt_K_half)

        # --- Loop over seeds for AAC-based arms ---
        for seed in SEEDS:
            # Pure AAC at full budget
            aac_m_full = cfgs["pure_aac"]["m"]
            aac_K0_full = cfgs["pure_aac"]["K0"]
            aac_h_full, aac_prep_full = build_aac(
                graph, lcc_tensor, lcc_seed, aac_K0_full, aac_m_full, seed,
            )
            aac_exps_full = np.array([
                astar(graph, s, t, heuristic=aac_h_full).expansions for s, t in queries
            ])
            aac_mean_full = float(np.mean(aac_exps_full))

            # Hybrid: AAC at half + ALT at half, combined via max
            aac_m_half = cfgs["hybrid_half"]["m"]
            aac_K0_half = cfgs["hybrid_half"]["K0"]
            aac_h_half, aac_prep_half = build_aac(
                graph, lcc_tensor, lcc_seed, aac_K0_half, aac_m_half, seed,
            )
            hybrid_h = make_hybrid_heuristic(aac_h_half, alt_h_half)
            hyb_exps = np.array([
                astar(graph, s, t, heuristic=hybrid_h).expansions for s, t in queries
            ])
            hyb_mean = float(np.mean(hyb_exps))

            # Paired Wilcoxon: best single-method vs hybrid (per query)
            best_single = np.minimum(aac_exps_full, alt_exps_full)
            diff = hyb_exps - best_single
            try:
                stat, p_two = scipy.stats.wilcoxon(diff, alternative="two-sided")
            except ValueError:
                stat, p_two = 0.0, 1.0

            best_single_mean = float(np.mean(best_single))
            print(
                f"    seed={seed}: AAC={aac_mean_full:.0f} "
                f"ALT={alt_mean_full:.0f} Hyb={hyb_mean:.0f} "
                f"min(AAC,ALT)={best_single_mean:.0f} "
                f"p(Hyb vs min)={p_two:.2e}"
            )

            rows.append({
                "graph_type": graph_type,
                "total_budget_B": B,
                "seed": seed,
                "pure_aac_m": aac_m_full,
                "pure_aac_K0": aac_K0_full,
                "pure_alt_K": alt_K_full,
                "hybrid_aac_m": aac_m_half,
                "hybrid_aac_K0": aac_K0_half,
                "hybrid_alt_K": alt_K_half,
                "dij_mean_exp": f"{dij_mean:.1f}",
                "pure_aac_mean_exp": f"{aac_mean_full:.1f}",
                "pure_alt_mean_exp": f"{alt_mean_full:.1f}",
                "hybrid_half_mean_exp": f"{hyb_mean:.1f}",
                "min_single_mean_exp": f"{best_single_mean:.1f}",
                "pure_aac_reduction_pct": f"{100.0 * (1.0 - aac_mean_full/dij_mean):.2f}",
                "pure_alt_reduction_pct": f"{100.0 * (1.0 - alt_mean_full/dij_mean):.2f}",
                "hybrid_half_reduction_pct": f"{100.0 * (1.0 - hyb_mean/dij_mean):.2f}",
                "wilcoxon_hyb_vs_minsingle_stat": f"{stat:.1f}",
                "wilcoxon_hyb_vs_minsingle_p": f"{p_two:.6e}",
                "aac_prep_time_s": f"{aac_prep_full:.2f}",
                "hybrid_aac_prep_time_s": f"{aac_prep_half:.2f}",
                "alt_prep_time_s": f"{alt_prep:.2f}",
            })

    return rows


CSV_COLUMNS = [
    "graph_type", "total_budget_B", "seed",
    "pure_aac_m", "pure_aac_K0", "pure_alt_K",
    "hybrid_aac_m", "hybrid_aac_K0", "hybrid_alt_K",
    "dij_mean_exp", "pure_aac_mean_exp", "pure_alt_mean_exp",
    "hybrid_half_mean_exp", "min_single_mean_exp",
    "pure_aac_reduction_pct", "pure_alt_reduction_pct", "hybrid_half_reduction_pct",
    "wilcoxon_hyb_vs_minsingle_stat", "wilcoxon_hyb_vs_minsingle_p",
    "aac_prep_time_s", "hybrid_aac_prep_time_s", "alt_prep_time_s",
]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    # --- SBM community graph ---
    print(f"\n{'='*70}\n  SBM COMMUNITY GRAPH (5x2000, p_in=0.05, p_out=0.001)\n{'='*70}")
    G_sbm = generate_community_graph(GRAPH_SEED)
    graph_sbm = nx_to_graph(G_sbm, weight_seed=GRAPH_SEED)
    print(f"  AAC: {graph_sbm.num_nodes:,} nodes, {graph_sbm.num_edges:,} edges")
    all_rows.extend(run_graph("community_sbm", graph_sbm))

    # --- Barabasi-Albert power-law graph ---
    print(f"\n{'='*70}\n  BARABASI-ALBERT (10k, m=5)\n{'='*70}")
    G_ba = generate_powerlaw_graph(GRAPH_SEED)
    graph_ba = nx_to_graph(G_ba, weight_seed=GRAPH_SEED)
    print(f"  AAC: {graph_ba.num_nodes:,} nodes, {graph_ba.num_edges:,} edges")
    all_rows.extend(run_graph("powerlaw_ba", graph_ba))

    # --- OGB-arXiv citation graph ---
    print(f"\n{'='*70}\n  OGB-ARXIV (~170k nodes, symmetrized LCC)\n{'='*70}")
    from scripts.run_nonroad_real import load_ogbn_arxiv
    G_arx = load_ogbn_arxiv()
    graph_arx = nx_to_graph(G_arx, weight_seed=GRAPH_SEED)
    print(f"  AAC: {graph_arx.num_nodes:,} nodes, {graph_arx.num_edges:,} edges")
    all_rows.extend(run_graph("ogbn_arxiv", graph_arx))

    write_csv(all_rows, OUTPUT_DIR / "matched_budget_hybrid.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
