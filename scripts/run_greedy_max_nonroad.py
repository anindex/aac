#!/usr/bin/env python
"""Greedy-Max covering oracle on non-road graphs.

The non-road matched-budget result (Table 10) shows pure AAC at m=B/4 from a
K0=4m teacher pool beats pure FPS-ALT at K=B/8 at matched memory. A skeptical
reviewer may ask: is the advantage from *learning* or from *pool size* (AAC
sees 4m FPS teachers; pure ALT sees only m/2)? The clean causal test, the
non-road analogue of the Greedy-Max column in Section 5.7 (Table 9), is:

    Greedy-Max oracle: from the same K0=4m FPS teacher pool that AAC sees,
    greedily pick the m landmarks that maximize the average ALT heuristic
    over the query set --- no gradient-based learning.

Three outcomes are possible:
  * Greedy-Max approx Pure AAC => pool-size advantage; any smart subsetter wins.
  * Greedy-Max << Pure AAC     => learning captures structure beyond coverage.
  * Greedy-Max > Pure AAC      => a non-learned oracle beats the learner.

Output: results/greedy_max_nonroad/greedy_max.csv with columns matching the
matched-budget hybrid CSV for easy cross-table joining.
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
import torch

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from experiments.utils import compute_strong_lcc, generate_queries

from scripts.run_ablation_selection import greedy_maximize_heuristic
from scripts.run_synthetic_experiments import (
    GRAPH_SEED,
    NUM_QUERIES,
    QUERY_SEED,
    generate_community_graph,
    generate_powerlaw_graph,
    nx_to_graph,
)


OUTPUT_DIR = _PROJECT_ROOT / "results" / "greedy_max_nonroad"
TOTAL_BUDGETS_BYTES = [32, 64, 128]
K0_FACTOR = 4


class TeacherSubset:
    """Lightweight wrapper holding a subset of teacher labels."""
    def __init__(self, d_out: torch.Tensor, d_in: torch.Tensor):
        self.d_out = d_out
        self.d_in = d_in


def run_graph(graph_type: str, graph) -> list[dict]:
    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    print(f"  LCC: {len(lcc_nodes):,} / {graph.num_nodes:,} nodes")

    queries = generate_queries(graph, NUM_QUERIES, seed=QUERY_SEED)

    print("  Dijkstra baseline...")
    dij_mean = float(np.mean(
        [dijkstra(graph, s, t).expansions for s, t in queries]
    ))
    print(f"  Dijkstra: mean={dij_mean:.0f}")

    # Precompute at the largest K0 once, then subset per-budget (FPS is a
    # prefix-preserving sequence, so first K0_small rows of the K0_large
    # teacher matrix equals the K0_small teacher matrix).
    K0_max = max(TOTAL_BUDGETS_BYTES) // 4 * K0_FACTOR  # = B_max for these factors
    print(f"  Precomputing K0_max={K0_max} FPS anchors + teacher SSSPs...")
    t0 = time.perf_counter()
    anchors_max = farthest_point_sampling(
        graph, K0_max, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
    )
    teacher_max = compute_teacher_labels(graph, anchors_max, use_gpu=False)
    print(f"    done in {time.perf_counter() - t0:.1f}s")

    rows: list[dict] = []
    for B in TOTAL_BUDGETS_BYTES:
        m = B // 4
        K0 = m * K0_FACTOR
        # Subset teachers to first K0 rows (FPS-prefix property).
        t_sub = TeacherSubset(teacher_max.d_out[:K0], teacher_max.d_in[:K0])
        # Pure-ALT matched-memory reference K:
        #   directed  -> K = B/8  (2K floats/vertex)
        #   undirected -> K = B/4 (K floats/vertex; d_in = d_out)
        alt_K = B // 8 if graph.is_directed else B // 4

        print(f"\n  === B={B} B/v  (K0={K0}, m={m}, alt_K={alt_K}, "
              f"is_directed={graph.is_directed}) ===")

        # Greedy-Max from the full K0-teacher pool (query-adaptive, no learning).
        t0 = time.perf_counter()
        h_greedy = greedy_maximize_heuristic(
            t_sub, m=m, queries=queries, is_directed=graph.is_directed,
        )
        greedy_exps = np.array([
            astar(graph, s, t, heuristic=h_greedy).expansions for s, t in queries
        ])
        greedy_mean = float(np.mean(greedy_exps))
        greedy_prep = time.perf_counter() - t0
        greedy_red = 100.0 * (1.0 - greedy_mean / dij_mean)
        print(f"    Greedy-Max: mean={greedy_mean:.0f}  red={greedy_red:.2f}%  [{greedy_prep:.1f}s]")

        # Pure-ALT reference (deterministic; reuse FPS from the teacher pool).
        t0 = time.perf_counter()
        alt_teacher = alt_preprocess(
            graph, alt_K, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
        )
        h_alt = make_alt_heuristic(alt_teacher)
        alt_exps = np.array([
            astar(graph, s, t, heuristic=h_alt).expansions for s, t in queries
        ])
        alt_mean = float(np.mean(alt_exps))
        alt_red = 100.0 * (1.0 - alt_mean / dij_mean)
        print(f"    Pure-ALT:   mean={alt_mean:.0f}  red={alt_red:.2f}%  [{time.perf_counter() - t0:.1f}s]")

        rows.append({
            "graph_type": graph_type,
            "total_budget_B": B,
            "K0_teacher_pool": K0,
            "m_selected": m,
            "alt_K_reference": alt_K,
            "dij_mean_exp": f"{dij_mean:.1f}",
            "greedy_max_mean_exp": f"{greedy_mean:.1f}",
            "greedy_max_reduction_pct": f"{greedy_red:.2f}",
            "pure_alt_mean_exp": f"{alt_mean:.1f}",
            "pure_alt_reduction_pct": f"{alt_red:.2f}",
            "greedy_max_prep_s": f"{greedy_prep:.2f}",
            "notes": "deterministic_given_graph_seed_and_queries",
        })
    return rows


CSV_COLUMNS = [
    "graph_type", "total_budget_B", "K0_teacher_pool", "m_selected",
    "alt_K_reference", "dij_mean_exp",
    "greedy_max_mean_exp", "greedy_max_reduction_pct",
    "pure_alt_mean_exp", "pure_alt_reduction_pct",
    "greedy_max_prep_s", "notes",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    print(f"\n{'='*70}\n  SBM COMMUNITY GRAPH (5x2000, p_in=0.05, p_out=0.001)\n{'='*70}")
    G_sbm = generate_community_graph(GRAPH_SEED)
    graph_sbm = nx_to_graph(G_sbm, weight_seed=GRAPH_SEED)
    print(f"  AAC: {graph_sbm.num_nodes:,} nodes, {graph_sbm.num_edges:,} edges")
    all_rows.extend(run_graph("community_sbm", graph_sbm))

    print(f"\n{'='*70}\n  BARABASI-ALBERT (10k, m=5)\n{'='*70}")
    G_ba = generate_powerlaw_graph(GRAPH_SEED)
    graph_ba = nx_to_graph(G_ba, weight_seed=GRAPH_SEED)
    print(f"  AAC: {graph_ba.num_nodes:,} nodes, {graph_ba.num_edges:,} edges")
    all_rows.extend(run_graph("powerlaw_ba", graph_ba))

    print(f"\n{'='*70}\n  OGB-ARXIV (~170k nodes, symmetrized LCC)\n{'='*70}")
    from scripts.run_nonroad_real import load_ogbn_arxiv
    G_arx = load_ogbn_arxiv()
    graph_arx = nx_to_graph(G_arx, weight_seed=GRAPH_SEED)
    print(f"  AAC: {graph_arx.num_nodes:,} nodes, {graph_arx.num_edges:,} edges")
    all_rows.extend(run_graph("ogbn_arxiv", graph_arx))

    out = OUTPUT_DIR / "greedy_max.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
