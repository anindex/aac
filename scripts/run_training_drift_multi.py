#!/usr/bin/env python
"""P2.1: Multi-graph, multi-budget, multi-seed forced-first-m drift experiment.

For each (graph, budget, seed) we:
  - Build the FPS pool of K0 landmarks (deterministic given seed).
  - Compute teacher labels for that pool.
  - Establish two reference lines:
      * forced-first-m baseline (FPS-from-pool reference, no training).
      * pure-ALT matched-memory baseline (independent FPS at K=m landmarks).
  - For each epoch checkpoint c in CHECKPOINTS, train a fresh compressor for
    c epochs and record A* expansion reduction over a fixed query set.

Outputs a tidy CSV per (graph, budget) at
  results/training_drift_multi/drift_<graph>_B<budget>.csv
with columns: graph, budget, seed, K0, m, epochs, mean_expansions,
  expansion_reduction_pct, dijkstra_mean, alt_ref_pct, forced_first_m_pct.

Scoping: per a 2 h wall-clock guard, only SBM and BA are included here. An
OGB-arXiv extension is documented as future work.
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

from run_synthetic_experiments import (
    generate_community_graph,
    generate_powerlaw_graph,
    nx_to_graph,
)

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

GRAPH_GENERATORS = {
    "sbm": generate_community_graph,
    "ba": generate_powerlaw_graph,
}


def make_graph(name: str, seed: int):
    G_nx = GRAPH_GENERATORS[name](seed=seed)
    return nx_to_graph(G_nx, weight_seed=seed)


def eval_heuristic(graph, queries, h, dij_mean: float) -> tuple[float, float]:
    exps = np.array(
        [astar(graph, s, t, heuristic=h).expansions for s, t in queries]
    )
    mean = float(exps.mean())
    red = 100.0 * (1.0 - mean / dij_mean)
    return mean, red


def forced_first_m_heuristic(teacher, m: int, is_directed: bool):
    """Build a AAC heuristic whose hard argmax selects the first m pool indices."""
    K = teacher.d_out.shape[0]
    comp = LinearCompressor(K=K, m=m, is_directed=is_directed)
    with torch.no_grad():
        W = torch.full((m, K), -10.0, dtype=comp.W.dtype)
        for i in range(m):
            W[i, i] = 10.0
        comp.W.copy_(W)
    comp.eval()
    d_out_t = teacher.d_out.t()
    with torch.no_grad():
        y = comp(d_out_t)
    return make_linear_heuristic(y, y, is_directed)


def trained_heuristic(teacher, m: int, is_directed: bool, epochs: int,
                      seed: int, lcc_tensor):
    K = teacher.d_out.shape[0]
    torch.manual_seed(seed)
    comp = LinearCompressor(K=K, m=m, is_directed=is_directed)
    cfg = TrainConfig(num_epochs=epochs, batch_size=256, lr=1e-3, seed=seed)
    if epochs > 0:
        train_linear_compressor(comp, teacher, cfg, valid_vertices=lcc_tensor)
    comp.eval()
    d_out_t = teacher.d_out.t()
    with torch.no_grad():
        y = comp(d_out_t)
    return make_linear_heuristic(y, y, is_directed)


def run_cell(graph_name: str, graph_seed: int, budget: int, K0: int, m: int,
             seeds: Iterable[int], checkpoints: Iterable[int],
             num_queries: int, out_path: Path) -> None:
    print(f"\n=== {graph_name.upper()} | B={budget} B/v | K0={K0} m={m} ===")
    t_start = time.perf_counter()
    graph = make_graph(graph_name, seed=graph_seed)
    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    queries = generate_queries(graph, num_queries, seed=graph_seed)
    dij_exps = np.array(
        [dijkstra(graph, s, t).expansions for s, t in queries]
    )
    dij_mean = float(dij_exps.mean())
    print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
          f"directed={graph.is_directed}")
    print(f"  LCC: {len(lcc_nodes)}, Dijkstra mean: {dij_mean:.1f} expansions")

    # Pure-ALT matched-memory reference (K = m on undirected).
    alt_teacher = alt_preprocess(graph, m, seed_vertex=lcc_seed,
                                 valid_vertices=lcc_tensor)
    alt_h = make_alt_heuristic(alt_teacher)
    alt_mean, alt_red = eval_heuristic(graph, queries, alt_h, dij_mean)
    print(f"  ALT K={m} reference: {alt_red:.2f}%  ({alt_mean:.1f} exp)")

    rows: list[dict] = []
    for seed in seeds:
        print(f"\n  -- seed {seed} --")
        # Build the K0 pool (deterministic given lcc_seed).
        pool = farthest_point_sampling(
            graph, K0, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
        )
        teacher = compute_teacher_labels(graph, pool, use_gpu=False)

        # Forced-first-m reference (FPS-from-pool, no training).
        h_forced = forced_first_m_heuristic(teacher, m, graph.is_directed)
        forced_mean, forced_red = eval_heuristic(
            graph, queries, h_forced, dij_mean,
        )
        print(f"     forced-first-{m}: {forced_red:.2f}% ({forced_mean:.1f})")

        for ckpt in checkpoints:
            t0 = time.perf_counter()
            h = trained_heuristic(
                teacher, m, graph.is_directed, ckpt, seed, lcc_tensor,
            )
            mean, red = eval_heuristic(graph, queries, h, dij_mean)
            elapsed = time.perf_counter() - t0
            print(f"     epochs={ckpt:>4d}: {red:.2f}% ({mean:.1f})  "
                  f"[{elapsed:.1f}s]")
            rows.append(dict(
                graph=graph_name,
                budget=budget,
                K0=K0,
                m=m,
                seed=seed,
                epochs=ckpt,
                mean_expansions=mean,
                expansion_reduction_pct=red,
                dijkstra_mean=dij_mean,
                alt_ref_pct=alt_red,
                forced_first_m_pct=forced_red,
            ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    elapsed = time.perf_counter() - t_start
    print(f"\n  Wrote {out_path} ({len(rows)} rows) in {elapsed:.1f}s.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+",
                    default=["sbm", "ba"], choices=list(GRAPH_GENERATORS))
    ap.add_argument("--budgets", nargs="+", type=int,
                    default=[32, 64, 128])
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--checkpoints", nargs="+", type=int,
                    default=[0, 50, 200, 500, 1000])
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--graph-seed", type=int, default=42,
                    help="Seed for graph generation (fixed across runs).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/training_drift_multi"))
    args = ap.parse_args()

    # Budget B B/v on undirected = K0 = 4*m, m = B/4. ALT K = m at matched mem.
    budget_to_km = {b: (4 * (b // 4), b // 4) for b in args.budgets}

    t_start = time.perf_counter()
    for graph_name in args.graphs:
        for budget in args.budgets:
            K0, m = budget_to_km[budget]
            out_path = args.out_dir / f"drift_{graph_name}_B{budget}.csv"
            run_cell(
                graph_name=graph_name,
                graph_seed=args.graph_seed,
                budget=budget,
                K0=K0,
                m=m,
                seeds=args.seeds,
                checkpoints=args.checkpoints,
                num_queries=args.num_queries,
                out_path=out_path,
            )
    total = time.perf_counter() - t_start
    print(f"\n\nALL DONE in {total/60:.1f} min "
          f"({len(args.graphs)*len(args.budgets)} cells).")


if __name__ == "__main__":
    main()
