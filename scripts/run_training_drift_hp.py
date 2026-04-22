#!/usr/bin/env python
"""Hyperparameter and initialization ablation for the training-drift cell.

Sweeps learning rate x batch size x K0 (and an extra "identity-on-first-m"
initialization arm) on the SBM B=32 cell where the training-drift diagnostic
is largest. The point of the sweep is to verify that the gap-to-teacher-
gradient drift documented in Section 5.10 is *not* an artifact of the
default training schedule:

  - lr        in {3e-4, 1e-3, 3e-3}
  - batch     in {128, 256, 512}
  - K0        in {32, 64}
  - init      in {"block_sparse" (default), "identity_first_m"}

For each combination we train for 200 epochs and report the test-set expansion
reduction; we always also report the forced-first-m baseline (the architectural
ceiling that the trained run is supposed to hit).

Output: results/training_drift_hp/sbm_b32_<combo>.csv (rolled up into one CSV).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

from run_synthetic_experiments import generate_community_graph, nx_to_graph

OUTPUT_DIR = _ROOT / "results" / "training_drift_hp"
NUM_QUERIES = 100
QUERY_SEED = 42
EPOCHS = 200


def _identity_first_m_init(K: int, m: int) -> torch.Tensor:
    """Initialization with one-hot rows on the first m pool indices."""
    W = torch.full((m, K), -10.0)
    for i in range(m):
        W[i, i] = 10.0
    return W + 0.001 * torch.randn(m, K)


def _eval(graph, queries, h, dij_mean):
    exps = np.array([astar(graph, s, t, heuristic=h).expansions for s, t in queries])
    mean = float(exps.mean())
    return mean, 100.0 * (1.0 - mean / dij_mean)


def _make_compressor(K: int, m: int, init: str) -> LinearCompressor:
    comp = LinearCompressor(K=K, m=m, is_directed=False)
    if init == "identity_first_m":
        with torch.no_grad():
            comp.W.copy_(_identity_first_m_init(K, m))
    return comp


def _forced_first_m_eval(teacher, m: int, graph, queries, dij_mean) -> tuple[float, float]:
    K = teacher.d_out.shape[0]
    comp = LinearCompressor(K=K, m=m, is_directed=False)
    with torch.no_grad():
        W = torch.full((m, K), -10.0, dtype=comp.W.dtype)
        for i in range(m):
            W[i, i] = 10.0
        comp.W.copy_(W)
    comp.eval()
    with torch.no_grad():
        y = comp(teacher.d_out.t())
    h = make_linear_heuristic(y, y, is_directed=False)
    return _eval(graph, queries, h, dij_mean)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--batches", nargs="+", type=int, default=[128, 256, 512])
    ap.add_argument("--K0s", nargs="+", type=int, default=[32, 64])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--inits", nargs="+", default=["block_sparse", "identity_first_m"])
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "sbm_b32_results.csv")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    G_nx = generate_community_graph(seed=42)
    graph = nx_to_graph(G_nx, weight_seed=42)
    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    queries = generate_queries(graph, args.num_queries, seed=QUERY_SEED)
    dij_exps = np.array([dijkstra(graph, s, t).expansions for s, t in queries])
    dij_mean = float(dij_exps.mean())
    print(f"SBM: V={graph.num_nodes:,} LCC={len(lcc_nodes):,}  "
          f"Dij mean={dij_mean:.0f}", flush=True)

    rows: list[dict] = []
    m = 8  # B=32 B/v on undirected = m = B/4
    t_start = time.perf_counter()
    for K0 in args.K0s:
        pool = farthest_point_sampling(graph, K0, seed_vertex=lcc_seed,
                                       valid_vertices=lcc_tensor)
        teacher = compute_teacher_labels(graph, pool, use_gpu=False)
        forced_mean, forced_red = _forced_first_m_eval(
            teacher, m, graph, queries, dij_mean,
        )
        print(f"\nK0={K0}: forced-first-{m} = {forced_red:.2f}% ({forced_mean:.0f})",
              flush=True)
        for lr in args.lrs:
            for batch in args.batches:
                for init in args.inits:
                    for seed in args.seeds:
                        torch.manual_seed(seed)
                        comp = _make_compressor(teacher.d_out.shape[0], m, init)
                        cfg = TrainConfig(
                            num_epochs=args.epochs, batch_size=batch, lr=lr, seed=seed,
                        )
                        t0 = time.perf_counter()
                        train_linear_compressor(comp, teacher, cfg,
                                                valid_vertices=lcc_tensor)
                        train_s = time.perf_counter() - t0
                        comp.eval()
                        with torch.no_grad():
                            y = comp(teacher.d_out.t())
                        h = make_linear_heuristic(y, y, is_directed=False)
                        mean, red = _eval(graph, queries, h, dij_mean)
                        rows.append(dict(
                            K0=K0, m=m, lr=lr, batch=batch, init=init,
                            seed=seed, epochs=args.epochs,
                            mean_expansions=mean, expansion_reduction_pct=red,
                            forced_first_m_pct=forced_red,
                            dij_mean=dij_mean, train_s=train_s,
                        ))
                        gap = red - forced_red
                        print(f"  K0={K0} lr={lr:g} bs={batch} init={init:>16s} "
                              f"seed={seed}: red={red:.2f}% gap={gap:+.2f}pp "
                              f"[{train_s:.1f}s]", flush=True)

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    elapsed = time.perf_counter() - t_start
    print(f"\nWrote {len(rows)} rows to {args.out} in {elapsed/60:.1f} min.",
          flush=True)


if __name__ == "__main__":
    main()
