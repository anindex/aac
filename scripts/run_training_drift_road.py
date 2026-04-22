#!/usr/bin/env python
"""Training-drift diagnostic on road graphs (DIMACS NY by default).

Mirrors :mod:`run_training_drift_multi` but supports directed road networks.
For each (graph, B/v, seed) we:
  - Build the FPS pool of K0 anchors (K0 = 4*K_alt, deterministic given LCC seed).
  - Compute teacher labels (forward + backward SSSPs).
  - Forced-first-m baseline: AAC compressor with one-hot rows on the *first*
    K_alt forward pool indices and the *first* K_alt backward pool indices.
    For directed graphs at matched memory B = 8*K_alt, m_fwd = m_bwd = K_alt.
  - Pure-ALT matched-memory reference: independent FPS-ALT at K = K_alt.
  - For each epoch checkpoint, train a fresh LinearCompressor and evaluate on
    a fixed query set.

Output: results/training_drift_road/drift_<graph>_B<budget>.csv
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

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.io import load_graph_npz
from aac.graphs.loaders.dimacs import load_dimacs
from aac.graphs.types import Graph
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

OUTPUT_DIR = _ROOT / "results" / "training_drift_road"
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_CKPTS = [0, 50, 200, 500, 1000]
NUM_QUERIES = 100
QUERY_SEED = 42

GRAPHS = {
    # Paper Table tab:training-drift-road reports NY only; FLA was
    # exploratory and is not cited.
    "ny":  {"loader": "dimacs",
            "gr": "data/dimacs/USA-road-d.NY.gr",
            "co": "data/dimacs/USA-road-d.NY.co"},
    "modena": {"loader": "osmnx", "path": "data/osmnx/modena.npz"},
    "manhattan": {"loader": "osmnx", "path": "data/osmnx/manhattan.npz"},
}


def _load(key: str) -> Graph:
    spec = GRAPHS[key]
    if spec["loader"] == "osmnx":
        return load_graph_npz(_ROOT / spec["path"])
    return load_dimacs(_ROOT / spec["gr"], _ROOT / spec["co"])


def _eval(graph, queries, h, dij_mean: float) -> tuple[float, float]:
    exps = np.array([astar(graph, s, t, heuristic=h).expansions for s, t in queries])
    mean = float(exps.mean())
    red = 100.0 * (1.0 - mean / dij_mean)
    return mean, red


def _forced_first_m_directed(teacher, m_fwd: int, m_bwd: int):
    """AAC identity selection on first m_fwd fwd and m_bwd bwd pool indices.

    Constructs a LinearCompressor in eval mode whose argmax selects landmark
    i for output dimension i. Handed to make_linear_heuristic verbatim.
    """
    K = teacher.d_out.shape[0]
    assert m_fwd <= K and m_bwd <= K
    comp = LinearCompressor(K=K, m=m_fwd + m_bwd, is_directed=True,
                            m_fwd_ratio=m_fwd / (m_fwd + m_bwd))
    # Force one-hot on the first m indices in each direction.
    with torch.no_grad():
        Wf = torch.full((m_fwd, K), -10.0, dtype=comp.W_fwd.dtype)
        for i in range(m_fwd):
            Wf[i, i] = 10.0
        Wb = torch.full((m_bwd, K), -10.0, dtype=comp.W_bwd.dtype)
        for i in range(m_bwd):
            Wb[i, i] = 10.0
        comp.W_fwd.copy_(Wf)
        comp.W_bwd.copy_(Wb)
    comp.eval()
    d_out_t = teacher.d_out.t()
    d_in_t = teacher.d_in.t()
    with torch.no_grad():
        y_fwd, y_bwd = comp(d_out_t, d_in_t)
    return make_linear_heuristic(y_fwd, y_bwd, is_directed=True)


def _trained_directed(teacher, m_fwd: int, m_bwd: int,
                      epochs: int, seed: int, lcc_tensor):
    K = teacher.d_out.shape[0]
    torch.manual_seed(seed)
    comp = LinearCompressor(K=K, m=m_fwd + m_bwd, is_directed=True,
                            m_fwd_ratio=m_fwd / (m_fwd + m_bwd))
    cfg = TrainConfig(num_epochs=epochs, batch_size=256, lr=1e-3, seed=seed)
    if epochs > 0:
        train_linear_compressor(comp, teacher, cfg, valid_vertices=lcc_tensor)
    comp.eval()
    d_out_t = teacher.d_out.t()
    d_in_t = teacher.d_in.t()
    with torch.no_grad():
        y_fwd, y_bwd = comp(d_out_t, d_in_t)
    return make_linear_heuristic(y_fwd, y_bwd, is_directed=True)


def run_cell(graph_name: str, budget_bpv: int,
             seeds: Iterable[int], checkpoints: Iterable[int],
             num_queries: int, out_path: Path) -> None:
    """Run the drift experiment for one (graph, budget) cell."""
    print(f"\n=== {graph_name.upper()} | B={budget_bpv} B/v ===", flush=True)
    t_start = time.perf_counter()
    graph = _load(graph_name)
    if not graph.is_directed:
        raise ValueError(f"{graph_name} is undirected; use run_training_drift_multi.py")
    K_alt = budget_bpv // 8  # directed matched memory: ALT K = B/8
    m_fwd = K_alt
    m_bwd = K_alt
    K0 = 4 * K_alt  # teacher pool size matched to the same_pool_first_m diagnostic
    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    queries = generate_queries(graph, num_queries, seed=QUERY_SEED)
    dij_exps = np.array([dijkstra(graph, s, t).expansions for s, t in queries])
    dij_mean = float(dij_exps.mean())
    print(f"  Graph: {graph.num_nodes:,} nodes  LCC: {len(lcc_nodes):,}  "
          f"directed=True  Dij mean={dij_mean:.0f}", flush=True)
    print(f"  Config: K_alt={K_alt}  K0={K0}  m_fwd={m_fwd}  m_bwd={m_bwd}",
          flush=True)

    # Pure-ALT matched-memory reference (deterministic given lcc_seed).
    alt_labels = alt_preprocess(graph, K_alt, seed_vertex=lcc_seed,
                                valid_vertices=lcc_tensor)
    alt_h = make_alt_heuristic(alt_labels)
    alt_mean, alt_red = _eval(graph, queries, alt_h, dij_mean)
    print(f"  ALT K={K_alt}: {alt_red:.2f}%  ({alt_mean:.0f} exp)", flush=True)

    rows: list[dict] = []
    for seed in seeds:
        print(f"\n  -- seed {seed} --", flush=True)
        # FPS pool of K0 anchors (deterministic given lcc_seed, seed unused for
        # the deterministic FPS path; per-seed variance comes from training only).
        pool = farthest_point_sampling(graph, K0, seed_vertex=lcc_seed,
                                       valid_vertices=lcc_tensor)
        teacher = compute_teacher_labels(graph, pool, use_gpu=False)

        h_forced = _forced_first_m_directed(teacher, m_fwd, m_bwd)
        forced_mean, forced_red = _eval(graph, queries, h_forced, dij_mean)
        print(f"     forced-first-({m_fwd}+{m_bwd}): "
              f"{forced_red:.2f}% ({forced_mean:.0f})", flush=True)

        for ckpt in checkpoints:
            t0 = time.perf_counter()
            h = _trained_directed(teacher, m_fwd, m_bwd, ckpt, seed, lcc_tensor)
            mean, red = _eval(graph, queries, h, dij_mean)
            elapsed = time.perf_counter() - t0
            print(f"     epochs={ckpt:>4d}: {red:.2f}% ({mean:.0f}) "
                  f"[{elapsed:.1f}s]", flush=True)
            rows.append(dict(
                graph=graph_name,
                budget=budget_bpv,
                K0=K0,
                m_fwd=m_fwd,
                m_bwd=m_bwd,
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
    print(f"\n  Wrote {out_path} ({len(rows)} rows) in {elapsed/60:.1f} min.",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=["ny"],
                    choices=list(GRAPHS.keys()))
    ap.add_argument("--budgets", nargs="+", type=int, default=[64])
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--checkpoints", nargs="+", type=int, default=DEFAULT_CKPTS)
    ap.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    t0 = time.perf_counter()
    for g in args.graphs:
        for b in args.budgets:
            out = args.out_dir / f"drift_{g}_B{b}.csv"
            run_cell(g, b, args.seeds, args.checkpoints, args.num_queries, out)
    print(f"\n\nALL DONE in {(time.perf_counter()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
