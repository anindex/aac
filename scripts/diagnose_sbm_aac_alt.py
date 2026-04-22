#!/usr/bin/env python
"""Diagnose AAC vs ALT on SBM at B=32.

Tests several hypotheses:
  H1: Are AAC's 8 selected landmarks unique?
  H2: Is AAC's FPS-32 pool a strict superset of ALT's FPS-8?
      (Both use same lcc_seed, so yes by construction.)
  H3: Does AAC training converge to a useful subset, or does it drift away
      from the optimal "first 8" baseline?
  H4: If we *force* AAC to select the first 8 of K0=32 (identity-init eval),
      does expansion reduction match ALT(K=8)?
  H5: Does AAC(K0=8, m=8) -- identity case -- match ALT(K=8) exactly?
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

# Import synthetic graph generators
sys.path.insert(0, str(_ROOT / "scripts"))
from run_synthetic_experiments import generate_community_graph, nx_to_graph


def main() -> None:
    print("=" * 70)
    print("DIAGNOSTIC: AAC vs ALT on SBM at B=32 B/v")
    print("=" * 70)

    # Reproduce synthetic setup
    G_nx = generate_community_graph(seed=42)
    graph = nx_to_graph(G_nx, weight_seed=42)
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
          f"directed={graph.is_directed}")

    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    print(f"LCC: {len(lcc_nodes)} nodes, lcc_seed={lcc_seed}")

    queries = generate_queries(graph, 100, seed=42)
    dij_exps = np.array([dijkstra(graph, s, t).expansions for s, t in queries])
    dij_mean = float(dij_exps.mean())
    print(f"Dijkstra baseline: mean={dij_mean:.1f} expansions")
    print()

    # ========================================================================
    # ALT K=8 (baseline)
    # ========================================================================
    print("[ALT K=8]")
    t0 = time.perf_counter()
    alt8_teacher = alt_preprocess(graph, 8, seed_vertex=lcc_seed, valid_vertices=lcc_tensor)
    alt8_h = make_alt_heuristic(alt8_teacher)
    alt8_anchors = alt8_teacher.anchor_indices.tolist()
    print(f"  Landmarks: {alt8_anchors}")
    alt8_exps = np.array([astar(graph, s, t, heuristic=alt8_h).expansions for s, t in queries])
    alt8_mean = float(alt8_exps.mean())
    alt8_red = 100.0 * (1.0 - alt8_mean / dij_mean)
    print(f"  Mean expansions: {alt8_mean:.1f} ({alt8_red:.2f}% reduction)")
    print()

    # ========================================================================
    # AAC K0=32 pool (should be superset of ALT K=8)
    # ========================================================================
    print("[AAC pool: FPS K0=32]")
    pool32 = farthest_point_sampling(
        graph, 32, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
    )
    pool32_list = pool32.tolist()
    print(f"  First 8 of 32: {pool32_list[:8]}")
    print(f"  ALT-8 == first-8-of-32? {alt8_anchors == pool32_list[:8]}")
    print()

    teacher32 = compute_teacher_labels(graph, pool32, use_gpu=False)

    # ========================================================================
    # H4: Identity-subset eval -- force AAC to select first 8 indices.
    # Manually construct compressor with argmax = [0..7]
    # ========================================================================
    print("[H4: AAC forced-to-first-8 (no training, just identity subset)]")
    # Create a compressor, override W to one-hot rows on indices 0..7
    comp_id = LinearCompressor(K=32, m=8, is_directed=False)
    with torch.no_grad():
        W = torch.full((8, 32), -10.0, dtype=comp_id.W.dtype)
        for i in range(8):
            W[i, i] = 10.0
        comp_id.W.copy_(W)
    comp_id.eval()
    d_out_t32 = teacher32.d_out.t()
    with torch.no_grad():
        y_id = comp_id(d_out_t32)
    h_id = make_linear_heuristic(y_id, y_id, graph.is_directed)
    id_exps = np.array([astar(graph, s, t, heuristic=h_id).expansions for s, t in queries])
    id_mean = float(id_exps.mean())
    id_red = 100.0 * (1.0 - id_mean / dij_mean)
    print(f"  Selected landmarks: {[i for i in range(8)]} (indices in K0=32 pool)")
    print(f"  Mean expansions: {id_mean:.1f} ({id_red:.2f}% reduction)")
    print(f"  Matches ALT? diff from ALT = {id_mean - alt8_mean:+.2f} expansions")
    print()

    # ========================================================================
    # H5: AAC(K0=8, m=8) identity case -- should match ALT exactly
    # ========================================================================
    print("[H5: AAC K0=8, m=8 (trained, but m=K0 is the identity case)]")
    teacher8 = compute_teacher_labels(graph, pool32[:8], use_gpu=False)
    torch.manual_seed(42)
    comp8 = LinearCompressor(K=8, m=8, is_directed=False)
    cfg = TrainConfig(num_epochs=200, batch_size=256, lr=1e-3, seed=42)
    train_linear_compressor(comp8, teacher8, cfg, valid_vertices=lcc_tensor)
    comp8.eval()
    idx8 = comp8.W.argmax(dim=-1).tolist()
    unique8 = len(set(idx8))
    d_out_t8 = teacher8.d_out.t()
    with torch.no_grad():
        y8 = comp8(d_out_t8)
    h8 = make_linear_heuristic(y8, y8, graph.is_directed)
    k08_exps = np.array([astar(graph, s, t, heuristic=h8).expansions for s, t in queries])
    k08_mean = float(k08_exps.mean())
    k08_red = 100.0 * (1.0 - k08_mean / dij_mean)
    print(f"  Selected indices: {idx8} ({unique8}/8 unique)")
    print(f"  Mean expansions: {k08_mean:.1f} ({k08_red:.2f}% reduction)")
    print(f"  vs ALT: {k08_mean - alt8_mean:+.2f} expansions")
    print()

    # ========================================================================
    # H1, H3: Normal AAC(K0=32, m=8) training (seed 42)
    # ========================================================================
    print("[H1+H3: AAC K0=32, m=8, seed 42 (normal training)]")
    torch.manual_seed(42)
    comp = LinearCompressor(K=32, m=8, is_directed=False)
    cfg = TrainConfig(num_epochs=200, batch_size=256, lr=1e-3, seed=42)
    train_linear_compressor(comp, teacher32, cfg, valid_vertices=lcc_tensor)
    comp.eval()
    idx = comp.W.argmax(dim=-1).tolist()
    unique = len(set(idx))
    with torch.no_grad():
        y = comp(d_out_t32)
    h_norm = make_linear_heuristic(y, y, graph.is_directed)
    aac_exps = np.array([astar(graph, s, t, heuristic=h_norm).expansions for s, t in queries])
    aac_mean = float(aac_exps.mean())
    aac_red = 100.0 * (1.0 - aac_mean / dij_mean)
    print(f"  Selected indices: {sorted(idx)} ({unique}/8 unique)")
    print(f"  FPS pool indices used: {sorted({pool32_list[i] for i in idx})}")
    print(f"  Mean expansions: {aac_mean:.1f} ({aac_red:.2f}% reduction)")
    print(f"  vs ALT: {aac_mean - alt8_mean:+.2f} expansions")
    print()

    # ========================================================================
    # H3 extended: longer training
    # ========================================================================
    print("[H3b: AAC K0=32, m=8, seed 42, 500 epochs]")
    torch.manual_seed(42)
    comp_long = LinearCompressor(K=32, m=8, is_directed=False)
    cfg_long = TrainConfig(num_epochs=500, batch_size=256, lr=1e-3, seed=42)
    train_linear_compressor(comp_long, teacher32, cfg_long, valid_vertices=lcc_tensor)
    comp_long.eval()
    idx_long = comp_long.W.argmax(dim=-1).tolist()
    unique_long = len(set(idx_long))
    with torch.no_grad():
        y_long = comp_long(d_out_t32)
    h_long = make_linear_heuristic(y_long, y_long, graph.is_directed)
    aac_long_exps = np.array([astar(graph, s, t, heuristic=h_long).expansions for s, t in queries])
    aac_long_mean = float(aac_long_exps.mean())
    aac_long_red = 100.0 * (1.0 - aac_long_mean / dij_mean)
    print(f"  Selected indices: {sorted(idx_long)} ({unique_long}/8 unique)")
    print(f"  Mean expansions: {aac_long_mean:.1f} ({aac_long_red:.2f}% reduction)")
    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  ALT K=8:                       {alt8_red:6.2f}%  ({alt8_mean:.1f} exp)")
    print(f"  AAC(K0=32,m=8) forced-first-8: {id_red:6.2f}%  ({id_mean:.1f} exp)  [should equal ALT]")
    print(f"  AAC(K0=8, m=8) trained:        {k08_red:6.2f}%  ({k08_mean:.1f} exp)  [K0=m, pool identity]")
    print(f"  AAC(K0=32,m=8) 200 epochs:     {aac_red:6.2f}%  ({aac_mean:.1f} exp)  [{unique}/8 unique]")
    print(f"  AAC(K0=32,m=8) 500 epochs:     {aac_long_red:6.2f}%  ({aac_long_mean:.1f} exp)  [{unique_long}/8 unique]")
    print()
    print("Interpretation:")
    if abs(id_mean - alt8_mean) < 2.0:
        print("  [OK] Forced-first-8 matches ALT -- heuristic plumbing is fair.")
    else:
        print(f"  [!!] Forced-first-8 differs from ALT by {id_mean-alt8_mean:.1f} -- unexpected bug.")
    if aac_red < id_red - 0.5:
        print(f"  [!] Trained AAC underperforms the identity baseline by {id_red-aac_red:.2f} pp --")
        print("      training is NOT converging to the optimal subset of the pool.")
    elif aac_red > id_red + 0.5:
        print(f"  [OK] Trained AAC beats forced-first-8 by {aac_red-id_red:.2f} pp -- pool access helps.")
    else:
        print(f"  [ok] Trained AAC ~= forced-first-8 ({aac_red-id_red:+.2f} pp) -- minor.")


if __name__ == "__main__":
    main()
