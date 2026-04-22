#!/usr/bin/env python
"""Coverage-aware regularizer sweep on non-road graphs (SBM, BA, OGB-arXiv).

Companion to ``scripts/run_coverage_aware.py`` (which targets Modena/Manhattan).
This script runs the same coverage-aware training (adding a differentiable
covering-radius regularizer to the AAC loss) on the three non-road graphs used
throughout Section 7 of the paper, so that the appendix can document whether the
regularizer helps on clustered, power-law, or citation topologies.

Loss::

    L = gap_loss + lambda_uniq * R_uniq + lambda_cov * R_cov

where ``R_cov = (1/B) sum_v soft_min_k d(v, l_k)`` over a mini-batch of vertices
and ``soft_min`` uses log-sum-exp with negative temperature ``alpha``.

At each ``lambda_cov`` we report:
    * AAC expansion reduction vs Dijkstra
    * ALT(K=m) reduction as a reference (matched deployed labels)
    * Wilcoxon test vs the baseline (lambda_cov=0) AAC run.

Output: results/coverage_aware/coverage_aware_nonroad_results.csv
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

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, compute_teacher_heuristic
from experiments.utils import compute_strong_lcc, generate_queries

from scripts.run_synthetic_experiments import (
    GRAPH_SEED,
    NUM_QUERIES,
    QUERY_SEED,
    generate_community_graph,
    generate_powerlaw_graph,
    nx_to_graph,
)

SENTINEL = 1e18

OUTPUT_DIR = _PROJECT_ROOT / "results" / "coverage_aware"
SEEDS = [42, 123, 456, 789, 1024]

# m=8 at K0=32 (32 B/v) -- a budget where the ALT-AAC gap matters on non-road.
K0 = 32
M = 8

NUM_EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3

LAMBDA_COV_VALUES = [0.0, 0.001, 0.01, 0.1]


def train_with_coverage(
    compressor: LinearCompressor,
    teacher_labels,
    config: TrainConfig,
    valid_vertices: torch.Tensor | None,
    lambda_cov: float,
) -> None:
    """Train AAC with optional coverage-aware regularizer.

    Mirrors ``scripts/run_coverage_aware.train_with_coverage`` but supports
    undirected graphs where ``d_in == d_out`` and the compressor has a single
    selector ``W`` rather than two.
    """
    torch.manual_seed(config.seed)
    optimizer = torch.optim.Adam(compressor.parameters(), lr=config.lr)

    d_out_t = teacher_labels.d_out.t()  # (V, K)
    d_in_t = teacher_labels.d_in.t()    # (V, K)
    V = d_out_t.shape[0]
    is_directed = teacher_labels.is_directed
    sentinel_thresh = 0.99 * SENTINEL

    N = valid_vertices.shape[0] if valid_vertices is not None else V

    tau_start = 1.0
    tau_end = 0.1
    uniqueness_lambda = 0.1
    alpha_cov = 10.0

    for epoch in range(config.num_epochs):
        compressor.train()
        progress = epoch / max(config.num_epochs - 1, 1)
        tau = tau_start * (tau_end / tau_start) ** progress

        if valid_vertices is not None:
            idx_s = torch.randint(0, N, (config.batch_size,))
            idx_t = torch.randint(0, N, (config.batch_size,))
            sources = valid_vertices[idx_s]
            targets = valid_vertices[idx_t]
        else:
            sources = torch.randint(0, V, (config.batch_size,))
            targets = torch.randint(0, V, (config.batch_size,))

        with torch.no_grad():
            h_teacher = compute_teacher_heuristic(teacher_labels, sources, targets)

        if is_directed:
            d_out_s = d_out_t[sources]
            d_out_t_batch = d_out_t[targets]
            d_in_s = d_in_t[sources]
            d_in_t_batch = d_in_t[targets]

            valid_out = (d_out_s < sentinel_thresh) & (d_out_t_batch < sentinel_thresh)
            valid_in = (d_in_s < sentinel_thresh) & (d_in_t_batch < sentinel_thresh)

            fwd_delta = torch.where(valid_out, d_out_t_batch - d_out_s, torch.zeros_like(d_out_s))
            bwd_delta = torch.where(valid_in, d_in_s - d_in_t_batch, torch.zeros_like(d_in_s))

            data_dtype = fwd_delta.dtype
            A_fwd = compressor._get_A_soft(compressor.W_fwd, tau, dtype=data_dtype)
            A_bwd = compressor._get_A_soft(compressor.W_bwd, tau, dtype=data_dtype)

            h_fwd = (fwd_delta @ A_fwd.t()).max(dim=-1).values
            h_bwd = (bwd_delta @ A_bwd.t()).max(dim=-1).values
            h_compressed = torch.maximum(
                torch.maximum(h_fwd, h_bwd), torch.zeros_like(h_fwd)
            )
        else:
            d_s = d_out_t[sources]
            d_t = d_out_t[targets]
            valid = (d_s < sentinel_thresh) & (d_t < sentinel_thresh)
            delta = torch.where(valid, d_s - d_t, torch.zeros_like(d_s))
            A = compressor._get_A_soft(compressor.W, tau, dtype=delta.dtype)
            h_compressed = (delta @ A.t()).abs().max(dim=-1).values

        gap = h_teacher - h_compressed
        loss = (
            gap.mean()
            + uniqueness_lambda * compressor.uniqueness_penalty()
            + config.cond_lambda * compressor.condition_regularization()
        )

        if lambda_cov > 0:
            if valid_vertices is not None:
                cov_idx = torch.randint(0, N, (config.batch_size,))
                cov_verts = valid_vertices[cov_idx]
            else:
                cov_verts = torch.randint(0, V, (config.batch_size,))

            if is_directed:
                d_to_v = d_out_t[cov_verts]
                soft_d_fwd = d_to_v @ A_fwd.t()
                soft_min_fwd = -torch.logsumexp(-alpha_cov * soft_d_fwd, dim=-1) / alpha_cov
                d_from_v = d_in_t[cov_verts]
                soft_d_bwd = d_from_v @ A_bwd.t()
                soft_min_bwd = -torch.logsumexp(-alpha_cov * soft_d_bwd, dim=-1) / alpha_cov
                cov_penalty = torch.maximum(soft_min_fwd, soft_min_bwd).mean()
            else:
                d_to_v = d_out_t[cov_verts]
                A = compressor._get_A_soft(compressor.W, tau, dtype=d_to_v.dtype)
                soft_d = d_to_v @ A.t()
                cov_penalty = (-torch.logsumexp(-alpha_cov * soft_d, dim=-1) / alpha_cov).mean()

            loss = loss + lambda_cov * cov_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def build_aac(graph, lcc_tensor, lcc_seed, seed: int, lambda_cov: float):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = torch.Generator().manual_seed(seed)
    anchors = farthest_point_sampling(
        graph, K0, seed_vertex=lcc_seed, rng=rng, valid_vertices=lcc_tensor,
    )
    teacher = compute_teacher_labels(graph, anchors, use_gpu=False)
    compressor = LinearCompressor(K=K0, m=M, is_directed=graph.is_directed)
    cfg = TrainConfig(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=seed)
    train_with_coverage(compressor, teacher, cfg, lcc_tensor, lambda_cov)

    compressor.eval()
    d_out_eval = teacher.d_out.t()
    d_in_eval = teacher.d_in.t()
    with torch.no_grad():
        if graph.is_directed:
            y_fwd, y_bwd = compressor(d_out_eval, d_in_eval)
            y_fwd, y_bwd = y_fwd.detach(), y_bwd.detach()
        else:
            y = compressor(d_out_eval)
            y_fwd = y_bwd = y.detach()
    h = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)
    return h, compressor.selection_stats()


def run_graph(graph_type: str, graph) -> list[dict]:
    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)
    print(f"  LCC: {len(lcc_nodes):,} / {graph.num_nodes:,} nodes")

    queries = generate_queries(graph, NUM_QUERIES, seed=QUERY_SEED)
    dij_exps = np.array([dijkstra(graph, s, t).expansions for s, t in queries])
    dij_mean = float(np.mean(dij_exps))
    print(f"  Dijkstra mean expansions: {dij_mean:.0f}")

    rows: list[dict] = []

    # ALT at K=m (same deployed labels as AAC): build once.
    alt_data = alt_preprocess(graph, M, seed_vertex=lcc_seed, valid_vertices=lcc_tensor)
    alt_h = make_alt_heuristic(alt_data)
    alt_exps = np.array([astar(graph, s, t, heuristic=alt_h).expansions for s, t in queries])
    alt_mean = float(np.mean(alt_exps))
    alt_red = 100.0 * (1.0 - alt_mean / dij_mean)
    print(f"  ALT(K={M}) mean={alt_mean:.0f}  reduction={alt_red:.2f}%")

    # Per-(seed, lambda_cov) AAC runs. Store baseline (lambda_cov=0) expansions
    # per-query to drive Wilcoxon tests against non-zero regularizer values.
    baseline_exps: dict[int, np.ndarray] = {}

    for seed in SEEDS:
        for lambda_cov in LAMBDA_COV_VALUES:
            t0 = time.perf_counter()
            h_aac, stats = build_aac(graph, lcc_tensor, lcc_seed, seed, lambda_cov)
            aac_exps = np.array(
                [astar(graph, s, t, heuristic=h_aac).expansions for s, t in queries]
            )
            aac_mean = float(np.mean(aac_exps))
            aac_red = 100.0 * (1.0 - aac_mean / dij_mean)
            prep_time = time.perf_counter() - t0

            if lambda_cov == 0.0:
                baseline_exps[seed] = aac_exps
                p_two = 1.0
                stat = 0.0
            else:
                base = baseline_exps.get(seed)
                diff = aac_exps - base if base is not None else None
                try:
                    stat, p_two = scipy.stats.wilcoxon(diff, alternative="two-sided")
                except ValueError:
                    stat, p_two = 0.0, 1.0

            print(
                f"    seed={seed} lambda_cov={lambda_cov}: "
                f"AAC mean={aac_mean:.0f} red={aac_red:.2f}%  "
                f"p vs baseline={p_two:.2e}  [{prep_time:.1f}s]"
            )

            rows.append({
                "graph_type": graph_type,
                "seed": seed,
                "lambda_cov": lambda_cov,
                "K0": K0,
                "m": M,
                "dij_mean_exp": f"{dij_mean:.1f}",
                "alt_mean_exp": f"{alt_mean:.1f}",
                "aac_mean_exp": f"{aac_mean:.1f}",
                "alt_reduction_pct": f"{alt_red:.2f}",
                "aac_reduction_pct": f"{aac_red:.2f}",
                "wilcoxon_vs_baseline_stat": f"{stat:.1f}",
                "wilcoxon_vs_baseline_p": f"{p_two:.6e}",
                "unique_fwd": stats["unique_fwd"],
                "unique_bwd": stats["unique_bwd"],
                "effective_unique_ratio": stats["effective_unique_ratio"],
                "prep_time_s": f"{prep_time:.2f}",
            })

    return rows


CSV_COLUMNS = [
    "graph_type", "seed", "lambda_cov", "K0", "m",
    "dij_mean_exp", "alt_mean_exp", "aac_mean_exp",
    "alt_reduction_pct", "aac_reduction_pct",
    "wilcoxon_vs_baseline_stat", "wilcoxon_vs_baseline_p",
    "unique_fwd", "unique_bwd", "effective_unique_ratio",
    "prep_time_s",
]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graphs", default="sbm,ba,ogbn",
        help="Comma-separated subset of {sbm, ba, ogbn}.",
    )
    parser.add_argument(
        "--output-name", default="coverage_aware_nonroad_results.csv",
    )
    args = parser.parse_args()
    which = {s.strip() for s in args.graphs.split(",")}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / args.output_name
    all_rows: list[dict] = []

    def _flush() -> None:
        write_csv(all_rows, csv_path)

    if "sbm" in which:
        print(f"\n{'='*70}\n  SBM COMMUNITY GRAPH (5x2000, p_in=0.05, p_out=0.001)\n{'='*70}")
        G_sbm = generate_community_graph(GRAPH_SEED)
        graph_sbm = nx_to_graph(G_sbm, weight_seed=GRAPH_SEED)
        print(f"  AAC: {graph_sbm.num_nodes:,} nodes, {graph_sbm.num_edges:,} edges")
        all_rows.extend(run_graph("community_sbm", graph_sbm))
        _flush()

    if "ba" in which:
        print(f"\n{'='*70}\n  BARABASI-ALBERT (10k, m=5)\n{'='*70}")
        G_ba = generate_powerlaw_graph(GRAPH_SEED)
        graph_ba = nx_to_graph(G_ba, weight_seed=GRAPH_SEED)
        print(f"  AAC: {graph_ba.num_nodes:,} nodes, {graph_ba.num_edges:,} edges")
        all_rows.extend(run_graph("powerlaw_ba", graph_ba))
        _flush()

    if "ogbn" in which:
        print(f"\n{'='*70}\n  OGB-ARXIV (~170k nodes, symmetrized LCC)\n{'='*70}")
        from scripts.run_nonroad_real import load_ogbn_arxiv
        G_arx = load_ogbn_arxiv()
        graph_arx = nx_to_graph(G_arx, weight_seed=GRAPH_SEED)
        print(f"  AAC: {graph_arx.num_nodes:,} nodes, {graph_arx.num_edges:,} edges")
        all_rows.extend(run_graph("ogbn_arxiv", graph_arx))
        _flush()

    _flush()
    print("\nDone.")


if __name__ == "__main__":
    main()
