#!/usr/bin/env python
"""Coverage-aware loss experiment for AAC.

Adds a differentiable covering-radius regularizer to the training loss:
    L = gap_loss + λ_uniq * R_uniq + λ_cov * R_cov

where R_cov = (1/B) Σ_v soft_min_k d(v, l_k) over a mini-batch of vertices,
and soft_min uses log-sum-exp with negative temperature.

Compares standard AAC vs coverage-aware AAC vs FPS-ALT on Modena and Manhattan.

Output: results/coverage_aware/coverage_aware_results.csv
"""

from __future__ import annotations

import csv
import logging
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
from aac.train.trainer import TrainConfig, compute_teacher_heuristic
from experiments.utils import generate_queries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SENTINEL = 1e18
OUTPUT_DIR = Path("results/coverage_aware")
GRAPHS = {
    "modena": Path("data/osmnx/modena.npz"),
    "manhattan": Path("data/osmnx/manhattan.npz"),
}
SEEDS = [42, 123, 456, 789, 1024]
NUM_QUERIES = 100
K0 = 64
M = 16  # m=16 -> 64 B/v, the main comparison budget
NUM_EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3
LAMBDA_COV_VALUES = [0.0, 0.001, 0.01, 0.1]  # 0.0 = baseline (no coverage)


def compute_lcc(graph):
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


def train_with_coverage(
    compressor: LinearCompressor,
    teacher_labels,
    config: TrainConfig,
    valid_vertices: torch.Tensor | None,
    lambda_cov: float,
) -> dict:
    """Train with optional coverage-aware regularizer."""
    torch.manual_seed(config.seed)
    optimizer = torch.optim.Adam(compressor.parameters(), lr=config.lr)

    d_out_t = teacher_labels.d_out.t()  # (V, K)
    d_in_t = teacher_labels.d_in.t()    # (V, K)
    V = d_out_t.shape[0]
    is_directed = teacher_labels.is_directed
    sentinel_thresh = 0.99 * SENTINEL

    N = valid_vertices.shape[0] if valid_vertices is not None else V

    train_losses = []
    tau_start = 1.0
    tau_end = 0.1
    uniqueness_lambda = 0.1

    for epoch in range(config.num_epochs):
        compressor.train()
        progress = epoch / max(config.num_epochs - 1, 1)
        tau = tau_start * (tau_end / tau_start) ** progress

        # Sample pairs for gap loss
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

        # Compressed heuristic
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

        # Coverage-aware regularizer
        if lambda_cov > 0:
            # Sample a batch of vertices for coverage computation
            if valid_vertices is not None:
                cov_idx = torch.randint(0, N, (config.batch_size,))
                cov_verts = valid_vertices[cov_idx]
            else:
                cov_verts = torch.randint(0, V, (config.batch_size,))

            # Get soft selection probabilities (not hard argmax)
            if is_directed:
                # Forward landmarks: A_fwd is (m_fwd, K0), soft probs
                # d_out_t[cov_verts] is (B, K0) -- distance from each landmark to vertex
                d_to_v = d_out_t[cov_verts]  # (B, K0)
                # Soft selected distances: for each compressed dim i, the soft
                # distance is sum_k A_fwd[i,k] * d(l_k, v)
                # Shape: (B, m_fwd)
                soft_d_fwd = d_to_v @ A_fwd.t()  # (B, m_fwd)
                # Soft min over selected landmarks (use logsumexp with neg temp)
                # min_i soft_d_fwd[v, i] ≈ -logsumexp(-alpha * soft_d_fwd) / alpha
                alpha = 10.0  # smoothness parameter
                # Mask out sentinel distances
                valid_mask_fwd = (d_to_v < sentinel_thresh).float()
                # For each vertex, the covering distance = min over selected landmarks
                soft_min_fwd = -torch.logsumexp(-alpha * soft_d_fwd, dim=-1) / alpha  # (B,)

                # Same for backward
                d_from_v = d_in_t[cov_verts]  # (B, K0)
                soft_d_bwd = d_from_v @ A_bwd.t()  # (B, m_bwd)
                soft_min_bwd = -torch.logsumexp(-alpha * soft_d_bwd, dim=-1) / alpha  # (B,)

                # Coverage penalty = mean over batch of max(fwd, bwd) covering dist
                cov_penalty = torch.maximum(soft_min_fwd, soft_min_bwd).mean()
            else:
                d_to_v = d_out_t[cov_verts]
                A = compressor._get_A_soft(compressor.W, tau, dtype=d_to_v.dtype)
                soft_d = d_to_v @ A.t()
                alpha = 10.0
                cov_penalty = (-torch.logsumexp(-alpha * soft_d, dim=-1) / alpha).mean()

            loss = loss + lambda_cov * cov_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    return {"train_loss": train_losses}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for graph_name, npz_path in GRAPHS.items():
        if not npz_path.exists():
            logger.warning("Graph %s not found, skipping", graph_name)
            continue

        logger.info("=== %s ===", graph_name)
        graph = load_graph_npz(npz_path)
        lcc_nodes, lcc_seed = compute_lcc(graph)
        lcc_tensor = torch.from_numpy(lcc_nodes)

        # Generate queries once
        queries = generate_queries(graph, NUM_QUERIES, seed=42)

        # Dijkstra baseline
        dij_expansions = []
        for s, t in queries:
            r = dijkstra(graph, s, t)
            dij_expansions.append(r.expansions)
        dij_mean = np.mean(dij_expansions)

        for seed in SEEDS:
            logger.info("  seed=%d", seed)

            # FPS-ALT baseline at K=8 (64 B/v)
            rng = torch.Generator().manual_seed(seed)
            alt_data = alt_preprocess(graph, 8, seed_vertex=lcc_seed, rng=rng,
                                      valid_vertices=lcc_tensor)
            h_alt = make_alt_heuristic(alt_data)
            alt_exps = []
            for s, t_node in queries:
                r = astar(graph, s, t_node, heuristic=h_alt)
                alt_exps.append(r.expansions)
            alt_mean = np.mean(alt_exps)
            alt_red = (1 - alt_mean / dij_mean) * 100

            for lambda_cov in LAMBDA_COV_VALUES:
                logger.info("    lambda_cov=%s", lambda_cov)

                # Teacher labels
                torch.manual_seed(seed)
                np.random.seed(seed)
                rng_aac = torch.Generator().manual_seed(seed)
                anchors = farthest_point_sampling(
                    graph, K0, seed_vertex=lcc_seed, rng=rng_aac,
                    valid_vertices=lcc_tensor,
                )
                teacher = compute_teacher_labels(graph, anchors)

                # Create compressor
                compressor = LinearCompressor(K0, M, is_directed=True)

                # Train with coverage-aware loss
                cfg = TrainConfig(
                    num_epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    lr=LR,
                    seed=seed,
                )
                train_with_coverage(compressor, teacher, cfg, lcc_tensor, lambda_cov)

                # Evaluate
                compressor.eval()
                d_out_eval = teacher.d_out.t()  # (V, K)
                d_in_eval = teacher.d_in.t()    # (V, K)
                with torch.no_grad():
                    y_fwd, y_bwd = compressor(d_out_eval, d_in_eval)
                h_aac = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)
                aac_exps = []
                all_optimal = True
                for s, t_node in queries:
                    r = astar(graph, s, t_node, heuristic=h_aac)
                    aac_exps.append(r.expansions)
                    # Quick admissibility check
                    r_dij = dijkstra(graph, s, t_node)
                    if abs(r.cost - r_dij.cost) > 1e-6 * max(r.cost, 1):
                        all_optimal = False

                aac_mean = np.mean(aac_exps)
                aac_red = (1 - aac_mean / dij_mean) * 100

                # Stats
                stats = compressor.selection_stats()

                results.append({
                    "graph": graph_name,
                    "seed": seed,
                    "lambda_cov": lambda_cov,
                    "aac_reduction_pct": aac_red,
                    "alt_reduction_pct": alt_red,
                    "aac_mean_exp": aac_mean,
                    "alt_mean_exp": alt_mean,
                    "dij_mean_exp": dij_mean,
                    "all_optimal": all_optimal,
                    "unique_fwd": stats["unique_fwd"],
                    "unique_bwd": stats["unique_bwd"],
                    "effective_unique_ratio": stats["effective_unique_ratio"],
                })

                label = "baseline" if lambda_cov == 0 else f"cov={lambda_cov}"
                logger.info(
                    "      %s: AAC %.1f%% | ALT %.1f%% | adm=%s",
                    label, aac_red, alt_red, all_optimal,
                )

    # Write CSV
    out_path = OUTPUT_DIR / "coverage_aware_results.csv"
    if results:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info("Results written to %s", out_path)
    else:
        logger.warning("No results to write")


if __name__ == "__main__":
    main()
