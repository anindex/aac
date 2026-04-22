#!/usr/bin/env python
"""Multi-seed ablation with full 10K training maps.

Runs the Warcraft contextual ablation (full_e2e, frozen_compressor,
frozen_encoder) using ALL available training maps with 3 seeds. Two
encoder families are supported via ``--encoder``:
- ``cnn``    (default): the small 3-layer CNN; backs the inline
  ``tab:ablation`` (cost regret) and the CNN row of
  ``tab:contextual-ablation`` in the paper.
- ``resnet``: the 5x5-stem + 3-residual-block ResNet (Appendix B); backs
  the ResNet row of ``tab:contextual-ablation``.

The cost-supervision weight ``alpha_cost`` is exposed via ``--alpha-cost``
(default ``1.0``); the paper's ResNet row uses ``alpha_cost=10``.

Outputs (filenames carry the encoder + alpha tag so different runs
do not collide):
    results/warcraft/ablation_results_<tag>_multiseed.csv
    results/warcraft/ablation_results_<tag>_perseed.csv
where ``<tag>`` is e.g. ``cnn_a1`` or ``resnet_a10``.

For backward compatibility, the canonical ``cnn_a1`` run also writes
``ablation_results_multiseed.csv``, ``ablation_results.csv``, and
``ablation_results_perseed.csv`` (the same names the original script used).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
import scipy.sparse.csgraph
import torch
import torch.nn as nn
import torch.optim as optim

from aac.compression.compressor import LinearCompressor
from aac.contextual.encoders import (
    WarcraftCNN,
    WarcraftResNet,
    build_grid_edge_index,
    cell_costs_to_edge_weights,
)
from aac.contextual.trainer import ContextualConfig, ContextualTrainer
from aac.embeddings.anchors import farthest_point_sampling
from aac.graphs.convert import edges_to_graph, graph_to_scipy
from aac.graphs.loaders.warcraft import build_warcraft_graph, load_warcraft_dataset
from aac.search.dijkstra import dijkstra

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "data/warcraft_real/warcraft_shortest_path_oneskin"
RESULTS_DIR = "results/warcraft"

GRID_SIZE = 12
IMG_SIZE = 96
K = 5
M = 8

SEEDS = [42, 123, 456]
N_TRAIN = 10000  # Use FULL training set (matches baseline training data)

NUM_EPOCHS = 100
BATCH_SIZE = 24
LR = 1e-3
BETA_INIT = 1.0
BETA_MAX = 30.0
BETA_GAMMA = 1.05
COND_LAMBDA = 0.01
T_INIT = 1.0
T_GAMMA = 1.05
PATIENCE = 15

ABLATION_MODES = ["full_e2e", "frozen_compressor", "frozen_encoder"]


def _build_encoder(encoder_kind: str) -> nn.Module:
    """Construct the requested encoder for the Warcraft 12x12 grid maps."""
    if encoder_kind == "cnn":
        return WarcraftCNN(grid_size=GRID_SIZE, img_size=IMG_SIZE)
    if encoder_kind == "resnet":
        return WarcraftResNet(grid_size=GRID_SIZE, img_size=IMG_SIZE)
    raise ValueError(
        f"Unknown encoder kind {encoder_kind!r}; expected 'cnn' or 'resnet'."
    )


def freeze_params(module: nn.Module, freeze: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(not freeze)


def _path_cost_on_graph(path, graph) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    crow = graph.crow_indices
    col = graph.col_indices
    vals = graph.values
    for u, v in zip(path[:-1], path[1:]):
        row_start = int(crow[u].item())
        row_end = int(crow[u + 1].item())
        found = False
        for idx in range(row_start, row_end):
            if int(col[idx].item()) == v:
                total += float(vals[idx].item())
                found = True
                break
        if not found:
            row_start = int(crow[v].item())
            row_end = int(crow[v + 1].item())
            for idx in range(row_start, row_end):
                if int(col[idx].item()) == u:
                    total += float(vals[idx].item())
                    found = True
                    break
        if not found:
            raise ValueError(f"Edge ({u}, {v}) not found in graph")
    return total


def _compute_path_metrics(pred_path, gt_path, pred_cost, opt_cost):
    pred_edges = set(zip(pred_path[:-1], pred_path[1:])) if len(pred_path) > 1 else set()
    gt_edges = set(zip(gt_path[:-1], gt_path[1:])) if len(gt_path) > 1 else set()
    match = pred_edges == gt_edges
    if not pred_edges and not gt_edges:
        jaccard = 1.0
    elif not pred_edges or not gt_edges:
        jaccard = 0.0
    else:
        jaccard = len(pred_edges & gt_edges) / len(pred_edges | gt_edges)
    cost_regret = (pred_cost - opt_cost) / opt_cost if opt_cost > 0 and not math.isinf(opt_cost) else 0.0
    return {"match": match, "jaccard": jaccard, "cost_regret": cost_regret}


def train_and_evaluate(
    mode: str,
    seed: int,
    train_data,
    first_graph,
    anchor_indices,
    dataset,
    encoder_kind: str = "cnn",
    alpha_cost: float = 1.0,
) -> dict:
    print(f"\n  [{mode}, encoder={encoder_kind}, alpha_cost={alpha_cost}, seed={seed}]")
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = _build_encoder(encoder_kind)
    compressor = LinearCompressor(K=K, m=M, is_directed=False)

    if mode == "frozen_compressor":
        freeze_params(compressor, freeze=True)
    elif mode == "frozen_encoder":
        freeze_params(encoder, freeze=True)

    config = ContextualConfig(
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        beta_init=BETA_INIT, beta_max=BETA_MAX, beta_gamma=BETA_GAMMA,
        cond_lambda=COND_LAMBDA, T_init=T_INIT, T_gamma=T_GAMMA,
        patience=PATIENCE, K=K, m=M, grid_size=GRID_SIZE, seed=seed,
        alpha_cost=alpha_cost,
    )

    trainer = ContextualTrainer(encoder=encoder, compressor=compressor, config=config, is_directed=False)

    trainable_params = [p for p in list(encoder.parameters()) + list(compressor.parameters()) if p.requires_grad]
    if trainable_params:
        trainer.optimizer = optim.Adam(trainable_params, lr=config.lr)
    else:
        trainer.optimizer = optim.Adam([torch.zeros(1, requires_grad=True)], lr=config.lr)

    t_train = time.perf_counter()
    metrics = trainer.train(train_data, first_graph, anchor_indices)
    train_time = time.perf_counter() - t_train
    final_epoch = int(metrics.final_epoch) + 1  # one-based; matches paper wording
    print(f"    Training: {final_epoch} epochs, {train_time:.1f}s")

    # Evaluate
    test_maps = dataset["test"]["maps"]
    test_weights = dataset["test"]["weights"]

    encoder.eval()
    compressor.eval()

    matches, jaccards, cost_regrets = [], [], []
    for i in range(len(test_weights)):
        graph, _ = build_warcraft_graph(test_weights[i])
        src_node, tgt_node = 0, graph.num_nodes - 1
        gt_result = dijkstra(graph, src_node, tgt_node)

        rgb = torch.from_numpy(test_maps[i]).permute(2, 0, 1).float() / 255.0
        context = rgb.unsqueeze(0)

        with torch.no_grad():
            cell_costs = encoder(context)
            cell_costs_2d = cell_costs.view(1, GRID_SIZE, GRID_SIZE)
            pred_edge_costs = cell_costs_to_edge_weights(cell_costs_2d, GRID_SIZE)

        src_idx, tgt_idx, _ = build_grid_edge_index(GRID_SIZE)
        pred_graph = edges_to_graph(
            src_idx, tgt_idx, pred_edge_costs.squeeze(0).to(torch.float64),
            num_nodes=graph.num_nodes, is_directed=True,
        )
        pred_result = dijkstra(pred_graph, src_node, tgt_node)
        true_cost = _path_cost_on_graph(pred_result.path, graph)
        pm = _compute_path_metrics(pred_result.path, gt_result.path, true_cost, gt_result.cost)
        matches.append(float(pm["match"]))
        jaccards.append(pm["jaccard"])
        cost_regrets.append(pm["cost_regret"])

    result = {
        "method": mode, "seed": seed,
        "path_match": float(np.mean(matches)),
        "jaccard": float(np.mean(jaccards)),
        "cost_regret": float(np.mean(cost_regrets)),
        "n_train": len(train_data), "n_test": len(test_weights),
        # final_epoch backs the paper's "CNN converges in ~30 of 100 epochs"
        # claim (Appendix B); per-run early-stopping point under PATIENCE=15.
        "final_epoch": final_epoch,
        "train_time_s": float(train_time),
    }
    print(f"    Results: match={result['path_match']:.3f}, jaccard={result['jaccard']:.3f}, regret={result['cost_regret']:.4f}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warcraft contextual ablation (full_e2e, frozen_compressor, "
            "frozen_encoder) over 3 seeds. Encoder and cost-supervision "
            "weight are configurable to back both rows of "
            "tab:contextual-ablation in the paper."
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=("cnn", "resnet"),
        default="cnn",
        help="Encoder architecture (default: cnn). 'resnet' uses the 5x5-stem "
             "+ 3 residual blocks (64 channels) variant from Appendix B.",
    )
    parser.add_argument(
        "--alpha-cost",
        type=float,
        default=1.0,
        help="Weight on the cost-supervision auxiliary loss (default: 1.0). "
             "Paper's ResNet row uses --alpha-cost 10.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Output filename tag includes encoder + alpha so multiple runs do not
    # collide in results/warcraft/ (e.g. cnn_a1, resnet_a10).
    alpha_tag = f"a{args.alpha_cost:g}".replace(".", "p")
    tag = f"{args.encoder}_{alpha_tag}"

    print(f"Multi-seed Warcraft ablation ({tag}, {N_TRAIN} training maps)")
    print(f"  Encoder: {args.encoder}")
    print(f"  Alpha-cost: {args.alpha_cost}")
    print(f"  Seeds: {SEEDS}")
    print(f"  N_TRAIN: {N_TRAIN}")

    # Load dataset once
    dataset = load_warcraft_dataset(DATA_DIR, GRID_SIZE)
    train_maps = dataset["train"]["maps"]
    train_weights = dataset["train"]["weights"]

    n_train = min(N_TRAIN, len(train_weights))
    print(f"  Building training data ({n_train} maps)...")
    t0 = time.perf_counter()

    train_data = []
    first_graph = None
    for i in range(n_train):
        graph, _ = build_warcraft_graph(train_weights[i])
        if first_graph is None:
            first_graph = graph
        sp = graph_to_scipy(graph)
        dist_matrix = scipy.sparse.csgraph.shortest_path(sp, directed=False)
        gt_dist = torch.tensor(dist_matrix, dtype=torch.float64)
        rgb = torch.from_numpy(train_maps[i]).permute(2, 0, 1).float() / 255.0
        gt_costs = torch.tensor(train_weights[i].flatten(), dtype=torch.float32)
        train_data.append((rgb.unsqueeze(0), gt_dist, gt_costs))

    print(f"  Training data built in {time.perf_counter() - t0:.1f}s")
    anchor_indices = farthest_point_sampling(first_graph, K)

    all_results = []
    for mode in ABLATION_MODES:
        for seed in SEEDS:
            result = train_and_evaluate(
                mode,
                seed,
                train_data,
                first_graph,
                anchor_indices,
                dataset,
                encoder_kind=args.encoder,
                alpha_cost=args.alpha_cost,
            )
            result["encoder"] = args.encoder
            result["alpha_cost"] = args.alpha_cost
            all_results.append(result)

    # Write per-seed results (tagged file is the canonical record for this
    # encoder/alpha combination). final_epoch + train_time_s back the paper's
    # "CNN converges in ~30 of 100 epochs" / training-cost claims.
    per_seed_tagged = os.path.join(RESULTS_DIR, f"ablation_results_{tag}_perseed.csv")
    cols = ["method", "seed", "encoder", "alpha_cost",
            "path_match", "jaccard", "cost_regret",
            "final_epoch", "train_time_s",
            "n_train", "n_test"]
    with open(per_seed_tagged, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_results)

    # Aggregate
    import pandas as pd
    df = pd.DataFrame(all_results)
    agg = df.groupby("method").agg(
        path_match_mean=("path_match", "mean"),
        path_match_std=("path_match", "std"),
        jaccard_mean=("jaccard", "mean"),
        jaccard_std=("jaccard", "std"),
        cost_regret_mean=("cost_regret", "mean"),
        cost_regret_std=("cost_regret", "std"),
        final_epoch_mean=("final_epoch", "mean"),
        final_epoch_max=("final_epoch", "max"),
        train_time_s_mean=("train_time_s", "mean"),
        n_seeds=("seed", "count"),
        n_train=("n_train", "first"),
    ).reset_index()
    agg["encoder"] = args.encoder
    agg["alpha_cost"] = args.alpha_cost

    agg_tagged = os.path.join(RESULTS_DIR, f"ablation_results_{tag}_multiseed.csv")
    agg.to_csv(agg_tagged, index=False, float_format="%.4f")

    # For the canonical default (cnn, alpha_cost=1.0) we additionally write the
    # legacy file paths so existing pipeline consumers (and the per-table
    # provenance headers in paper/) continue to work unchanged.
    is_canonical_cnn = args.encoder == "cnn" and abs(args.alpha_cost - 1.0) < 1e-9
    if is_canonical_cnn:
        per_seed_path = os.path.join(RESULTS_DIR, "ablation_results_perseed.csv")
        with open(per_seed_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(all_results)
        agg_path = os.path.join(RESULTS_DIR, "ablation_results_multiseed.csv")
        agg.to_csv(agg_path, index=False, float_format="%.4f")

    # Write a backward-compatible flat summary (per encoder/alpha cell).
    compat_path = os.path.join(
        RESULTS_DIR, f"ablation_results_{tag}.csv"
    )
    compat_cols = ["method", "encoder", "alpha_cost",
                   "path_match", "jaccard", "cost_regret", "source"]
    with open(compat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=compat_cols)
        writer.writeheader()
        for _, row in agg.iterrows():
            writer.writerow({
                "method": row["method"],
                "encoder": args.encoder,
                "alpha_cost": args.alpha_cost,
                "path_match": row["path_match_mean"],
                "jaccard": row["jaccard_mean"],
                "cost_regret": row["cost_regret_mean"],
                "source": (
                    f"AAC ablation ({int(row['n_seeds'])} seeds, "
                    f"{int(row['n_train'])} train maps, encoder={args.encoder}, "
                    f"alpha_cost={args.alpha_cost})"
                ),
            })
    if is_canonical_cnn:
        legacy_compat_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
        legacy_compat_cols = ["method", "path_match", "jaccard",
                              "cost_regret", "source"]
        with open(legacy_compat_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=legacy_compat_cols)
            writer.writeheader()
            for _, row in agg.iterrows():
                writer.writerow({
                    "method": row["method"],
                    "path_match": row["path_match_mean"],
                    "jaccard": row["jaccard_mean"],
                    "cost_regret": row["cost_regret_mean"],
                    "source": (
                        f"AAC ablation ({int(row['n_seeds'])} seeds, "
                        f"{int(row['n_train'])} train maps, encoder=cnn, "
                        f"alpha_cost=1.0)"
                    ),
                })
        # Add published baseline
        writer.writerow({
            "method": "datasp_published",
            "path_match": 0.547,
            "jaccard": float("nan"),
            "cost_regret": 0.173,
            "source": "Vlastelica et al. (ICLR 2020) Table 1, 12x12 Warcraft",
        })

    # Print summary
    print(f"\n{'='*80}")
    print(f"  {'Method':<22} {'Match':>12} {'Jaccard':>12} {'Regret':>16}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*16}")
    for _, row in agg.iterrows():
        print(f"  {row['method']:<22} "
              f"{row['path_match_mean']:.3f}+/-{row['path_match_std']:.3f} "
              f"{row['jaccard_mean']:.3f}+/-{row['jaccard_std']:.3f} "
              f"{row['cost_regret_mean']:.4f}+/-{row['cost_regret_std']:.4f}")

    print(f"\nResults: {per_seed_tagged}, {agg_tagged}, {compat_path}")


if __name__ == "__main__":
    main()
