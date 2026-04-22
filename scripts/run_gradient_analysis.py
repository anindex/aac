#!/usr/bin/env python
"""Gradient flow analysis: verify differentiable backpropagation through encoder and compressor.

Trains AAC Contextual for 30 epochs on a subset of Warcraft maps
with cost supervision (alpha_cost=1.0), logging per-epoch gradient norms for
encoder (WarcraftCNN) and compressor (LinearCompressor) parameter groups.

Produces: results/warcraft/gradient_flow.csv

This provides evidence for Contribution 4 (differentiable label compression):
if both parameter groups receive non-zero gradients during training, the
pipeline truly backpropagates through the smooth Bellman-Ford and
Gumbel-softmax compressor.

Usage:
    python scripts/run_gradient_analysis.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from pathlib import Path

# Make src/ importable so `aac` and `experiments` resolve to src/.
_project_root = str(Path(__file__).resolve().parent.parent)
_src_dir = str(Path(_project_root) / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import scipy.sparse.csgraph
import torch
import torch.nn as nn
import torch.optim as optim

from aac.compression.compressor import LinearCompressor
from aac.compression.smooth import smoothed_heuristic_undirected
from aac.contextual.encoders import WarcraftCNN
from aac.contextual.loss import contextual_loss
from aac.contextual.pipeline import contextual_forward
from aac.embeddings.anchors import farthest_point_sampling
from aac.graphs.convert import graph_to_scipy
from aac.graphs.loaders.warcraft import build_warcraft_graph, load_warcraft_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "data/warcraft_real/warcraft_shortest_path_oneskin"
RESULTS_DIR = "results/warcraft"
OUTPUT_CSV = "gradient_flow.csv"

GRID_SIZE = 12
IMG_SIZE = 96
K = 5
M = 8
SEED = 42

NUM_EPOCHS = 30
N_TRAIN = 50

# Training config (matches contextual.yaml)
BATCH_SIZE = 24
LR = 1e-3
BETA_INIT = 1.0
BETA_MAX = 30.0
BETA_GAMMA = 1.05
ALPHA_PATH = 1.0
ALPHA_COST = 1.0
COND_LAMBDA = 0.01
T_INIT = 1.0
T_GAMMA = 1.05
T_MAX = 100.0
PATIENCE = 30  # high patience to avoid early stop during gradient analysis

CSV_COLUMNS = [
    "epoch",
    "encoder_grad_norm",
    "compressor_grad_norm",
    "encoder_max_grad",
    "compressor_max_grad",
    "loss",
]


# ---------------------------------------------------------------------------
# Gradient collection
# ---------------------------------------------------------------------------
def collect_gradient_norms(
    encoder: nn.Module, compressor: nn.Module
) -> dict[str, float]:
    """Collect gradient norms for encoder and compressor parameter groups.

    For each module, iterates over parameters and computes:
    - Total L2 norm: sqrt(sum(grad_norm_i^2)) across all parameters
    - Max individual parameter gradient norm

    Must be called AFTER loss.backward() but BEFORE optimizer.step().

    Args:
        encoder: Neural edge cost encoder module.
        compressor: Compressor module.

    Returns:
        Dict with keys: encoder_grad_norm, compressor_grad_norm,
        encoder_max_grad, compressor_max_grad.
    """
    result: dict[str, float] = {}

    for name, module in [("encoder", encoder), ("compressor", compressor)]:
        sum_sq = 0.0
        max_norm = 0.0
        for param in module.parameters():
            if param.grad is not None:
                pnorm = param.grad.norm().item()
                sum_sq += pnorm ** 2
                max_norm = max(max_norm, pnorm)
        result[f"{name}_grad_norm"] = math.sqrt(sum_sq)
        result[f"{name}_max_grad"] = max_norm

    return result


# ---------------------------------------------------------------------------
# Custom training loop with gradient logging
# ---------------------------------------------------------------------------
def run_gradient_analysis(
    train_data: list[tuple[torch.Tensor, ...]],
    graph_template,
    anchor_indices: torch.Tensor,
) -> list[dict]:
    """Custom training loop that logs gradient norms per epoch.

    Mirrors ContextualTrainer.train() but captures gradient norms
    between loss.backward() and optimizer.step().

    Args:
        train_data: List of (context_image, gt_distances, gt_cell_costs) tuples.
        graph_template: Graph with topology.
        anchor_indices: (K,) anchor vertex indices.

    Returns:
        List of per-epoch gradient records (dicts matching CSV_COLUMNS).
    """
    torch.manual_seed(SEED)
    random.seed(SEED)

    V = graph_template.num_nodes

    encoder = WarcraftCNN(grid_size=GRID_SIZE, img_size=IMG_SIZE)
    compressor = LinearCompressor(K=K, m=M, is_directed=False)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(compressor.parameters()), lr=LR
    )

    records: list[dict] = []

    for epoch in range(NUM_EPOCHS):
        # Compute annealed beta and T
        beta = min(BETA_INIT * (BETA_GAMMA ** epoch), BETA_MAX)
        T = min(T_INIT * (T_GAMMA ** epoch), T_MAX)

        # Sample batch
        batch_size = min(BATCH_SIZE, len(train_data))
        batch_indices = random.sample(range(len(train_data)), batch_size)

        epoch_loss_sum = 0.0
        encoder.train()
        compressor.train()
        optimizer.zero_grad()

        for idx in batch_indices:
            sample = train_data[idx]
            context, gt_distances = sample[0], sample[1]
            gt_cell_costs = sample[2] if len(sample) > 2 else None

            # Forward pass through the full pipeline
            output = contextual_forward(
                encoder=encoder,
                compressor=compressor,
                context=context,
                graph_template=graph_template,
                anchor_indices=anchor_indices,
                beta=beta,
                compressed_dim=M,
                is_directed=False,
            )

            # Sample query pairs
            num_queries = min(32, V)
            qgen = torch.Generator().manual_seed(idx)
            sources = torch.randint(0, V, (num_queries,), generator=qgen)
            targets = torch.randint(0, V, (num_queries,), generator=qgen)

            # Compute smoothed heuristic from compressed labels
            y = output.compressed_labels  # (V, m)
            y_s = y[sources]
            y_t = y[targets]
            h_smooth = smoothed_heuristic_undirected(y_s, y_t, T)

            # True distances
            d_true = gt_distances[sources, targets].to(h_smooth.dtype)

            # Cost supervision: MSE between predicted and GT cell costs
            cost_loss_val = torch.tensor(0.0)
            if gt_cell_costs is not None and output.cell_costs is not None:
                pred_cell_costs = output.cell_costs.squeeze(0)
                cost_loss_val = nn.functional.mse_loss(
                    pred_cell_costs.to(gt_cell_costs.dtype),
                    gt_cell_costs,
                )

            # Combined loss
            loss = contextual_loss(
                d_true=d_true,
                h_smooth=h_smooth,
                compressor=compressor,
                aux_loss_val=cost_loss_val,
                alpha_aux=ALPHA_COST,
                cond_lambda=COND_LAMBDA,
            )

            loss_scaled = loss / batch_size
            loss_scaled.backward()
            epoch_loss_sum += loss.item()

        # BEFORE optimizer.step(): collect gradient norms
        grad_norms = collect_gradient_norms(encoder, compressor)

        # Step optimizer
        optimizer.step()

        epoch_loss_avg = epoch_loss_sum / batch_size

        record = {
            "epoch": epoch,
            "encoder_grad_norm": grad_norms["encoder_grad_norm"],
            "compressor_grad_norm": grad_norms["compressor_grad_norm"],
            "encoder_max_grad": grad_norms["encoder_max_grad"],
            "compressor_max_grad": grad_norms["compressor_max_grad"],
            "loss": epoch_loss_avg,
        }
        records.append(record)

        print(
            f"  Epoch {epoch:3d}: loss={epoch_loss_avg:.4f}  "
            f"enc_grad={grad_norms['encoder_grad_norm']:.6f}  "
            f"comp_grad={grad_norms['compressor_grad_norm']:.6f}  "
            f"beta={beta:.2f}  T={T:.2f}"
        )

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run gradient flow analysis on Warcraft data and write CSV."""
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Gradient Flow Analysis: Encoder vs Compressor")
    print("=" * 70)
    print(f"Config: K={K}, m={M}, epochs={NUM_EPOCHS}, N_train={N_TRAIN}, alpha_cost={ALPHA_COST}")
    print(f"Data: {DATA_DIR}")
    print()

    # Load real Pogancic dataset
    print("Loading Warcraft dataset...", end=" ", flush=True)
    t0 = time.perf_counter()
    dataset = load_warcraft_dataset(DATA_DIR, GRID_SIZE)
    t_load = time.perf_counter() - t0
    print(f"done ({t_load:.1f}s)")

    train_maps = dataset["train"]["maps"]
    train_weights = dataset["train"]["weights"]

    # Build training data (first N_TRAIN maps)
    n_use = min(N_TRAIN, len(train_weights))
    print(f"Building training data ({n_use} maps)...", end=" ", flush=True)

    train_data: list[tuple[torch.Tensor, ...]] = []
    first_graph = None

    for i in range(n_use):
        graph, _ = build_warcraft_graph(train_weights[i])
        if first_graph is None:
            first_graph = graph

        sp = graph_to_scipy(graph)
        dist_matrix = scipy.sparse.csgraph.shortest_path(sp, directed=False)
        gt_dist = torch.tensor(dist_matrix, dtype=torch.float64)

        rgb = torch.from_numpy(train_maps[i]).permute(2, 0, 1).float() / 255.0
        gt_costs = torch.tensor(train_weights[i].flatten(), dtype=torch.float32)
        train_data.append((rgb.unsqueeze(0), gt_dist, gt_costs))

    print("done")

    if first_graph is None:
        print("ERROR: No training maps loaded")
        return

    # Select anchors
    anchor_indices = farthest_point_sampling(first_graph, K)
    print(f"Anchors: {anchor_indices.tolist()}")
    print()

    # Run gradient analysis
    print("Training with gradient logging...")
    print("-" * 70)
    t_train_start = time.perf_counter()
    records = run_gradient_analysis(train_data, first_graph, anchor_indices)
    t_train = time.perf_counter() - t_train_start
    print("-" * 70)
    print(f"Training complete: {len(records)} epochs, {t_train:.1f}s")
    print()

    # Write CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, OUTPUT_CSV)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote: {csv_path} ({len(records)} rows)")
    print()

    # Summary
    print("=" * 70)
    print("GRADIENT FLOW SUMMARY")
    print("=" * 70)

    first = records[0]
    last = records[-1]
    print(f"  First epoch (0): enc_grad={first['encoder_grad_norm']:.6f}  "
          f"comp_grad={first['compressor_grad_norm']:.6f}")
    print(f"  Last epoch ({last['epoch']}):  enc_grad={last['encoder_grad_norm']:.6f}  "
          f"comp_grad={last['compressor_grad_norm']:.6f}")

    # Count epochs where both groups have non-zero gradients
    both_nonzero = sum(
        1 for r in records
        if r["encoder_grad_norm"] > 0 and r["compressor_grad_norm"] > 0
    )
    total = len(records)
    pct = both_nonzero / total * 100

    print(f"\n  Epochs with both non-zero: {both_nonzero}/{total} ({pct:.0f}%)")

    if pct > 50:
        print("\n  PASS: gradients flow to both encoder and compressor")
    else:
        print("\n  FAIL: insufficient gradient flow to both parameter groups")

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
