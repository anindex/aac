"""Contextual training loop with time and memory tracking.

Trains the encoder + compressor end-to-end via:
    context -> encoder -> edge costs -> smooth BF -> embed -> compress -> loss

Supports beta annealing (smooth -> hard BF) and temperature annealing
(smooth -> hard heuristic), with per-epoch timing and GPU memory tracking.

Follows the TrainConfig pattern from src/aac/train/trainer.py.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim

from aac.compression.smooth import smoothed_heuristic_directed, smoothed_heuristic_undirected
from aac.contextual.loss import contextual_loss
from aac.contextual.pipeline import contextual_forward
from aac.graphs.types import Graph


@dataclass
class ContextualConfig:
    """Configuration for Contextual training.

    Attributes:
        num_epochs: Maximum number of training epochs.
        batch_size: Number of samples per training batch.
        lr: Learning rate for Adam optimizer.
        beta_init: Initial smooth BF inverse temperature (start smooth).
        beta_max: Maximum beta value (anneal toward hard min).
        beta_gamma: Beta annealing factor (beta = min(beta_init * gamma^epoch, beta_max)).
        alpha_path: Weight for path supervision loss.
        cond_lambda: Weight for condition number regularization.
        T_init: Initial heuristic smoothing temperature.
        T_gamma: Temperature annealing factor.
        seed: Random seed for reproducibility.
        patience: Early stopping patience (epochs without improvement).
        K: Number of anchors for smooth BF.
        m: Compressed label dimension.
        grid_size: Grid size for Warcraft (inferred from encoder if not set).
    """

    num_epochs: int = 100
    batch_size: int = 24
    lr: float = 1e-3
    beta_init: float = 1.0
    beta_max: float = 30.0
    beta_gamma: float = 1.05
    alpha_path: float = 1.0
    cond_lambda: float = 0.01
    T_init: float = 1.0
    T_gamma: float = 1.05
    T_max: float = 100.0
    seed: int = 42
    patience: int = 10
    K: int = 5
    m: int = 8
    grid_size: int = 12
    alpha_cost: float = 0.0


@dataclass
class ContextualMetrics:
    """Metrics from Contextual training run.

    Attributes:
        per_epoch_loss: Loss value at each epoch.
        per_epoch_time_sec: Wall-clock time (seconds) for each epoch.
        peak_memory_bytes: Peak GPU memory allocated during training.
        total_time_sec: Total wall-clock training time.
        final_epoch: Last epoch completed (0-indexed).
        final_beta: Beta value at the last epoch.
        final_T: Temperature value at the last epoch.
    """

    per_epoch_loss: list[float] = field(default_factory=list)
    per_epoch_time_sec: list[float] = field(default_factory=list)
    peak_memory_bytes: int = 0
    total_time_sec: float = 0.0
    final_epoch: int = 0
    final_beta: float = 1.0
    final_T: float = 1.0


class ContextualTrainer:
    """End-to-end trainer for Contextual: encoder + compressor.

    Trains both the neural edge cost encoder and the compressor
    jointly via the differentiable pipeline: encoder -> smooth BF -> compress.

    Args:
        encoder: Neural edge cost encoder (WarcraftCNN or CabspottingMLP).
        compressor: Compressor module (LinearCompressor) for label compression.
        config: Training configuration.
        is_directed: Whether graphs are directed (default False for Warcraft).
    """

    def __init__(
        self,
        encoder: nn.Module,
        compressor: nn.Module,
        config: ContextualConfig,
        is_directed: bool = False,
    ) -> None:
        self.encoder = encoder
        self.compressor = compressor
        self.config = config
        self.is_directed = is_directed

        # Create Adam optimizer over both encoder and compressor parameters
        self.optimizer = optim.Adam(
            list(encoder.parameters()) + list(compressor.parameters()),
            lr=config.lr,
        )

    def train(
        self,
        train_data: list[tuple[torch.Tensor, ...]],
        graph_template: Graph,
        anchor_indices: torch.Tensor,
        ground_truth_paths: object | None = None,
    ) -> ContextualMetrics:
        """Train encoder + compressor end-to-end.

        Args:
            train_data: List of tuples, either:
                (context_tensor, ground_truth_distances) -- 2-tuple, or
                (context_tensor, ground_truth_distances, gt_cell_costs) -- 3-tuple
                with cost supervision (requires alpha_cost > 0 in config).
                For Warcraft: context_tensor is (1, 3, H_img, W_img) terrain image.
                ground_truth_distances: (V, V) pairwise distances.
                gt_cell_costs: (V,) ground truth cell costs (flat).
            graph_template: Graph with topology (edges/nodes) but placeholder weights.
            anchor_indices: (K,) tensor of anchor vertex indices.
            ground_truth_paths: Optional path supervision data (not used in v1).

        Returns:
            ContextualMetrics with per-epoch loss, timing, and memory.
        """
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)

        metrics = ContextualMetrics()
        V = graph_template.num_nodes

        heuristic_fn = (
            smoothed_heuristic_directed
            if self.is_directed
            else smoothed_heuristic_undirected
        )

        # Reset GPU memory tracking if available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        total_start = time.perf_counter()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            epoch_start = time.perf_counter()

            # Compute annealed beta and T
            beta = min(
                self.config.beta_init * (self.config.beta_gamma ** epoch),
                self.config.beta_max,
            )
            T = min(
                self.config.T_init * (self.config.T_gamma ** epoch),
                self.config.T_max,
            )

            # Sample a batch from train_data
            batch_size = min(self.config.batch_size, len(train_data))
            batch_indices = random.sample(range(len(train_data)), batch_size)

            epoch_loss_sum = 0.0
            self.encoder.train()
            self.compressor.train()
            self.optimizer.zero_grad()

            for idx in batch_indices:
                sample = train_data[idx]
                context, gt_distances = sample[0], sample[1]
                gt_cell_costs = sample[2] if len(sample) > 2 else None

                # Callers must provide context with batch dimension:
                #   CNN: (1, C, H, W)  |  MLP: (1, E, F)
                # Forward pass through the full pipeline
                # Derive tau from T: high T -> low tau (sharp selection)
                tau = max(0.1, 1.0 / T)
                output = contextual_forward(
                    encoder=self.encoder,
                    compressor=self.compressor,
                    context=context,
                    graph_template=graph_template,
                    anchor_indices=anchor_indices,
                    beta=beta,
                    compressed_dim=self.config.m,
                    is_directed=self.is_directed,
                    tau=tau,
                )

                # Sample query pairs for gap-closing loss
                # Vary seed by epoch for training diversity
                num_queries = min(32, V)
                qgen = torch.Generator().manual_seed(epoch * len(train_data) + idx)
                sources = torch.randint(0, V, (num_queries,), generator=qgen)
                targets = torch.randint(0, V, (num_queries,), generator=qgen)

                # Compute smoothed heuristic from compressed labels
                y = output.compressed_labels  # (V, m)
                y_s = y[sources]
                y_t = y[targets]
                h_smooth = heuristic_fn(y_s, y_t, T)

                # True distances for these query pairs
                d_true = gt_distances[sources, targets].to(h_smooth.dtype)

                # Cost supervision: MSE between predicted and GT cell costs
                cost_loss_val = torch.tensor(0.0)
                if (
                    gt_cell_costs is not None
                    and self.config.alpha_cost > 0
                    and output.cell_costs is not None
                ):
                    pred_cell_costs = output.cell_costs.squeeze(0)  # (V,)
                    cost_loss_val = torch.nn.functional.mse_loss(
                        pred_cell_costs.to(gt_cell_costs.dtype),
                        gt_cell_costs,
                    )

                # Combined loss
                loss = contextual_loss(
                    d_true=d_true,
                    h_smooth=h_smooth,
                    compressor=self.compressor,
                    aux_loss_val=cost_loss_val,
                    alpha_aux=self.config.alpha_cost,
                    cond_lambda=self.config.cond_lambda,
                )

                # Accumulate loss (we backprop once per batch)
                loss_scaled = loss / batch_size
                loss_scaled.backward()
                epoch_loss_sum += loss.item()

            # Update parameters
            self.optimizer.step()

            epoch_time = time.perf_counter() - epoch_start
            epoch_loss_avg = epoch_loss_sum / batch_size

            metrics.per_epoch_loss.append(epoch_loss_avg)
            metrics.per_epoch_time_sec.append(epoch_time)
            metrics.final_epoch = epoch
            metrics.final_beta = beta
            metrics.final_T = T

            # Early stopping
            if epoch_loss_avg < best_loss - 1e-6:
                best_loss = epoch_loss_avg
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.config.patience:
                break

        # Record total time and peak memory
        metrics.total_time_sec = time.perf_counter() - total_start

        if torch.cuda.is_available():
            metrics.peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        else:
            metrics.peak_memory_bytes = 0

        return metrics

    def evaluate(
        self,
        test_data: list[tuple[torch.Tensor, torch.Tensor]],
        graph_template: Graph,
        anchor_indices: torch.Tensor,
    ) -> dict:
        """Evaluate model on test data without gradients.

        Computes cost regret and heuristic gap on test samples.

        Args:
            test_data: List of (context_tensor, ground_truth_distances).
            graph_template: Graph template for forward pass.
            anchor_indices: (K,) anchor indices.

        Returns:
            Dict with evaluation metrics: avg_gap, avg_regret, num_samples.
        """
        self.encoder.eval()
        self.compressor.eval()

        V = graph_template.num_nodes
        heuristic_fn = (
            smoothed_heuristic_directed
            if self.is_directed
            else smoothed_heuristic_undirected
        )

        total_gap = 0.0
        total_queries = 0

        with torch.no_grad():
            for context, gt_distances in test_data:

                output = contextual_forward(
                    encoder=self.encoder,
                    compressor=self.compressor,
                    context=context,
                    graph_template=graph_template,
                    anchor_indices=anchor_indices,
                    beta=self.config.beta_max,
                    compressed_dim=self.config.m,
                    is_directed=self.is_directed,
                )

                # Sample query pairs
                num_queries = min(64, V)
                sources = torch.randint(0, V, (num_queries,))
                targets = torch.randint(0, V, (num_queries,))

                y = output.compressed_labels
                h = heuristic_fn(y[sources], y[targets], 100.0)  # high T for near-hard max
                d = gt_distances[sources, targets].to(h.dtype)

                gap = (d - h).mean().item()
                total_gap += gap
                total_queries += 1

        return {
            "avg_gap": total_gap / max(total_queries, 1),
            "num_samples": len(test_data),
        }
