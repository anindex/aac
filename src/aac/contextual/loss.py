"""Loss functions for Contextual end-to-end training.

Provides three loss functions:
- path_kl_loss: KL-divergence on edge usage distributions (path supervision)
- cost_regret_loss: relative cost regret between predicted and optimal paths
- contextual_loss: combined gap-closing + path loss + condition regularization
"""

from __future__ import annotations

import torch

import torch.nn as nn


def path_kl_loss(
    predicted_edge_logprobs: torch.Tensor,
    target_path_edges: torch.Tensor,
) -> torch.Tensor:
    """KL-divergence path loss matching DataSP's approach.

    Computes KL(target || predicted) = sum(target * (log(target) - predicted_logprobs))
    over edges, averaged over batch. Zero entries in target are masked out.

    Args:
        predicted_edge_logprobs: (B, E) log-probabilities of edge usage.
        target_path_edges: (B, E) binary indicator of ground-truth shortest path edges.

    Returns:
        Scalar loss (averaged over batch).
    """
    # Mask zero entries in target to avoid log(0)
    mask = target_path_edges > 0
    # log(target) where target > 0
    log_target = torch.zeros_like(target_path_edges)
    log_target[mask] = torch.log(target_path_edges[mask])

    # KL(target || predicted) = sum_e target_e * (log(target_e) - predicted_e)
    kl_per_sample = (target_path_edges * (log_target - predicted_edge_logprobs) * mask.float()).sum(dim=-1)
    return kl_per_sample.mean()


def cost_regret_loss(
    predicted_costs: torch.Tensor,
    optimal_costs: torch.Tensor,
) -> torch.Tensor:
    """Relative cost regret: (predicted - optimal) / optimal, averaged over batch.

    Args:
        predicted_costs: (B,) predicted path costs.
        optimal_costs: (B,) optimal path costs (must be > 0).

    Returns:
        Scalar loss.
    """
    regret = torch.abs(predicted_costs - optimal_costs) / optimal_costs.clamp(min=1e-8)
    return regret.mean()


def contextual_loss(
    d_true: torch.Tensor,
    h_smooth: torch.Tensor,
    compressor: nn.Module,
    aux_loss_val: torch.Tensor | float = 0.0,
    alpha_aux: float = 1.0,
    cond_lambda: float = 0.01,
    # Backward-compatible aliases
    path_loss_val: torch.Tensor | float | None = None,
    alpha_path: float | None = None,
) -> torch.Tensor:
    """Combined Contextual loss: gap-closing + auxiliary loss + condition regularization.

    loss = gap_closing + alpha_aux * aux_loss_val + cond_lambda * cond_reg

    The gap-closing component (d_true - h_smooth).mean() should be >= 0
    by admissibility and is minimized to tighten the heuristic.

    The auxiliary loss can be cost supervision (MSE on cell costs) or
    path supervision (KL on edge usage).

    Args:
        d_true: (B,) true shortest-path distances.
        h_smooth: (B,) smoothed heuristic values.
        compressor: Compressor module with condition_regularization() method.
        aux_loss_val: Pre-computed auxiliary loss (scalar). Cost or path supervision.
        alpha_aux: Weight for auxiliary loss term.
        cond_lambda: Weight for condition number regularization.
        path_loss_val: Deprecated alias for aux_loss_val.
        alpha_path: Deprecated alias for alpha_aux.

    Returns:
        Scalar loss tensor.
    """
    # Support deprecated aliases
    if path_loss_val is not None:
        aux_loss_val = path_loss_val
    if alpha_path is not None:
        alpha_aux = alpha_path

    gap = (d_true - h_smooth).clamp(min=0).mean()
    cond_reg = compressor.condition_regularization()
    loss = gap + alpha_aux * aux_loss_val + cond_lambda * cond_reg
    return loss
