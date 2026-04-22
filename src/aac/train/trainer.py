"""Compressor training loop with temperature annealing and early stopping.

Supports LinearCompressor (primary, Gumbel-softmax selection),
PositiveCompressor (log-domain max-plus), and DualCompressor (separate fwd/bwd).

Gap-closing loss:
    L = E[h_teacher(s,t) - h_{A,T}(s,t)] + lambda * cond_reg

Key insight: minimizing (d_true - h_compressed) and (h_teacher - h_compressed)
produce identical gradients w.r.t. compressor parameters, because d_true and
h_teacher are both constants w.r.t. the compressor. Using h_teacher avoids
computing O(V^2) pairwise distances -- the teacher labels are already available
from anchor SSSP.

Temperature annealing: For LinearCompressor, the temperature is the
Gumbel-softmax tau (high = explore landmarks, low = commit to selections).
For PositiveCompressor, the temperature is the logsumexp scale parameter
(T increases over training, pushing smooth heuristic toward the hard max).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.optim as optim

from aac.compression.compressor import DualCompressor, LinearCompressor, PositiveCompressor
from aac.compression.smooth import smoothed_heuristic_directed, smoothed_heuristic_undirected
from aac.graphs.types import TeacherLabels
from aac.train.loss import gap_closing_loss
from aac.utils.numerics import SENTINEL


@dataclass
class TrainConfig:
    """Configuration for compressor training.

    Attributes:
        num_epochs: Maximum number of training epochs.
        batch_size: Number of (source, target) pairs per batch.
        lr: Learning rate for Adam optimizer.
        cond_lambda: Weight for condition number regularization (R_ent in paper).
        uniq_lambda: Weight for the row-uniqueness penalty on the Gumbel-softmax
            selection matrix (R_uniq in the paper). Default 0.0 because the
            ablation in `results/lambda_uniq_ablation/` and Appendix G of the
            paper shows that the temperature schedule + R_ent already yield
            effective unique-ratio 1.0 at every cell we tested, so the term is
            functionally inert in our regime. Kept in the code for tighter
            m/K_0 ratios where mode collapse may bind.
        T_init: Initial temperature for smoothed heuristic.
        gamma: Temperature annealing factor (T = T_init * gamma^epoch).
        seed: Random seed for reproducibility.
        val_every: Validate every N epochs.
        patience: Early stopping patience on validation loss plateau.
    """

    num_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    cond_lambda: float = 0.01
    uniq_lambda: float = 0.0
    T_init: float = 1.0
    gamma: float = 1.05
    seed: int = 42
    val_every: int = 10
    patience: int = 20


def compute_teacher_heuristic(
    teacher_labels: TeacherLabels,
    sources: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute ALT teacher heuristic for a batch of (source, target) pairs.

    Uses the uncompressed landmark distances directly. This is equivalent
    to the identity-compression case (A=I) of AAC.

    Sentinel masking: landmarks where either vertex is unreachable are excluded.

    Args:
        teacher_labels: TeacherLabels with d_out (K, V) and d_in (K, V).
        sources: (B,) int64 source vertex indices.
        targets: (B,) int64 target vertex indices.

    Returns:
        (B,) fp64 tensor of teacher heuristic values.
    """
    d_out = teacher_labels.d_out  # (K, V)
    d_in = teacher_labels.d_in    # (K, V)
    sentinel_thresh = 0.99 * SENTINEL

    d_out_s = d_out[:, sources]  # (K, B)
    d_out_t = d_out[:, targets]  # (K, B)

    # Mask landmarks where either vertex is unreachable
    valid_out = (d_out_s < sentinel_thresh) & (d_out_t < sentinel_thresh)

    if not teacher_labels.is_directed:
        # Undirected ALT: max_k |d(k,u) - d(k,t)|
        diff = torch.abs(d_out_s - d_out_t)
        diff = torch.where(valid_out, diff, torch.zeros_like(diff))
        return torch.max(diff, dim=0).values
    else:
        d_in_s = d_in[:, sources]   # (K, B)
        d_in_t = d_in[:, targets]   # (K, B)
        valid_in = (d_in_s < sentinel_thresh) & (d_in_t < sentinel_thresh)

        # Forward bound: max_k(d_out(k,t) - d_out(k,s))
        fwd = d_out_t - d_out_s
        fwd = torch.where(valid_out, fwd, torch.full_like(fwd, float('-inf')))

        # Backward bound: max_k(d_in(k,s) - d_in(k,t))
        bwd = d_in_s - d_in_t
        bwd = torch.where(valid_in, bwd, torch.full_like(bwd, float('-inf')))

        max_fwd = torch.max(fwd, dim=0).values  # (B,)
        max_bwd = torch.max(bwd, dim=0).values  # (B,)
        return torch.clamp(torch.maximum(max_fwd, max_bwd), min=0.0)


def train_compressor(
    compressor: PositiveCompressor,
    phi: torch.Tensor,
    teacher_labels: TeacherLabels,
    config: TrainConfig = TrainConfig(),
    val_pairs: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    valid_vertices: Optional[torch.Tensor] = None,
) -> dict:
    """Train compressor via gap-closing loss with temperature annealing.

    Uses teacher heuristic (ALT lower bound from uncompressed labels) as the
    training target. This is gradient-equivalent to using true distances
    because h_teacher is constant w.r.t. compressor parameters.

    Args:
        compressor: The PositiveCompressor to train.
        phi: (V, 2K) teacher embedding in log-coordinates.
        teacher_labels: Teacher distance labels for computing h_teacher.
        config: Training configuration.
        val_pairs: Optional (val_sources, val_targets) for validation.
        valid_vertices: Optional (N,) int64 tensor of valid vertex indices
            to sample from (e.g., LCC vertices). If None, samples from all V.

    Returns:
        Dict with keys: 'train_loss' (list[float]), 'val_loss' (list[float]),
        'final_epoch' (int), 'final_T' (float), 'final_cond' (float).
    """
    torch.manual_seed(config.seed)
    optimizer = optim.Adam(compressor.parameters(), lr=config.lr)
    V = phi.shape[0]

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    patience_counter = 0
    final_epoch = 0

    is_directed = teacher_labels.is_directed
    heuristic_fn = smoothed_heuristic_directed if is_directed else smoothed_heuristic_undirected

    # Determine vertex pool for sampling
    if valid_vertices is not None:
        vertex_pool = valid_vertices
        N = vertex_pool.shape[0]
    else:
        vertex_pool = None
        N = V

    # Pre-allocate index map (reused every epoch)
    idx_map = torch.empty(V, dtype=torch.long)

    for epoch in range(config.num_epochs):
        T = config.T_init * (config.gamma ** epoch)
        compressor.train()

        # Sample random (s,t) pairs from valid vertices
        if vertex_pool is not None:
            idx_s = torch.randint(0, N, (config.batch_size,))
            idx_t = torch.randint(0, N, (config.batch_size,))
            sources = vertex_pool[idx_s]
            targets = vertex_pool[idx_t]
        else:
            sources = torch.randint(0, V, (config.batch_size,))
            targets = torch.randint(0, V, (config.batch_size,))

        # Compute teacher heuristic (constant w.r.t. compressor params)
        with torch.no_grad():
            h_teacher = compute_teacher_heuristic(teacher_labels, sources, targets)

        # Forward pass -- only compute for sampled vertices (not all V)
        unique_verts = torch.unique(torch.cat([sources, targets]))
        phi_batch = phi[unique_verts]  # (U, 2K)
        y_batch = compressor(phi_batch)  # (U, m)
        # Map original indices to batch indices
        idx_map[unique_verts] = torch.arange(unique_verts.shape[0])
        y_s = y_batch[idx_map[sources]]
        y_t = y_batch[idx_map[targets]]

        # Smooth heuristic
        h_smooth = heuristic_fn(y_s, y_t, T)

        # Loss: minimize teacher-compression gap
        loss = gap_closing_loss(h_teacher, h_smooth, compressor, config.cond_lambda)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        final_epoch = epoch

        # Validation
        if val_pairs is not None and epoch % config.val_every == 0:
            compressor.eval()
            with torch.no_grad():
                vs, vt = val_pairs
                val_unique = torch.unique(torch.cat([vs, vt]))
                y_val_batch = compressor(phi[val_unique])
                val_map = torch.empty(V, dtype=torch.long)
                val_map[val_unique] = torch.arange(val_unique.shape[0])
                y_vs, y_vt = y_val_batch[val_map[vs]], y_val_batch[val_map[vt]]
                h_val = heuristic_fn(y_vs, y_vt, T)
                h_teacher_val = compute_teacher_heuristic(teacher_labels, vs, vt)
                val_loss = (h_teacher_val - h_val).clamp(min=0).mean().item()
                val_losses.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= config.patience:
                    break

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "final_epoch": final_epoch,
        "final_T": config.T_init * (config.gamma ** final_epoch),
        "final_cond": compressor.condition_number(),
    }


def train_dual_compressor(
    compressor: DualCompressor,
    teacher_labels: TeacherLabels,
    config: TrainConfig = TrainConfig(),
    valid_vertices: torch.Tensor | None = None,
) -> dict:
    """Train a DualCompressor via gap-closing loss.

    Compresses forward (d_out) and backward (d_in) labels separately.
    The smooth heuristic for directed graphs is:
        h_smooth = max(soft_max(y_bwd_s - y_bwd_t), soft_max(y_fwd_t - y_fwd_s))
    For undirected:
        h_smooth = soft_max(|y_s - y_t|) [using both +delta and -delta]

    Args:
        compressor: DualCompressor with forward and backward sub-compressors.
        teacher_labels: Teacher distance labels.
        config: Training configuration.
        valid_vertices: Optional (N,) tensor of valid vertex indices to sample from.

    Returns:
        Dict with training metrics.
    """
    import math

    torch.manual_seed(config.seed)
    optimizer = torch.optim.Adam(compressor.parameters(), lr=config.lr)

    d_out_t = teacher_labels.d_out.t()  # (V, K) -- d(landmark, v)
    d_in_t = teacher_labels.d_in.t()    # (V, K) -- d(v, landmark)
    V = d_out_t.shape[0]
    is_directed = teacher_labels.is_directed

    if valid_vertices is not None:
        N = valid_vertices.shape[0]
    else:
        N = V

    train_losses: list[float] = []
    final_epoch = 0
    best_loss = float("inf")
    patience_counter = 0

    # Pre-allocate index map (reused every epoch)
    idx_map = torch.empty(V, dtype=torch.long)

    for epoch in range(config.num_epochs):
        T = config.T_init * (config.gamma ** epoch)
        compressor.train()

        # Sample pairs
        if valid_vertices is not None:
            idx_s = torch.randint(0, N, (config.batch_size,))
            idx_t = torch.randint(0, N, (config.batch_size,))
            sources = valid_vertices[idx_s]
            targets = valid_vertices[idx_t]
        else:
            sources = torch.randint(0, V, (config.batch_size,))
            targets = torch.randint(0, V, (config.batch_size,))

        # Teacher heuristic
        with torch.no_grad():
            h_teacher = compute_teacher_heuristic(teacher_labels, sources, targets)

        # Forward pass -- only sampled vertices
        unique_verts = torch.unique(torch.cat([sources, targets]))
        d_out_batch = d_out_t[unique_verts]  # (U, K)
        d_in_batch = d_in_t[unique_verts]    # (U, K)
        y_fwd_batch, y_bwd_batch = compressor(d_out_batch, d_in_batch, hard=False)

        idx_map[unique_verts] = torch.arange(unique_verts.shape[0])

        y_fwd_s = y_fwd_batch[idx_map[sources]]
        y_fwd_t = y_fwd_batch[idx_map[targets]]
        y_bwd_s = y_bwd_batch[idx_map[sources]]
        y_bwd_t = y_bwd_batch[idx_map[targets]]

        # Smooth heuristic via logsumexp
        if is_directed:
            # Backward bound: smooth max of (y_bwd_s - y_bwd_t)
            bwd_delta = y_bwd_s - y_bwd_t
            h_bwd = torch.logsumexp(T * bwd_delta, dim=-1) / T - math.log(bwd_delta.shape[-1]) / T
            # Forward bound: smooth max of (y_fwd_t - y_fwd_s)
            fwd_delta = y_fwd_t - y_fwd_s
            h_fwd = torch.logsumexp(T * fwd_delta, dim=-1) / T - math.log(fwd_delta.shape[-1]) / T
            h_smooth = torch.maximum(h_bwd, h_fwd)
        else:
            # Undirected: L-inf norm max|delta| (NOT variation norm max-min)
            h_smooth = smoothed_heuristic_undirected(y_bwd_s, y_bwd_t, T)

        # Loss
        gap = h_teacher - h_smooth
        loss = gap.mean() + config.cond_lambda * compressor.condition_regularization()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()
        train_losses.append(epoch_loss)
        final_epoch = epoch

        # Early stopping on training loss plateau
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.patience:
            break

    return {
        "train_loss": train_losses,
        "val_loss": [],
        "final_epoch": final_epoch,
        "final_T": config.T_init * (config.gamma ** final_epoch),
        "final_cond": compressor.condition_number(),
    }


def train_linear_compressor(
    compressor: LinearCompressor,
    teacher_labels: TeacherLabels,
    config: TrainConfig = TrainConfig(),
    valid_vertices: torch.Tensor | None = None,
) -> dict:
    """Train a LinearCompressor via Gumbel-softmax hard landmark selection.

    Uses straight-through Gumbel-softmax to learn discrete landmark selection
    with gradient flow. Temperature tau anneals from high (explore landmarks)
    to low (commit to selections).

    A uniqueness penalty discourages multiple rows from selecting the same
    landmark, maximizing coverage of the teacher label space.

    Args:
        compressor: LinearCompressor to train.
        teacher_labels: Teacher distance labels.
        config: Training configuration (T_init/gamma used for tau annealing).
        valid_vertices: Optional vertex subset to sample from.

    Returns:
        Dict with training metrics.
    """
    torch.manual_seed(config.seed)
    optimizer = torch.optim.Adam(compressor.parameters(), lr=config.lr)

    d_out_t = teacher_labels.d_out.t()  # (V, K)
    d_in_t = teacher_labels.d_in.t()    # (V, K)
    V = d_out_t.shape[0]
    is_directed = teacher_labels.is_directed
    sentinel_thresh = 0.99 * SENTINEL

    if valid_vertices is not None:
        N = valid_vertices.shape[0]
    else:
        N = V

    train_losses: list[float] = []
    final_epoch = 0
    # Tau annealing: start warm (explore), cool down (commit)
    tau_start = config.T_init  # reuse T_init as tau_start
    tau_end = 0.1
    uniqueness_lambda = config.uniq_lambda

    for epoch in range(config.num_epochs):
        compressor.train()
        # Anneal tau: exponential decay from tau_start to tau_end
        progress = epoch / max(config.num_epochs - 1, 1)
        tau = tau_start * (tau_end / tau_start) ** progress

        # Sample pairs
        if valid_vertices is not None:
            idx_s = torch.randint(0, N, (config.batch_size,))
            idx_t = torch.randint(0, N, (config.batch_size,))
            sources = valid_vertices[idx_s]
            targets = valid_vertices[idx_t]
        else:
            sources = torch.randint(0, V, (config.batch_size,))
            targets = torch.randint(0, V, (config.batch_size,))

        # Teacher heuristic (constant w.r.t. compressor)
        with torch.no_grad():
            h_teacher = compute_teacher_heuristic(teacher_labels, sources, targets)

        # Compressed heuristic via Gumbel-softmax selection
        if is_directed:
            d_out_s = d_out_t[sources]
            d_out_t_batch = d_out_t[targets]
            d_in_s = d_in_t[sources]
            d_in_t_batch = d_in_t[targets]

            valid_out = (d_out_s < sentinel_thresh) & (d_out_t_batch < sentinel_thresh)
            valid_in = (d_in_s < sentinel_thresh) & (d_in_t_batch < sentinel_thresh)

            # Use compressor forward with Gumbel-softmax tau
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

        # Loss: gap + uniqueness penalty + entropy regularization
        gap = h_teacher - h_compressed
        loss = (
            gap.mean()
            + uniqueness_lambda * compressor.uniqueness_penalty()
            + config.cond_lambda * compressor.condition_regularization()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        final_epoch = epoch

    return {
        "train_loss": train_losses,
        "val_loss": [],
        "final_epoch": final_epoch,
        "final_T": tau,
        "final_cond": compressor.condition_number(),
    }
