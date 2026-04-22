"""Compression process visualization for AAC.

Visualizes what the LinearCompressor learned:
- Row-stochastic weight matrix heatmap (m x K)
- Landmark overlay: FPS teacher vs learned selected landmarks
- Gumbel-softmax selection matrix evolution animation (GIF)

All functions use the consolidated style from ``aac.viz.style``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch

from aac.viz.style import PALETTE, TMLR_FULL_WIDTH


def weight_matrix_heatmap(
    W: torch.Tensor | np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "Selection Matrix Weights",
    cmap: str = "YlOrRd",
    ylabel: str = "Compressed dim $i$",
    xlabel: str = "Teacher landmark $k$",
    show_colorbar: bool = True,
) -> plt.Figure:
    """Render an m x K row-stochastic weight matrix as a heatmap.

    Shows which teacher landmarks each compressed dimension selects.
    After Gumbel-softmax training, the matrix should be near-binary
    (each row concentrating weight on one landmark).

    The input W is the raw logits; softmax is applied per row to display
    the actual row-stochastic selection probabilities.

    Args:
        W: (m, K) weight logit matrix (raw parameters, pre-softmax).
        ax: Matplotlib axes to draw on.  If None, a new figure is created.
        title: Figure title.
        cmap: Colormap name (sequential recommended for probabilities).
        ylabel: Y-axis label.
        xlabel: X-axis label.

    Returns:
        The matplotlib Figure containing the heatmap.
    """
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().float().numpy()
    else:
        W_np = np.asarray(W, dtype=np.float32)

    # Apply softmax to get row-stochastic probabilities
    exp_W = np.exp(W_np - W_np.max(axis=1, keepdims=True))  # numerically stable
    probs = exp_W / exp_W.sum(axis=1, keepdims=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, probs.shape[1] * 0.15 + 2), max(3, probs.shape[0] * 0.4 + 1)))
    else:
        fig = ax.figure

    im = ax.imshow(probs, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    # Add colorbar only when caller hasn't opted out (e.g., shared colorbar layouts)
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Selection probability")

    # Add tick labels if dimensions are small enough
    m, K = probs.shape
    if m <= 32:
        ax.set_yticks(range(m))
        ax.set_yticklabels([f"{i}" for i in range(m)], fontsize=max(6, 10 - m // 5))
    if K <= 64:
        ax.set_xticks(range(0, K, max(1, K // 16)))

    return fig


def landmark_overlay(
    coords: np.ndarray,
    fps_ids: np.ndarray,
    learned_fwd_ids: np.ndarray,
    learned_bwd_ids: np.ndarray,
    ax_left: plt.Axes | None = None,
    ax_right: plt.Axes | None = None,
    bg_subsample: int = 5000,
    K0: int | None = None,
    m: int | None = None,
) -> plt.Figure:
    """Side-by-side overlay of FPS teacher landmarks vs AAC-selected landmarks.

    Left panel shows all K0 FPS landmarks on the graph background.
    Right panel shows learned forward and backward landmarks.

    Args:
        coords: (V, 2) vertex coordinates (x, y).
        fps_ids: (K0,) vertex indices of FPS teacher landmarks.
        learned_fwd_ids: Vertex indices of learned forward landmarks.
        learned_bwd_ids: Vertex indices of learned backward landmarks.
        ax_left: Axes for FPS panel. If None, creates new figure.
        ax_right: Axes for learned panel. If None, creates new figure.
        bg_subsample: Max background nodes to plot for performance.
        K0: Number of teacher landmarks (for legend). Inferred if None.
        m: Total compressed dims (for legend). Inferred if None.

    Returns:
        The matplotlib Figure containing both panels.
    """
    if K0 is None:
        K0 = len(fps_ids)
    if m is None:
        m = len(learned_fwd_ids) + len(learned_bwd_ids)

    if ax_left is None and ax_right is None:
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(TMLR_FULL_WIDTH, TMLR_FULL_WIDTH * 0.5)
        )
    elif ax_left is None or ax_right is None:
        raise ValueError("Provide both ax_left and ax_right, or neither.")
    else:
        fig = ax_left.figure
    assert ax_left is not None and ax_right is not None

    V = coords.shape[0]
    x, y = coords[:, 0], coords[:, 1]

    # Subsample background
    step = max(1, V // bg_subsample)
    bg_idx = np.arange(0, V, step)

    # --- Left panel: FPS Teacher Landmarks ---
    ax_left.scatter(
        x[bg_idx], y[bg_idx],
        s=0.3, c="#d0d0d0", alpha=0.5, edgecolors="none", rasterized=True,
    )
    ax_left.scatter(
        x[fps_ids], y[fps_ids],
        s=30, c=[PALETTE[0]], marker="o",
        edgecolors="black", linewidths=0.5, zorder=5,
        label=f"FPS landmarks ($K_0$={K0})",
    )
    ax_left.set_title(f"FPS Teacher Landmarks ($K_0$={K0})")
    ax_left.set_xlabel("$x$")
    ax_left.set_ylabel("$y$")
    ax_left.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # --- Right panel: AAC Selected Landmarks ---
    ax_right.scatter(
        x[bg_idx], y[bg_idx],
        s=0.3, c="#d0d0d0", alpha=0.5, edgecolors="none", rasterized=True,
    )

    # Forward landmarks (circles)
    ax_right.scatter(
        x[learned_fwd_ids], y[learned_fwd_ids],
        s=50, c=[PALETTE[0]], marker="o",
        edgecolors="black", linewidths=0.7, zorder=5,
    )

    # Backward landmarks (triangles)
    ax_right.scatter(
        x[learned_bwd_ids], y[learned_bwd_ids],
        s=50, c=[PALETTE[2]], marker="^",
        edgecolors="black", linewidths=0.7, zorder=5,
    )

    m_fwd = len(learned_fwd_ids)
    m_bwd = len(learned_bwd_ids)
    fwd_handle = mlines.Line2D(
        [], [], marker="o", color="w", markerfacecolor=PALETTE[0],
        markeredgecolor="black", markersize=8,
        label=f"Forward ($m_{{fwd}}$={m_fwd})",
    )
    bwd_handle = mlines.Line2D(
        [], [], marker="^", color="w", markerfacecolor=PALETTE[2],
        markeredgecolor="black", markersize=8,
        label=f"Backward ($m_{{bwd}}$={m_bwd})",
    )
    ax_right.legend(
        handles=[fwd_handle, bwd_handle],
        loc="lower right", fontsize=8, framealpha=0.9,
    )
    ax_right.set_title(f"AAC Selected Landmarks ($m$={m})")
    ax_right.set_xlabel("$x$")
    ax_right.set_ylabel("$y$")

    # Match axis limits
    x_range = float(x.max() - x.min())
    y_range = float(y.max() - y.min())
    pad = 0.02 * max(x_range, y_range) if x_range > 0 else 0.1
    for ax in (ax_left, ax_right):
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_ylim(y.min() - pad, y.max() + pad)
        ax.set_aspect("equal")

    fig.tight_layout()
    return fig


def selection_evolution_gif(
    snapshots: Sequence[tuple[float, torch.Tensor | np.ndarray]],
    output_path: str | Path,
    fps: int = 5,
    dpi: int = 150,
    cmap: str = "YlOrRd",
    title_prefix: str = "Selection Matrix",
) -> Path:
    """Create an animated GIF of selection matrix evolution during training.

    Each frame shows the row-stochastic weight matrix at a different epoch,
    with Gumbel-softmax temperature annotated.

    Args:
        snapshots: List of (tau_or_epoch, W_logits) tuples.
            Each W_logits is (m, K) raw weight logits.
            The first element is either tau value or epoch number.
        output_path: Path for the output GIF file.
        fps: Frames per second (default 5 for readable evolution).
        dpi: Resolution (150 is sufficient for animation).
        cmap: Colormap for the heatmap.
        title_prefix: Prefix for frame titles.

    Returns:
        Path to the created GIF file.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not snapshots:
        raise ValueError("snapshots must contain at least one frame")

    # Convert all snapshots to numpy probability matrices
    frames: list[tuple[float, np.ndarray]] = []
    for label_val, W in snapshots:
        if isinstance(W, torch.Tensor):
            W_np = W.detach().cpu().float().numpy()
        else:
            W_np = np.asarray(W, dtype=np.float32)
        # Softmax to get probabilities
        exp_W = np.exp(W_np - W_np.max(axis=1, keepdims=True))
        probs = exp_W / exp_W.sum(axis=1, keepdims=True)
        frames.append((label_val, probs))

    m, K = frames[0][1].shape

    # Create figure
    fig, ax = plt.subplots(figsize=(max(6, K * 0.12 + 2), max(3, m * 0.35 + 1.5)))
    im = ax.imshow(
        frames[0][1], aspect="auto", cmap=cmap,
        vmin=0.0, vmax=1.0, interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Selection probability")
    ax.set_ylabel("Compressed dim $i$")
    ax.set_xlabel("Teacher landmark $k$")

    # Determine if labels are tau values or epoch numbers
    # Heuristic: if first value > 0 and < 100 and not integer, treat as tau
    first_val = frames[0][0]
    is_tau = not float(first_val).is_integer() or (0 < first_val < 10)

    title_text = ax.set_title("")

    def update(frame_idx: int) -> list:
        label_val, probs = frames[frame_idx]
        im.set_data(probs)
        if is_tau:
            title_text.set_text(
                f"{title_prefix} (frame {frame_idx + 1}/{len(frames)}, "
                f"$\\tau$={label_val:.2f})"
            )
        else:
            title_text.set_text(
                f"{title_prefix} (epoch {int(label_val)}, "
                f"frame {frame_idx + 1}/{len(frames)})"
            )
        return [im, title_text]

    anim = FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 // fps, blit=False,
    )

    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)

    return output_path
