#!/usr/bin/env python
"""Generate an animated GIF comparing Dijkstra, ALT, and AAC search expansions.

Produces a side-by-side 3-panel animation showing how each method progressively
expands nodes on a 2D grid with obstacles. Dijkstra floods uniformly, ALT focuses
toward the goal with classical landmarks, and AAC achieves similar focus with
fewer stored values per vertex.

Output: assets/expansion_comparison.gif

No datasets required -- uses a synthetic grid world.

Usage:
    python scripts/generate_readme_gif.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor
from aac.compression.smooth import make_aac_heuristic
from aac.graphs.convert import edges_to_graph
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from aac.viz.style import OKABE_ITO, setup_style

GRID_SIZE = 30
SEED = 42
K = 16      # ALT landmarks (at matched memory, ALT uses K floats)
M = 16      # AAC compressed dimensions (same budget as ALT)
FPS = 12
DURATION_SEC = 4.0
OUTPUT = _PROJECT_ROOT / "assets" / "expansion_comparison.gif"


def build_grid(grid_size: int, obstacles: set[tuple[int, int]]):
    """Build 8-connected grid graph with obstacles."""
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1)]
    sqrt2 = 1.4142135623730951

    node_to_rc = {}
    rc_to_node = {}
    nid = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) not in obstacles:
                node_to_rc[nid] = (r, c)
                rc_to_node[(r, c)] = nid
                nid += 1

    V = nid
    sources, targets, weights = [], [], []
    for n, (r, c) in node_to_rc.items():
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if (nr, nc) in rc_to_node:
                nbr = rc_to_node[(nr, nc)]
                if n < nbr:
                    w = sqrt2 if (dr != 0 and dc != 0) else 1.0
                    sources.append(n)
                    targets.append(nbr)
                    weights.append(w)

    graph = edges_to_graph(
        torch.tensor(sources, dtype=torch.int64),
        torch.tensor(targets, dtype=torch.int64),
        torch.tensor(weights, dtype=torch.float64),
        num_nodes=V, is_directed=False,
    )
    return graph, node_to_rc, rc_to_node, V


def make_obstacle_pattern(gs: int) -> set[tuple[int, int]]:
    """Create a maze-like obstacle pattern with guaranteed connectivity."""
    obs = set()
    # Vertical wall at col 8, rows 3-12 (gap at top and bottom)
    for r in range(3, 13):
        obs.add((r, 8))
    # Gap at row 7-8 in this wall
    obs.discard((7, 8))
    obs.discard((8, 8))

    # Vertical wall at col 16, rows 10-25 (gap at top and bottom)
    for r in range(10, 26):
        obs.add((r, 16))
    # Gap at rows 17-18
    obs.discard((17, 16))
    obs.discard((18, 16))

    # Horizontal wall at row 14, cols 3-14 (gap at left and right)
    for c in range(3, 15):
        if (14, c) not in obs:
            obs.add((14, c))
    # Gap at cols 9-10
    obs.discard((14, 9))
    obs.discard((14, 10))

    # Horizontal wall at row 20, cols 18-27 (gap at left)
    for c in range(18, 28):
        if (20, c) not in obs:
            obs.add((20, c))
    # Gap at cols 22-23
    obs.discard((20, 22))
    obs.discard((20, 23))

    # Small L-shaped obstacle block for visual interest
    for r in range(5, 9):
        obs.add((r, 20))
    for c in range(20, 24):
        obs.add((5, c))

    return obs


def render_frame(
    gs: int,
    obstacles: set,
    node_to_rc: dict,
    expanded_lists: list[list[int]],
    paths: list[list[int]],
    labels: list[str],
    colors: list[str],
    frame_frac: float,
    src_rc: tuple[int, int],
    tgt_rc: tuple[int, int],
) -> Image.Image:
    """Render one animation frame as a PIL Image."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), dpi=120)
    fig.patch.set_facecolor("#1a1a2e")

    for i, (ax, expanded, path, label, color) in enumerate(
        zip(axes, expanded_lists, paths, labels, colors)
    ):
        # How many nodes to show at this frame
        n_show = max(1, int(len(expanded) * frame_frac))
        visible = expanded[:n_show]

        # Build grid image
        grid = np.full((gs, gs, 4), [0.1, 0.1, 0.15, 1.0])  # dark background

        # Obstacles
        for r, c in obstacles:
            grid[r, c] = [0.25, 0.25, 0.3, 1.0]

        # Expansion heatmap
        cmap = plt.get_cmap("YlOrRd")
        norm = mcolors.Normalize(0, max(len(expanded) - 1, 1))
        for order, nid in enumerate(visible):
            if nid in node_to_rc:
                r, c = node_to_rc[nid]
                rgba = cmap(norm(order))
                grid[r, c] = list(rgba)

        ax.imshow(grid, interpolation="nearest", aspect="equal")

        # Draw path if animation is > 80% done
        if frame_frac > 0.8:
            path_alpha = min(1.0, (frame_frac - 0.8) / 0.2)
            path_rows = [node_to_rc[n][0] for n in path if n in node_to_rc]
            path_cols = [node_to_rc[n][1] for n in path if n in node_to_rc]
            ax.plot(path_cols, path_rows, color="white", linewidth=2.0,
                    alpha=path_alpha, zorder=10)

        # Source and target markers
        ax.plot(src_rc[1], src_rc[0], "o", color="#00ff88", markersize=8,
                markeredgecolor="white", markeredgewidth=1.5, zorder=15)
        ax.plot(tgt_rc[1], tgt_rc[0], "*", color="#ff4466", markersize=12,
                markeredgecolor="white", markeredgewidth=1.0, zorder=15)

        # Title
        title_color = "#4fc3f7" if "AAC" in label else "white"
        ax.set_title(
            f"{label}\n{n_show:,} / {len(expanded):,} expansions",
            color=title_color, fontsize=11, fontweight="bold", pad=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        # Highlight AAC panel with a colored border
        border_color = "#4fc3f7" if "AAC" in label else "#333355"
        border_width = 2.5 if "AAC" in label else 1.5
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(border_width)

    fig.tight_layout(pad=1.5)
    fig.canvas.draw()

    # Convert to PIL
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = Image.fromarray(buf[:, :, :3])  # drop alpha
    plt.close(fig)
    return img


def main():
    setup_style()
    torch.manual_seed(SEED)

    obstacles = make_obstacle_pattern(GRID_SIZE)
    graph, node_to_rc, rc_to_node, V = build_grid(GRID_SIZE, obstacles)

    src_rc, tgt_rc = (2, 2), (27, 27)
    src = rc_to_node[src_rc]
    tgt = rc_to_node[tgt_rc]

    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, {V} nodes")
    print(f"Query: {src_rc} -> {tgt_rc}")

    # Dijkstra
    r_dij = dijkstra(graph, src, tgt, track_expansions=True)
    print(f"Dijkstra: {r_dij.expansions} expansions, cost={r_dij.cost:.2f}")

    # ALT -- uses K landmarks directly
    teacher_alt = alt_preprocess(graph, K)
    h_alt = make_alt_heuristic(teacher_alt)
    r_alt = astar(graph, src, tgt, h_alt, track_expansions=True)
    print(f"ALT (K={K}): {r_alt.expansions} expansions, cost={r_alt.cost:.2f}")

    # AAC -- learns to select m=M from a larger K0=32 teacher pool
    K0 = 32
    teacher_aac = alt_preprocess(graph, K0)
    compressor = LinearCompressor(K=K0, m=M, is_directed=False)
    config = TrainConfig(
        num_epochs=300, batch_size=256, lr=1e-2,
        cond_lambda=0.01, T_init=1.0, gamma=1.05, seed=SEED, patience=30,
    )
    train_linear_compressor(compressor, teacher_aac, config)
    compressor.eval()
    with torch.no_grad():
        d_out_t = teacher_aac.d_out.t().to(torch.float64)
        compressed = compressor(d_out_t)
    h_aac = make_aac_heuristic(compressed, is_directed=False)
    r_aac = astar(graph, src, tgt, h_aac, track_expansions=True)
    print(f"AAC (m={M}): {r_aac.expansions} expansions, cost={r_aac.cost:.2f}")

    # Generate frames
    n_frames = int(FPS * DURATION_SEC)
    frames = []

    expanded_lists = [r_dij.expanded_nodes, r_alt.expanded_nodes, r_aac.expanded_nodes]
    paths = [r_dij.path, r_alt.path, r_aac.path]
    labels = [
        "Dijkstra (no heuristic)",
        f"ALT (K={K}, {K} values/v)",
        f"▸ AAC (m={M}, {M} values/v)",
    ]
    colors = [OKABE_ITO["black"], OKABE_ITO["vermillion"], OKABE_ITO["blue"]]

    print(f"\nGenerating {n_frames} frames...")
    for fi in range(n_frames):
        frac = (fi + 1) / n_frames
        frame = render_frame(
            GRID_SIZE, obstacles, node_to_rc,
            expanded_lists, paths, labels, colors,
            frac, src_rc, tgt_rc,
        )
        frames.append(frame)
        if (fi + 1) % 10 == 0:
            print(f"  Frame {fi + 1}/{n_frames}")

    # Add a few static frames at the end showing the final state
    for _ in range(int(FPS * 1.5)):
        frames.append(frames[-1])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame_duration = int(1000 / FPS)
    frames[0].save(
        str(OUTPUT),
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True,
    )
    print(f"\nSaved GIF to {OUTPUT} ({len(frames)} frames, {len(frames) * frame_duration / 1000:.1f}s)")


if __name__ == "__main__":
    main()
