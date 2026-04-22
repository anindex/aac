#!/usr/bin/env python
"""Visualize FPS vs AAC vs Greedy-Max vs ALT-first-m on the SBM graph.

Four-panel companion to Figure 2 (NY road network): on a non-road graph
with community structure each selector picks a qualitatively different
subset --- consistent with the non-road results in Section 5.9 where neither
learned nor greedy selection beats FPS at matched memory.  The
ALT-first-m panel surfaces the actual matched-memory baseline so the figure
self-contains all four selectors discussed in section 5.7 (the panel is
algebraically equal to FPS-ALT $K{=}m$ via the forced-first-m identity).

Pipeline: generate the SBM graph used by the non-road experiments
(5 x 2000, p_in=0.05, p_out=0.001), run FPS to select K0=64 teacher
landmarks, then on the SAME teacher pool select m=16 landmarks three ways:
    (i)   AAC: train the linear compressor for 200 epochs.
    (ii)  Greedy-Max: greedily pick landmarks maximizing average ALT
          heuristic over a fixed query set.
    (iii) ALT first-m: take the first m landmarks of the FPS pool
          (= FPS-ALT K=m by the forced-first-m identity).
Four panels share the spring-layout embedding, colored by community.

Output: paper/figures/sbm_landmark_placement.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from aac.compression.compressor import LinearCompressor
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.train.trainer import TrainConfig, train_linear_compressor
from aac.viz.style import (
    METHOD_COLORS,
    OKABE_ITO,
    TMLR_FULL_WIDTH,
    setup_style,
)
from experiments.utils import compute_strong_lcc, generate_queries
from scripts.run_synthetic_experiments import (
    GRAPH_SEED,
    generate_community_graph,
    nx_to_graph,
)

# Semantically named colors drawn from the canonical Okabe-Ito palette
# (src/aac/viz/style.py).  Distinct hues per selector keep the legend
# unambiguous: FPS teacher reuses AAC's blue, AAC-selected uses vermillion,
# Greedy-Max gets sky-blue with a triangle marker, and the new ALT first-m
# arm uses the green slot (algebraically = FPS-ALT $K{=}m$).
_FPS_COLOR    = METHOD_COLORS["aac"]
_AAC_COLOR    = OKABE_ITO["vermillion"]
_GREEDY_COLOR = METHOD_COLORS["greedy_max"]
_ALT_COLOR    = OKABE_ITO["green"]

K0 = 64
M = 16
TRAIN_EPOCHS = 200
LR = 1e-3
SEED = 42
NUM_QUERIES_GREEDY = 100
OUTPUT = _PROJECT_ROOT / "paper" / "figures" / "sbm_landmark_placement.pdf"


def greedy_max_select_indices(teacher, m: int, queries) -> np.ndarray:
    """Mirror of scripts.run_ablation_selection.greedy_maximize_heuristic, but
    returns the selected pool indices instead of the assembled heuristic.
    Undirected case only (SBM graph)."""
    d_out = teacher.d_out.numpy()  # (K0, V)
    sources = np.array([int(s) for s, _ in queries])
    targets = np.array([int(t) for _, t in queries])
    h_per_lm = np.maximum(
        np.abs(d_out[:, sources] - d_out[:, targets]), 0.0,
    )
    Q = len(queries)
    selected: list[int] = []
    current_max = np.zeros(Q)
    for _ in range(m):
        gains = np.maximum(h_per_lm - current_max[np.newaxis, :], 0.0)
        mean_gains = gains.mean(axis=1)
        for idx in selected:
            mean_gains[idx] = -1.0
        best = int(np.argmax(mean_gains))
        selected.append(best)
        current_max = np.maximum(current_max, h_per_lm[best])
    return np.array(selected, dtype=np.int64)


def main() -> None:
    setup_style()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Generating SBM graph...")
    G_nx = generate_community_graph(GRAPH_SEED)
    # Recover block membership from block_model generator: 5 blocks of 2000.
    block_sizes = [2000] * 5
    community = np.concatenate(
        [np.full(s, i) for i, s in enumerate(block_sizes)]
    )
    graph = nx_to_graph(G_nx, weight_seed=GRAPH_SEED)
    print(f"  {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")

    lcc_nodes, lcc_seed = compute_strong_lcc(graph)
    lcc_tensor = torch.tensor(lcc_nodes, dtype=torch.int64)

    print(f"Running FPS to select K0={K0} teacher landmarks...")
    fps_landmarks = farthest_point_sampling(
        graph, K0, seed_vertex=lcc_seed, valid_vertices=lcc_tensor,
    )
    fps_ids = fps_landmarks.numpy()

    print("Computing teacher labels (K0 SSSPs)...")
    teacher = compute_teacher_labels(graph, fps_landmarks, use_gpu=False)

    print(f"Training LinearCompressor (K={K0}, m={M}, undirected)...")
    compressor = LinearCompressor(K=K0, m=M, is_directed=False)
    cfg = TrainConfig(num_epochs=TRAIN_EPOCHS, batch_size=256, lr=LR, seed=SEED)
    train_linear_compressor(compressor, teacher, cfg, valid_vertices=lcc_tensor)

    compressor.eval()
    selection = compressor.selected_landmarks()
    # Undirected: single "landmarks" key (directed: "fwd"/"bwd").
    sel_key = "landmarks" if "landmarks" in selection else "fwd"
    sel_indices = np.asarray(selection[sel_key], dtype=np.int64)
    sel_vertex_ids = fps_ids[sel_indices]
    n_unique = len(set(sel_indices.tolist()))
    print(f"  AAC selected m={M}, unique={n_unique}/{M}")

    print(f"Generating {NUM_QUERIES_GREEDY} queries for Greedy-Max scoring...")
    queries = generate_queries(graph, NUM_QUERIES_GREEDY, seed=SEED)
    print("Greedy-Max selection on the same K0 teacher pool...")
    greedy_indices = greedy_max_select_indices(teacher, M, queries)
    greedy_vertex_ids = fps_ids[greedy_indices]
    print(f"  Greedy-Max selected m={M}, "
          f"unique={len(set(greedy_indices.tolist()))}/{M}")

    print("Computing spring layout (this takes a moment for 10k nodes)...")
    pos = nx.spring_layout(G_nx, k=0.15, iterations=30, seed=SEED)
    pos_arr = np.array([pos[i] for i in range(graph.num_nodes)])

    # ALT first-m arm.  Algebraically equal to FPS-ALT
    # $K{=}m$ via the forced-first-m identity; the figure now shows all
    # four selectors discussed in section 5.7 without forcing the reader
    # to mentally reconstruct the matched-memory baseline.
    alt_indices = np.arange(M, dtype=np.int64)
    alt_vertex_ids = fps_ids[alt_indices]

    print("Plotting...")
    fig, axes = plt.subplots(
        1, 4, figsize=(TMLR_FULL_WIDTH, TMLR_FULL_WIDTH * 0.30),
    )

    community_colors = plt.get_cmap("tab10")(community / max(community.max(), 1))

    def _draw_background(ax):
        # Higher alpha so the five SBM communities read clearly without
        # overwhelming the foreground landmarks.
        ax.scatter(
            pos_arr[:, 0], pos_arr[:, 1],
            s=1.2, c=community_colors, alpha=0.40, linewidths=0,
            rasterized=True, zorder=1,
        )

    def _draw_unselected(ax, sel_idx):
        unselected_mask = np.ones(K0, dtype=bool)
        unselected_mask[sel_idx] = False
        unselected_ids = fps_ids[unselected_mask]
        if len(unselected_ids) > 0:
            ax.scatter(
                pos_arr[unselected_ids, 0], pos_arr[unselected_ids, 1],
                s=7, c="#b8b8b8", marker="o", alpha=0.55,
                edgecolors="#7a7a7a", linewidths=0.25, zorder=3,
            )

    ax_fps, ax_alt, ax_aac, ax_greedy = axes

    # Smaller in-panel landmark markers across all four panels so the
    # community structure stays readable underneath.  FPS landmarks now
    # use a bolder edge for higher contrast against the cloud.
    _LANDMARK_S = 26  # base marker area (pt^2)
    _LANDMARK_LW = 0.7  # marker edge width

    # Panel 1: all K0 FPS teacher landmarks.
    _draw_background(ax_fps)
    ax_fps.scatter(
        pos_arr[fps_ids, 0], pos_arr[fps_ids, 1],
        s=_LANDMARK_S, c=[_FPS_COLOR], marker="o",
        edgecolors="black", linewidths=_LANDMARK_LW, zorder=5,
    )
    ax_fps.set_title(f"(a) FPS Teacher ($K_0{{=}}{K0}$)", fontsize=9)
    ax_fps.set_xticks([]); ax_fps.set_yticks([])

    # Panel 2: ALT first-m (= FPS-ALT $K{=}m$ by the forced-first-m identity).
    _draw_background(ax_alt)
    _draw_unselected(ax_alt, alt_indices)
    ax_alt.scatter(
        pos_arr[alt_vertex_ids, 0], pos_arr[alt_vertex_ids, 1],
        s=_LANDMARK_S, c=[_ALT_COLOR], marker="s",
        edgecolors="black", linewidths=_LANDMARK_LW, zorder=5,
    )
    ax_alt.set_title(f"(b) ALT first-$m$ ($m{{=}}{M}$)", fontsize=9)
    ax_alt.set_xticks([]); ax_alt.set_yticks([])

    # Panel 3: AAC-selected highlighted.
    _draw_background(ax_aac)
    _draw_unselected(ax_aac, sel_indices)
    ax_aac.scatter(
        pos_arr[sel_vertex_ids, 0], pos_arr[sel_vertex_ids, 1],
        s=_LANDMARK_S, c=[_AAC_COLOR], marker="o",
        edgecolors="black", linewidths=_LANDMARK_LW, zorder=5,
    )
    ax_aac.set_title(f"(c) AAC Selected ($m{{=}}{M}$)", fontsize=9)
    ax_aac.set_xticks([]); ax_aac.set_yticks([])

    # Panel 4: Greedy-Max selected highlighted.
    _draw_background(ax_greedy)
    _draw_unselected(ax_greedy, greedy_indices)
    ax_greedy.scatter(
        pos_arr[greedy_vertex_ids, 0], pos_arr[greedy_vertex_ids, 1],
        s=_LANDMARK_S, c=[_GREEDY_COLOR], marker="^",
        edgecolors="black", linewidths=_LANDMARK_LW, zorder=5,
    )
    ax_greedy.set_title(f"(d) Greedy-Max ($m{{=}}{M}$)", fontsize=9)
    ax_greedy.set_xticks([]); ax_greedy.set_yticks([])

    fps_handle = mlines.Line2D(
        [], [], color=_FPS_COLOR, marker="o", linestyle="",
        markersize=6, markeredgecolor="black",
        label=f"FPS teacher ($K_0{{=}}{K0}$)",
    )
    drop_handle = mlines.Line2D(
        [], [], color="#cccccc", marker="o", linestyle="", markersize=5,
        markeredgecolor="#999999", label="FPS (unselected)",
    )
    alt_handle = mlines.Line2D(
        [], [], color=_ALT_COLOR, marker="s", linestyle="",
        markersize=6, markeredgecolor="black",
        label=f"ALT first-$m$ = FPS-ALT $K{{=}}{M}$",
    )
    aac_handle = mlines.Line2D(
        [], [], color=_AAC_COLOR, marker="o", linestyle="",
        markersize=7, markeredgecolor="black", label=f"AAC ($m{{=}}{M}$)",
    )
    greedy_handle = mlines.Line2D(
        [], [], color=_GREEDY_COLOR, marker="^", linestyle="",
        markersize=7, markeredgecolor="black", label=f"Greedy-Max ($m{{=}}{M}$)",
    )
    fig.legend(
        handles=[fps_handle, drop_handle, alt_handle, aac_handle, greedy_handle],
        loc="outside lower center", ncol=5, frameon=False, fontsize=7,
        columnspacing=1.0,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
