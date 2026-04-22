#!/usr/bin/env python
"""Visualize FPS landmarks vs AAC-selected landmarks on the DIMACS NY road network.

Produces a 1x2 figure showing:
  Left:  FPS teacher landmarks (K0=64) on road network background
  Right: AAC-selected landmarks (m=16, 8 fwd + 8 bwd) on road network background

The LinearCompressor is trained briefly (200 epochs) on a random subset of
vertex pairs to learn which landmarks to keep. The goal is to show spatial
distribution, not to achieve best performance.

Output: results/paper/landmark_selection.pdf (or --output path)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Add project root to sys.path so aac module is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import scipy.sparse.csgraph  # noqa: E402

from aac.compression.compressor import LinearCompressor  # noqa: E402
from aac.embeddings.anchors import farthest_point_sampling  # noqa: E402
from aac.graphs.convert import graph_to_scipy  # noqa: E402
from aac.graphs.loaders.dimacs import load_dimacs  # noqa: E402
from aac.viz.style import (  # noqa: E402
    METHOD_COLORS,
    OKABE_ITO,
    TMLR_FULL_WIDTH,
    setup_style,
)

# Semantically named colors for landmark roles, all drawn from the canonical
# Okabe-Ito palette declared in src/aac/viz/style.py.
# the AAC selection is *one* method that selects *both* a forward and a
# backward subset, so we use a single AAC hue (vermillion) with marker
# shape (filled circle vs hollow triangle) carrying the directionality.
_FPS_COLOR = METHOD_COLORS["aac"]          # teacher pool (AAC's source pool)
_AAC_COLOR = OKABE_ITO["vermillion"]       # AAC-selected (one method, two directions)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K0 = 64           # Number of FPS teacher landmarks
M = 16            # Total compressed dimensions (8 fwd + 8 bwd)
TRAIN_EPOCHS = 200
TRAIN_PAIRS = 1000  # Random vertex pairs for training
LR = 0.01
TAU_START = 2.0
TAU_END = 0.1
SEED = 42
BG_SUBSAMPLE = 5000  # Background nodes to plot


def compute_sssp_from_landmarks(
    scipy_csr: "scipy.sparse.csr_matrix",
    landmark_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SSSP distances from landmarks (d_out) and to landmarks (d_in).

    Returns:
        d_out: (K, V) forward distances from each landmark
        d_in:  (K, V) backward distances to each landmark
    """
    K = len(landmark_ids)
    V = scipy_csr.shape[0]

    # Forward: d_out[k, v] = dist(landmark_k, v)
    d_out = scipy.sparse.csgraph.dijkstra(scipy_csr, indices=landmark_ids)
    d_out = np.where(np.isinf(d_out), 1e18, d_out)

    # Backward: d_in[k, v] = dist(v, landmark_k) = d(landmark_k, v) on transpose
    # For directed graph, transpose and compute
    d_in = scipy.sparse.csgraph.dijkstra(scipy_csr.T, indices=landmark_ids)
    d_in = np.where(np.isinf(d_in), 1e18, d_in)

    return d_out, d_in


def train_compressor(
    d_out: np.ndarray,
    d_in: np.ndarray,
    rng: np.random.Generator,
) -> LinearCompressor:
    """Train a LinearCompressor on random vertex pairs.

    Uses gap-closing loss: minimize sum of (h_teacher - h_compressed)
    to make compressed heuristic as tight as possible.
    """
    K, V = d_out.shape
    compressor = LinearCompressor(K=K, m=M, is_directed=True)
    compressor.train()
    optimizer = torch.optim.Adam(compressor.parameters(), lr=LR)

    # Convert to tensors (V, K)
    d_out_t = torch.tensor(d_out.T, dtype=torch.float64)  # (V, K)
    d_in_t = torch.tensor(d_in.T, dtype=torch.float64)    # (V, K)

    # Pre-sample random pairs for all epochs
    pair_sources = rng.integers(0, V, size=(TRAIN_EPOCHS, TRAIN_PAIRS))
    pair_targets = rng.integers(0, V, size=(TRAIN_EPOCHS, TRAIN_PAIRS))

    for epoch in range(TRAIN_EPOCHS):
        # Temperature annealing: linear decay
        progress = epoch / max(1, TRAIN_EPOCHS - 1)
        tau = TAU_START + (TAU_END - TAU_START) * progress

        srcs = pair_sources[epoch]
        tgts = pair_targets[epoch]

        # Teacher heuristic: max over landmarks of |d_out[k,t] - d_out[k,s]| etc.
        # ALT-style: h(s,t) = max_k(max(d_out[k,t] - d_out[k,s], d_in[k,s] - d_in[k,t], 0))
        h_fwd = d_out[:, tgts] - d_out[:, srcs]   # (K, N_pairs) -- forward bound
        h_bwd = d_in[:, srcs] - d_in[:, tgts]     # (K, N_pairs) -- backward bound
        h_teacher_np = np.maximum(h_fwd.max(axis=0), h_bwd.max(axis=0))
        h_teacher_np = np.maximum(h_teacher_np, 0.0)
        h_teacher = torch.tensor(h_teacher_np, dtype=torch.float64)

        # Compressed heuristic
        y_fwd_s, y_bwd_s = compressor(d_out_t[srcs], d_in_t[srcs], tau=tau)
        y_fwd_t, y_bwd_t = compressor(d_out_t[tgts], d_in_t[tgts], tau=tau)

        # h_compressed = max(max(y_bwd_s - y_bwd_t), max(y_fwd_t - y_fwd_s), 0)
        bwd_bound = (y_bwd_s - y_bwd_t).max(dim=-1).values  # (N_pairs,)
        fwd_bound = (y_fwd_t - y_fwd_s).max(dim=-1).values  # (N_pairs,)
        h_compressed = torch.maximum(bwd_bound, fwd_bound)
        h_compressed = torch.clamp(h_compressed, min=0.0)

        # Gap-closing loss: minimize teacher - compressed (want compressed close to teacher)
        gap = h_teacher - h_compressed
        loss = gap.mean()

        # Add uniqueness penalty to encourage diverse landmark selection
        loss = loss + 0.1 * compressor.uniqueness_penalty()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{TRAIN_EPOCHS}: loss={loss.item():.1f}, tau={tau:.2f}")

    compressor.eval()
    return compressor


def main(output: str) -> None:
    setup_style()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # 1. Load NY graph with coordinates
    print("Loading DIMACS NY graph...")
    gr_path = os.path.join(_PROJECT_ROOT, "data/dimacs/USA-road-d.NY.gr")
    co_path = os.path.join(_PROJECT_ROOT, "data/dimacs/USA-road-d.NY.co")
    graph = load_dimacs(gr_path, co_path=co_path)
    V = graph.num_nodes
    print(f"  {V} nodes, {graph.num_edges} edges")

    # Extract coordinates (scaled by 1e-6 to get lon/lat degrees)
    coords = graph.coordinates.numpy()  # (V, 2) -- x=longitude, y=latitude scaled by 1e6
    lon = coords[:, 0] / 1e6
    lat = coords[:, 1] / 1e6

    # 2. Run FPS to select K0=64 teacher landmarks
    print(f"Running FPS to select {K0} teacher landmarks...")
    fps_landmarks = farthest_point_sampling(graph, K0, seed_vertex=0)
    fps_ids = fps_landmarks.numpy()
    print(f"  Selected {len(fps_ids)} landmarks")

    # 3. Compute SSSP from landmarks
    print("Computing SSSP from landmarks...")
    scipy_csr = graph_to_scipy(graph)
    d_out, d_in = compute_sssp_from_landmarks(scipy_csr, fps_ids)
    print(f"  d_out: {d_out.shape}, d_in: {d_in.shape}")

    # 4. Train LinearCompressor
    print(f"Training LinearCompressor (K={K0}, m={M}, {TRAIN_EPOCHS} epochs)...")
    compressor = train_compressor(d_out, d_in, rng)

    # 5. Extract selected landmarks
    selection = compressor.selected_landmarks()
    fwd_indices = selection["fwd"]  # indices into 0..K0-1
    bwd_indices = selection["bwd"]  # indices into 0..K0-1

    # Map to graph vertex IDs
    fwd_vertex_ids = fps_ids[fwd_indices]
    bwd_vertex_ids = fps_ids[bwd_indices]

    print(f"  Selected fwd landmarks ({len(fwd_indices)}): {fwd_vertex_ids[:5]}...")
    print(f"  Selected bwd landmarks ({len(bwd_indices)}): {bwd_vertex_ids[:5]}...")

    # Check for unique selections
    n_unique_fwd = len(set(fwd_indices))
    n_unique_bwd = len(set(bwd_indices))
    print(f"  Unique fwd: {n_unique_fwd}/{len(fwd_indices)}, bwd: {n_unique_bwd}/{len(bwd_indices)}")

    # 6. Create figure
    print("Creating figure...")

    # Extract road network edges for background drawing
    coo = scipy_csr.tocoo()
    edge_src, edge_dst = coo.row, coo.col
    # Subsample edges for performance (NY has ~730K edges)
    EDGE_SUBSAMPLE = 60000
    if len(edge_src) > EDGE_SUBSAMPLE:
        rng_edges = np.random.default_rng(42)
        edge_idx = rng_edges.choice(len(edge_src), EDGE_SUBSAMPLE, replace=False)
        edge_src, edge_dst = edge_src[edge_idx], edge_dst[edge_idx]
    print(f"  Drawing {len(edge_src)} edges (subsampled from {coo.nnz})")

    # Build set of selected landmark vertex IDs for highlighting
    selected_set = set(fwd_vertex_ids.tolist()) | set(bwd_vertex_ids.tolist())
    unselected_fps = np.array([v for v in fps_ids if v not in selected_set])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(TMLR_FULL_WIDTH, TMLR_FULL_WIDTH * 0.5))

    def _draw_edges(ax):
        """Draw road network edges with enough contrast to read as the
        graph topology, but light enough to recede behind the landmarks."""
        from matplotlib.collections import LineCollection
        segs = np.stack([
            np.column_stack([lon[edge_src], lat[edge_src]]),
            np.column_stack([lon[edge_dst], lat[edge_dst]]),
        ], axis=1)  # (N, 2, 2)
        lc = LineCollection(
            segs,
            colors="#9aa4ad", linewidths=0.30, alpha=0.55,
            rasterized=True, zorder=1,
        )
        ax.add_collection(lc)

    # ---- Left panel: FPS Teacher Landmarks ----
    _draw_edges(ax_left)
    ax_left.scatter(
        lon[fps_ids], lat[fps_ids],
        s=18, c=[_FPS_COLOR], marker="o",
        edgecolors="black", linewidths=0.5, zorder=5,
        label=f"FPS landmarks ($K_0$={K0})",
    )
    ax_left.set_title(f"FPS Teacher Landmarks ($K_0$={K0})", fontsize=11)
    ax_left.set_xlabel("Longitude")
    ax_left.set_ylabel("Latitude")

    # ---- Right panel: AAC Selected Landmarks ----
    _draw_edges(ax_right)

    # Forward landmarks: filled circles, AAC vermillion.  Smaller markers
    # than earlier versions so the road-network background reads as context, not as a
    # competing data layer.
    ax_right.scatter(
        lon[fwd_vertex_ids], lat[fwd_vertex_ids],
        s=34, c=[_AAC_COLOR], marker="o",
        edgecolors="black", linewidths=0.7, zorder=5,
    )
    # Backward landmarks: open triangles, same AAC hue (one method, two
    # directions, marker-shape carries the directionality so the panel
    # no longer reads as ``two methods'').
    ax_right.scatter(
        lon[bwd_vertex_ids], lat[bwd_vertex_ids],
        s=46, facecolors="none", edgecolors=_AAC_COLOR,
        linewidths=1.3, marker="^", zorder=6,
    )

    ax_right.set_title(f"AAC Selected Landmarks ($m$={M})", fontsize=11)
    ax_right.set_xlabel("Longitude")
    ax_right.set_ylabel("Latitude")

    # Shared horizontal legend below both panels.  Three entries (FPS pool,
    # AAC fwd, AAC bwd); the dropped count goes in the caption.  Marker
    # sizes match the in-panel sizes so the legend reads accurately.
    fps_handle = mlines.Line2D(
        [], [], marker="o", color="w", markerfacecolor=_FPS_COLOR,
        markeredgecolor="black", markersize=5.5,
        label=f"FPS teacher pool ($K_0{{=}}{K0}$)",
    )
    fwd_handle = mlines.Line2D(
        [], [], marker="o", color="w", markerfacecolor=_AAC_COLOR,
        markeredgecolor="black", markersize=6,
        label=f"AAC forward ($m_{{\\mathrm{{fwd}}}}{{=}}{M // 2}$)",
    )
    bwd_handle = mlines.Line2D(
        [], [], marker="^", color="w", markerfacecolor="none",
        markeredgecolor=_AAC_COLOR, markeredgewidth=1.3, markersize=7,
        label=f"AAC backward ($m_{{\\mathrm{{bwd}}}}{{=}}{M - M // 2}$)",
    )
    fig.legend(
        handles=[fps_handle, fwd_handle, bwd_handle],
        loc="outside lower center", ncol=3, fontsize=8, framealpha=0.9,
        edgecolor="0.8", columnspacing=1.4, handletextpad=0.4,
    )

    # Crop: trim the empty western/eastern margins of NY state by
    # cropping to the central interquartile longitude band (most of the
    # population/edge density lives there); latitude crop is symmetric.
    lon_lo, lon_hi = np.quantile(lon[fps_ids], [0.0, 1.0])
    lat_lo, lat_hi = np.quantile(lat[fps_ids], [0.0, 1.0])
    lon_pad = (lon_hi - lon_lo) * 0.04
    lat_pad = (lat_hi - lat_lo) * 0.04
    for ax in (ax_left, ax_right):
        ax.set_xlim(lon_lo - lon_pad, lon_hi + lon_pad)
        ax.set_ylim(lat_lo - lat_pad, lat_hi + lat_pad)
        ax.set_aspect("equal")

    # Save
    outpath = Path(output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), format="pdf")
    plt.close(fig)
    print(f"Saved landmark visualization to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate landmark selection visualization on DIMACS NY"
    )
    parser.add_argument(
        "--output",
        default="paper/figures/landmark_selection.pdf",
        help="Output PDF path (default: paper/figures/landmark_selection.pdf,"
             " the canonical location consumed by paper/main.tex).",
    )
    args = parser.parse_args()
    main(args.output)
