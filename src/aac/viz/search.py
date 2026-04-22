"""Search behavior visualization: expansion heatmaps, comparison panels, contour maps.

Provides reusable plotting functions for visualizing A* search behavior.
CLI scripts in ``scripts/`` call these functions to produce TMLR-quality figures.

All functions accept a matplotlib ``Axes`` and return a ``ScalarMappable``
for colorbar creation.  Colormaps, marker styles, and figure sizing follow
the project's standard publication style.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from aac.viz.style import METHOD_LABELS, TMLR_FULL_WIDTH

if TYPE_CHECKING:
    from matplotlib.cm import ScalarMappable
    from aac.search.types import SearchResult


# ---------------------------------------------------------------------------
# Expansion heatmap (road network, scatter-based)
# ---------------------------------------------------------------------------


def plot_expansion_heatmap(
    ax: plt.Axes,
    coords: np.ndarray,
    expanded_nodes: list[int],
    path: list[int],
    source: int,
    target: int,
    *,
    all_nodes: bool = True,
    cmap: str = "YlOrRd",
    node_size: float = 0.5,
    bg_node_size: float = 0.3,
    path_linewidth: float = 1.5,
    norm: mcolors.Normalize | None = None,
) -> ScalarMappable:
    """Plot expansion heatmap on a road network using scatter markers.

    Expanded nodes are colored by expansion order (early = light yellow,
    late = dark red) using the ``YlOrRd`` colormap.  Unexpanded nodes are
    shown as faint outline dots.  The shortest path is drawn as a bold
    black line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the plot.
    coords : ndarray of shape (V, 2)
        Node coordinates (e.g., longitude, latitude).
    expanded_nodes : list[int]
        Node IDs in expansion order.
    path : list[int]
        Shortest path node sequence.
    source, target : int
        Source and target node IDs.
    all_nodes : bool
        If True, draw all nodes as faint background outlines.
    cmap : str
        Colormap name for expanded nodes.
    node_size : float
        Marker size for expanded nodes.
    bg_node_size : float
        Marker size for background nodes.
    path_linewidth : float
        Line width for the shortest path overlay.
    norm : matplotlib.colors.Normalize or None
        Shared color normalization.  If None, auto-created from data.

    Returns
    -------
    ScalarMappable
        The expanded-nodes scatter, usable for colorbar creation.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    if norm is None:
        norm = mcolors.Normalize(vmin=0, vmax=max(len(expanded_nodes) - 1, 1))

    # Background: all nodes as faint outlines
    if all_nodes:
        ax.scatter(
            x,
            y,
            s=bg_node_size,
            facecolors="none",
            edgecolors="#D0D0D0",
            linewidths=0.3,
            alpha=0.5,
            rasterized=True,
        )

    # Expanded nodes colored by expansion order
    exp = np.array(expanded_nodes)
    exp_order = np.arange(len(expanded_nodes))
    sc = ax.scatter(
        x[exp],
        y[exp],
        s=node_size,
        c=exp_order,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )

    # Shortest path overlay
    path_arr = np.array(path)
    ax.plot(
        x[path_arr],
        y[path_arr],
        color="#000000",
        linewidth=path_linewidth,
        zorder=5,
    )

    # Source marker: green circle
    ax.plot(
        x[source],
        y[source],
        "o",
        color="green",
        markersize=6,
        zorder=6,
    )

    # Target marker: red star
    ax.plot(
        x[target],
        y[target],
        "*",
        color="red",
        markersize=8,
        zorder=6,
    )

    # Reduce tick clutter
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.tick_params(labelsize=8)

    return sc


# ---------------------------------------------------------------------------
# Expansion heatmap (grid, imshow-based)
# ---------------------------------------------------------------------------


def plot_expansion_heatmap_grid(
    ax: plt.Axes,
    H: int,
    W: int,
    expanded_nodes: list[int],
    path: list[int],
    source: int,
    target: int,
    *,
    cmap: str = "YlOrRd",
    path_linewidth: float = 2.0,
    norm: mcolors.Normalize | None = None,
) -> ScalarMappable:
    """Plot expansion heatmap on a grid graph using imshow.

    Expanded cells are colored by expansion order; unexpanded cells are
    transparent (NaN).  Uses ``divmod(node_id, W)`` for coordinate
    conversion.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the plot.
    H, W : int
        Grid height and width.
    expanded_nodes : list[int]
        Node IDs in expansion order.
    path : list[int]
        Shortest path node sequence.
    source, target : int
        Source and target node IDs.
    cmap : str
        Colormap name.
    path_linewidth : float
        Line width for the shortest path overlay.
    norm : matplotlib.colors.Normalize or None
        Shared color normalization.  If None, auto-created from data.

    Returns
    -------
    ScalarMappable
        The AxesImage, usable for colorbar creation.
    """
    if norm is None:
        norm = mcolors.Normalize(vmin=0, vmax=max(len(expanded_nodes) - 1, 1))

    # Build 2D expansion-order grid (NaN for unexpanded)
    grid = np.full((H, W), np.nan)
    for order, node_id in enumerate(expanded_nodes):
        r, c = divmod(node_id, W)
        grid[r, c] = order

    im = ax.imshow(
        grid,
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        norm=norm,
    )

    # Path overlay: convert node IDs to (col, row) coordinates
    path_cols = [node_id % W for node_id in path]
    path_rows = [node_id // W for node_id in path]
    ax.plot(
        path_cols,
        path_rows,
        color="#000000",
        linewidth=path_linewidth,
        zorder=5,
    )

    # Source marker: green circle
    s_r, s_c = divmod(source, W)
    ax.plot(s_c, s_r, "o", color="green", markersize=8, zorder=6)

    # Target marker: red star
    t_r, t_c = divmod(target, W)
    ax.plot(t_c, t_r, "*", color="red", markersize=10, zorder=6)

    # Grid images: hide pixel-index ticks (meaningless for the reader)
    ax.set_xticks([])
    ax.set_yticks([])

    return im


# ---------------------------------------------------------------------------
# 3-panel comparison figure
# ---------------------------------------------------------------------------


def plot_comparison_panel(
    results: list[SearchResult],
    method_keys: list[str],
    *,
    coords: np.ndarray | None = None,
    grid_dims: tuple[int, int] | None = None,
    cmap: str = "YlOrRd",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Create a 1x3 comparison figure with shared colorbar.

    Compares node expansions across three methods (e.g., Dijkstra vs ALT
    vs AAC) using a shared color normalization so the visual
    difference in search effort is immediately obvious.

    Parameters
    ----------
    results : list of SearchResult
        Three SearchResult objects with populated ``expanded_nodes``.
    method_keys : list of str
        Three method keys for ``METHOD_LABELS`` lookup.
    coords : ndarray of shape (V, 2) or None
        Node coordinates for road-network rendering.
    grid_dims : (H, W) or None
        Grid dimensions for grid rendering.
    cmap : str
        Colormap name.
    figsize : (width, height) or None
        Figure size override.  Defaults to ``(TMLR_FULL_WIDTH, 2.5)``.

    Returns
    -------
    matplotlib.figure.Figure
        The completed comparison figure.

    Raises
    ------
    ValueError
        If neither ``coords`` nor ``grid_dims`` is provided.
    """
    if coords is None and grid_dims is None:
        raise ValueError("Either coords or grid_dims must be provided")

    if figsize is None:
        figsize = (TMLR_FULL_WIDTH, 2.2)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1, 1, 1, 0.04],
        wspace=0.08,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[0, 3])

    # Validate inputs
    if not results:
        raise ValueError("results must contain at least one SearchResult")
    for r in results:
        if r.expanded_nodes is None or r.path is None:
            raise ValueError(
                "All SearchResult objects must have track_expansions=True "
                "(expanded_nodes and path must not be None)"
            )

    # Shared norm: max expansions across all methods so panels are comparable
    max_expansions = max(len(r.expanded_nodes) for r in results)  # type: ignore[arg-type]
    shared_norm = mcolors.Normalize(vmin=0, vmax=max(max_expansions - 1, 1))

    mappable = None
    for i, (result, key) in enumerate(zip(results, method_keys)):
        ax = axes[i]
        expanded = result.expanded_nodes
        path = result.path

        # Determine source / target from path
        src = path[0] if path else 0
        tgt = path[-1] if path else 0

        if grid_dims is not None:
            H, W = grid_dims
            sm = plot_expansion_heatmap_grid(
                ax,
                H,
                W,
                expanded,
                path,
                src,
                tgt,
                cmap=cmap,
                norm=shared_norm,
            )
        else:
            sm = plot_expansion_heatmap(
                ax,
                coords,
                expanded,
                path,
                src,
                tgt,
                cmap=cmap,
                norm=shared_norm,
            )

        mappable = sm

        # Panel title: method name + expansion count
        label = METHOD_LABELS.get(key, key)
        count = len(expanded)
        ax.set_title(f"{label} ({count:,} exp.)", fontsize=9, pad=4)

        # Remove y-axis labels/ticks on middle and right panels (shared scale)
        if i > 0:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        # Reduce tick label sizes on all panels
        ax.tick_params(labelsize=7)

    # Shared colorbar on the right column
    cbar = fig.colorbar(mappable, cax=cbar_ax, label="Expansion Order")
    cbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.04, right=0.93, bottom=0.05, top=0.92)

    return fig


# ---------------------------------------------------------------------------
# Heuristic contour map (road network, tricontourf)
# ---------------------------------------------------------------------------


def plot_heuristic_contour(
    ax: plt.Axes,
    coords: np.ndarray,
    h_values: np.ndarray,
    path: list[int],
    target: int,
    *,
    cmap: str = "Blues",
    levels: int = 30,
    alpha: float = 0.9,
    path_color: str = "#D55E00",
    path_linewidth: float = 2.0,
    subsample_step: int | None = None,
) -> object:
    """Plot heuristic quality contour map on a road network.

    Uses ``matplotlib.tri.Triangulation`` and ``tricontourf`` for
    smooth filled contours over irregularly spaced nodes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the plot.
    coords : ndarray of shape (V, 2)
        Node coordinates.
    h_values : ndarray of shape (V,)
        Heuristic values ``h(v, target)`` for all nodes.
    path : list[int]
        Shortest path node sequence.
    target : int
        Target node ID.
    cmap : str
        Colormap for contour fill.
    levels : int
        Number of contour levels.
    alpha : float
        Contour transparency.
    path_color : str
        Color for the path overlay line.
    path_linewidth : float
        Width for the path overlay line.
    subsample_step : int or None
        Step size for subsampling nodes.  If None, auto-computed as
        ``max(1, V // 20000)``.

    Returns
    -------
    object
        The tricontourf mappable, usable for colorbar creation.
    """
    import matplotlib.tri as tri

    x = coords[:, 0]
    y = coords[:, 1]
    h = np.asarray(h_values)

    # Subsample for performance
    if subsample_step is None:
        subsample_step = max(1, len(coords) // 20000)

    idx = np.arange(0, len(coords), subsample_step)

    triang = tri.Triangulation(x[idx], y[idx])
    tcf = ax.tricontourf(
        triang,
        h[idx],
        levels=levels,
        cmap=cmap,
        alpha=alpha,
    )

    # Path overlay
    path_arr = np.array(path)
    ax.plot(
        x[path_arr],
        y[path_arr],
        color=path_color,
        linewidth=path_linewidth,
        zorder=5,
    )

    # Target marker: gold star
    ax.plot(
        x[target],
        y[target],
        "*",
        color="#FFD700",
        markersize=12,
        markeredgecolor="black",
        zorder=6,
    )

    # Reduce tick clutter
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.tick_params(labelsize=8)

    return tcf


# ---------------------------------------------------------------------------
# Heuristic contour map (grid, imshow-based)
# ---------------------------------------------------------------------------


def plot_heuristic_contour_grid(
    ax: plt.Axes,
    H: int,
    W: int,
    h_values: np.ndarray,
    path: list[int],
    target: int,
    *,
    cmap: str = "Blues",
    alpha: float = 0.9,
    path_color: str = "#D55E00",
    path_linewidth: float = 2.0,
) -> ScalarMappable:
    """Plot heuristic quality map on a grid using imshow with bilinear interpolation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the plot.
    H, W : int
        Grid height and width.
    h_values : ndarray of shape (H*W,)
        Heuristic values ``h(v, target)`` for all grid nodes.
    path : list[int]
        Shortest path node sequence.
    target : int
        Target node ID.
    cmap : str
        Colormap for the heuristic landscape.
    alpha : float
        Image transparency.
    path_color : str
        Color for the path overlay line.
    path_linewidth : float
        Width for the path overlay line.

    Returns
    -------
    ScalarMappable
        The AxesImage, usable for colorbar creation.
    """
    h = np.asarray(h_values)
    grid = h.reshape(H, W)

    im = ax.imshow(
        grid,
        cmap=cmap,
        origin="upper",
        interpolation="bilinear",
        alpha=alpha,
    )

    # Path overlay
    path_cols = [node_id % W for node_id in path]
    path_rows = [node_id // W for node_id in path]
    ax.plot(
        path_cols,
        path_rows,
        color=path_color,
        linewidth=path_linewidth,
        zorder=5,
    )

    # Target marker: gold star
    t_r, t_c = divmod(target, W)
    ax.plot(
        t_c,
        t_r,
        "*",
        color="#FFD700",
        markersize=12,
        markeredgecolor="black",
        zorder=6,
    )

    # Grid images: hide pixel-index ticks (meaningless for the reader)
    ax.set_xticks([])
    ax.set_yticks([])

    return im
