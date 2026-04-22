"""Experimental results visualization: Pareto frontier, ablation bar, latency bars, dashboard.

Provides reusable plotting functions for presenting quantitative results from
AAC experiments.  CLI scripts in ``scripts/`` call these functions to
produce TMLR-quality figures.

All functions accept a matplotlib ``Axes`` (or DataFrames for the dashboard)
and follow the composable pattern established by :mod:`aac.viz.search` and
:mod:`aac.viz.compression`.  No file I/O inside plotting functions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from aac.viz.style import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    PALETTE,
    TMLR_FULL_WIDTH,
)

__all__ = [
    "PARETO_METHOD_MAP",
    "ABLATION_LABELS",
    "ABLATION_COLORS",
    "plot_pareto_frontier",
    "plot_ablation_bar",
    "plot_latency_bars",
    "plot_results_dashboard",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARETO_METHOD_MAP: dict[str, str] = {
    "AAC": "aac",
    "ALT": "alt",
    "FastMap": "fastmap",
    "Dijkstra": "dijkstra",
}
"""Maps CSV method names to :mod:`aac.viz.style` keys."""

ABLATION_LABELS: dict[str, str] = {
    "full_e2e": "Full E2E",
    "frozen_compressor": "Frozen\nCompressor",
    "frozen_encoder": "Frozen\nEncoder",
    "datasp_published": "DataSP",
}
"""Display labels for ablation methods."""

ABLATION_COLORS: dict[str, tuple[float, ...]] = {
    "full_e2e": PALETTE[0],
    "frozen_compressor": PALETTE[1],
    "frozen_encoder": PALETTE[2],
    "datasp_published": PALETTE[3],
}
"""Colors for ablation bar chart bars."""


# ---------------------------------------------------------------------------
# RESULT-01: Pareto frontier plot
# ---------------------------------------------------------------------------


def plot_pareto_frontier(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str = "",
    show_legend: bool = True,
) -> None:
    """Plot Pareto frontier of memory budget vs expansion reduction.

    For AAC, computes the Pareto-optimal envelope (best reduction at each
    ``bytes_per_vertex`` across K0 configurations).  For ALT and FastMap,
    plots directly.  Dijkstra (0% baseline) is skipped.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the plot.
    df : pandas.DataFrame
        DataFrame with columns: ``method``, ``bytes_per_vertex``,
        ``expansion_reduction_pct``.
    title : str
        Panel title (e.g., ``"(a) NY (264K nodes)"``).
    show_legend : bool
        Whether to show the legend.
    """
    for csv_key, style_key in PARETO_METHOD_MAP.items():
        method_df = df[df["method"] == csv_key].copy()
        if method_df.empty or csv_key == "Dijkstra":
            continue

        color = METHOD_COLORS[style_key]
        marker = METHOD_MARKERS[style_key]
        label = METHOD_LABELS[style_key]

        if csv_key == "AAC":
            # Pareto-optimal envelope: best reduction at each bytes_per_vertex
            envelope = (
                method_df.groupby("bytes_per_vertex")["expansion_reduction_pct"]
                .max()
                .reset_index()
                .sort_values("bytes_per_vertex")
            )
            ax.plot(
                envelope["bytes_per_vertex"],
                envelope["expansion_reduction_pct"],
                marker=marker,
                color=color,
                label=label,
                linewidth=1.5,
                markersize=7,
            )
        else:
            method_df = method_df.sort_values("bytes_per_vertex")
            ax.plot(
                method_df["bytes_per_vertex"],
                method_df["expansion_reduction_pct"],
                marker=marker,
                color=color,
                label=label,
                linewidth=1.5,
                markersize=7,
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Bytes per vertex", fontsize=10)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    if title:
        ax.set_title(title, fontsize=10)
    if show_legend:
        ax.legend(fontsize=8, loc="lower right")


# ---------------------------------------------------------------------------
# RESULT-02: Ablation bar chart
# ---------------------------------------------------------------------------


def plot_ablation_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str = "cost_regret",
    *,
    errors: dict[str, float] | None = None,
    title: str = "Cost Regret Comparison",
) -> None:
    """Plot ablation bar chart comparing cost regret across methods.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the plot.
    df : pandas.DataFrame
        DataFrame with columns: ``method``, ``{metric}``.
    metric : str
        Metric column name (default ``"cost_regret"``).
    errors : dict or None
        Optional error values per method key (e.g., ``{"full_e2e": 0.05}``).
        Methods not in the dict get zero error.
    title : str
        Panel title.
    """
    methods = [m for m in ABLATION_LABELS if m in df["method"].values]
    x = np.arange(len(methods))
    values = [df[df["method"] == m][metric].mean() for m in methods]  # .mean() handles multi-row
    colors = [ABLATION_COLORS[m] for m in methods]
    labels = [ABLATION_LABELS[m] for m in methods]

    yerr = None
    if errors:
        yerr = [errors.get(m, 0) for m in methods]

    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        yerr=yerr,
        capsize=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    metric_label = metric.replace("_", " ").title()
    ax.set_ylabel(metric_label, fontsize=10)

    if title:
        ax.set_title(title, fontsize=10)

    # Annotate bar values
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )


# ---------------------------------------------------------------------------
# RESULT-03 (partial): Latency bars
# ---------------------------------------------------------------------------


def plot_latency_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str = "Query Latency (p50)",
    memory_budget: dict[str, str] | None = None,
) -> None:
    """Plot grouped bar chart of p50 latency by method and graph.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the plot.
    df : pandas.DataFrame
        DataFrame with columns: ``graph``, ``method``, ``p50_ms``.
    title : str
        Panel title.
    memory_budget : dict or None
        Optional mapping from method key to budget string
        (e.g., ``{"aac": "64B/v", "alt": "128B/v"}``).
        Shown in legend labels for memory context.
    """
    graphs = sorted(df["graph"].unique())
    methods = sorted(df["method"].unique())
    n_methods = len(methods)
    x = np.arange(len(graphs))
    bar_width = 0.8 / max(n_methods, 1)

    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        latencies = []
        for graph in graphs:
            graph_method = method_df[method_df["graph"] == graph]
            if not graph_method.empty:
                latencies.append(graph_method["p50_ms"].values[0])
            else:
                latencies.append(0)

        color = METHOD_COLORS.get(method, PALETTE[i % len(PALETTE)])
        label = METHOD_LABELS.get(method, method.replace("_", " ").title())

        # Append memory budget to legend label if provided
        if memory_budget and method in memory_budget:
            label = f"{label} ({memory_budget[method]})"

        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            latencies,
            bar_width,
            label=label,
            color=color,
        )

    ax.set_xlabel("Graph", fontsize=10)
    ax.set_ylabel("p50 latency (ms)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(graphs, fontsize=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.legend(fontsize=8)

    if title:
        ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------------
# RESULT-03: Multi-panel dashboard
# ---------------------------------------------------------------------------


def plot_results_dashboard(
    df_ny: pd.DataFrame,
    df_fla: pd.DataFrame,
    df_ablation: pd.DataFrame,
    df_timing: pd.DataFrame,
    *,
    memory_budget: dict[str, str] | None = None,
) -> plt.Figure:
    """Create a 4-subplot dashboard combining all result visualizations.

    Layout:
    - Top row: Pareto frontier for NY (left) and FLA (right), shared y-axis.
    - Bottom row: Ablation bar chart (left) and latency bars (right).

    Parameters
    ----------
    df_ny : pandas.DataFrame
        Pareto sweep data for NY graph.
    df_fla : pandas.DataFrame
        Pareto sweep data for FLA graph.
    df_ablation : pandas.DataFrame
        Ablation results (method, cost_regret).
    df_timing : pandas.DataFrame
        Timing data (graph, method, p50_ms).
    memory_budget : dict or None
        Optional memory budget annotations for latency panel.

    Returns
    -------
    matplotlib.figure.Figure
        The completed dashboard figure.
    """
    fig = plt.figure(figsize=(TMLR_FULL_WIDTH, 5.5))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 0.85],
        hspace=0.38,
        wspace=0.30,
    )

    # Top row: Pareto frontier panels
    ax_ny = fig.add_subplot(gs[0, 0])
    ax_fla = fig.add_subplot(gs[0, 1], sharey=ax_ny)
    plot_pareto_frontier(ax_ny, df_ny, title="(a) NY (264K nodes)")
    plot_pareto_frontier(
        ax_fla, df_fla, title="(b) FLA (1.07M nodes)", show_legend=False
    )
    ax_ny.set_ylabel("Expansion reduction (\\%)", fontsize=10)

    # Bottom row: ablation + latency
    ax_ablation = fig.add_subplot(gs[1, 0])
    ax_latency = fig.add_subplot(gs[1, 1])
    plot_ablation_bar(ax_ablation, df_ablation, title="(c) Ablation: Cost Regret")
    plot_latency_bars(
        ax_latency,
        df_timing,
        title="(d) Query Latency (p50)",
        memory_budget=memory_budget,
    )

    return fig
