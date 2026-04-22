"""Publication-quality figure generation for AAC experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from aac.viz.style import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    setup_style,
)
from aac.viz.style import (
    PALETTE as COLORBLIND_PALETTE,
)

# Re-export for backward compatibility
__all__ = [
    "METHOD_COLORS",
    "METHOD_MARKERS",
    "COLORBLIND_PALETTE",
    "setup_plot_style",
    "plot_pareto_frontier",
    "plot_compression_ratio",
    "plot_pareto_two_panel",
    "plot_latency_comparison",
]


def setup_plot_style() -> None:
    """Configure matplotlib for publication-quality plots.

    Delegates to the consolidated :func:`aac.viz.style.setup_style`.
    Kept as a thin wrapper for backward compatibility.
    """
    setup_style()


def _method_label(method: str) -> str:
    """Convert method key to display label using canonical METHOD_LABELS."""
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def plot_pareto_frontier(
    df: pd.DataFrame,
    output_pdf: str,
    x_metric: str = "memory_bytes_per_vertex",
    y_metric: str = "expansions_mean",
    graph_name: str | None = None,
) -> None:
    """Plot Pareto frontier of memory vs search quality for all methods.

    Args:
        df: Aggregated results DataFrame.
        output_pdf: Path for the output PDF file.
        x_metric: Column for x-axis (default: memory per vertex).
        y_metric: Column for y-axis (default: mean expansions).
        graph_name: If provided, filter to this graph only.
    """
    setup_plot_style()

    plot_df = df.copy()
    if graph_name is not None:
        plot_df = plot_df[plot_df["graph"] == graph_name]

    if plot_df.empty:
        return

    fig, ax = plt.subplots()

    for method in plot_df["method"].unique():
        method_df = plot_df[plot_df["method"] == method].sort_values(x_metric)
        marker = METHOD_MARKERS.get(method, "^")
        color = METHOD_COLORS.get(method, COLORBLIND_PALETTE[4])
        label = _method_label(method)

        ax.plot(
            method_df[x_metric],
            method_df[y_metric],
            marker=marker,
            color=color,
            label=label,
            linewidth=1.5,
            markersize=7,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Bytes per vertex")
    ax.set_ylabel("Mean node expansions")
    title = "Pareto frontier: memory vs search quality"
    if graph_name:
        title += f" ({graph_name})"
    ax.set_title(title)
    ax.legend(loc="best")

    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def plot_compression_ratio(
    df: pd.DataFrame,
    output_pdf: str,
    graph_name: str | None = None,
) -> None:
    """Plot gap-closing ratio vs m for each K0 (EXP-06).

    For AAC results, plots how the compression dimension *m*
    affects the gap-closing ratio ``1 - expansions / dijkstra_expansions``
    for each anchor count *K0*.

    Args:
        df: Aggregated results DataFrame.
        output_pdf: Path for the output PDF file.
        graph_name: If provided, filter to this graph only.
    """
    setup_plot_style()

    plot_df = df.copy()
    if graph_name is not None:
        plot_df = plot_df[plot_df["graph"] == graph_name]

    # Get Dijkstra baseline expansions per graph
    dijkstra_df = plot_df[plot_df["method"] == "dijkstra"]
    aac_df = plot_df[plot_df["method"] == "aac"]

    if aac_df.empty:
        return

    fig, ax = plt.subplots()

    # Compute gap-closing ratio per graph
    for graph in aac_df["graph"].unique():
        dij_row = dijkstra_df[dijkstra_df["graph"] == graph]
        if dij_row.empty:
            continue
        dij_exp = dij_row["expansions_mean"].values[0]
        if dij_exp == 0:
            continue

        graph_aac = aac_df[aac_df["graph"] == graph]
        k0_values = sorted(graph_aac["K0"].unique())

        for i, k0 in enumerate(k0_values):
            if k0 == 0:
                continue
            k0_data = graph_aac[graph_aac["K0"] == k0].sort_values("m")
            if k0_data.empty:
                continue

            gap_ratio = 1.0 - k0_data["expansions_mean"].values / dij_exp

            color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
            suffix = f" ({graph})" if graph_name is None else ""
            ax.plot(
                k0_data["m"].values,
                gap_ratio,
                marker="o",
                color=color,
                label=f"K0={k0}{suffix}",
                linewidth=1.5,
                markersize=6,
            )

    ax.set_xlabel("Compression dimension $m$")
    ax.set_ylabel("Gap-closing ratio")
    title = "Compression ratio analysis (EXP-06)"
    if graph_name:
        title += f" ({graph_name})"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(bottom=0)

    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def plot_pareto_two_panel(
    df_ny: pd.DataFrame,
    df_fla: pd.DataFrame,
    output_pdf: str,
) -> None:
    """Plot two-panel Pareto frontier (NY + FLA) of bytes/vertex vs expansion reduction.

    For AAC, plots the Pareto-optimal envelope (best reduction at each
    bytes_per_vertex level across K0 configurations). For ALT and FastMap,
    plots directly (one point per bytes_per_vertex).

    Args:
        df_ny: Pareto sweep DataFrame for NY graph.
        df_fla: Pareto sweep DataFrame for FLA graph.
        output_pdf: Path for the output PDF file.
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    panels = [
        (axes[0], df_ny, "(a) NY (264K nodes)"),
        (axes[1], df_fla, "(b) FLA (1.07M nodes)"),
    ]

    method_display = {
        "AAC": ("AAC", METHOD_MARKERS["aac"], METHOD_COLORS["aac"]),
        "ALT": ("ALT", METHOD_MARKERS["alt"], METHOD_COLORS["alt"]),
        "FastMap": ("FastMap", METHOD_MARKERS["fastmap"], METHOD_COLORS["fastmap"]),
    }

    for ax, df, title in panels:
        for method_key, (label, marker, color) in method_display.items():
            method_df = df[df["method"] == method_key].copy()
            if method_df.empty:
                continue

            if method_key == "AAC":
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
        ax.set_xlabel("Bytes per vertex")
        ax.set_ylim(0, 100)
        ax.set_title(title)

    axes[0].set_ylabel("Expansion reduction (\\%)")

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def plot_latency_comparison(
    df: pd.DataFrame,
    output_pdf: str,
) -> None:
    """Plot grouped bar chart of p50 latency by method and graph.

    Args:
        df: Aggregated results DataFrame.
        output_pdf: Path for the output PDF file.
    """
    setup_plot_style()

    if df.empty:
        return

    fig, ax = plt.subplots()

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

        color = METHOD_COLORS.get(method, COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)])
        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            latencies,
            bar_width,
            label=_method_label(method),
            color=color,
        )

    ax.set_xlabel("Graph")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("Query latency comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(graphs)
    ax.legend(loc="best")

    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)
