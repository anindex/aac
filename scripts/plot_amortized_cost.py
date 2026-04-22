#!/usr/bin/env python
"""Total amortized cost figure across all four DIMACS road networks.

For each method (AAC, FPS-ALT) at the canonical 64-B/v matched-memory
configuration on every DIMACS road graph (NY, BAY, COL, FLA), plots the
total wall-clock cost
    T_total(N) = T_offline + N * T_per_query
for query workloads N in [1, 1e5], with the breakeven N (the N at which
AAC's offline overhead is amortized by lower per-query latency) annotated
per panel.  The 2x2 layout makes the headline ``AAC is the cheaper
total-wall-clock choice on all four DIMACS graphs'' claim self-evident
at a glance instead of requiring the reader to chase prose breakeven
numbers (NY 1276; BAY 1924; COL 170; FLA 522).

Inputs are read at runtime from results/dimacs/timing_p50_p95.csv so the
figure stays in sync with the canonical timing CSV; the previous version
hard-coded stale paper-table numbers and went out of sync after the
matched-memory K=8 (vs K=16) timing fix.

Output: paper/figures/amortized_cost.pdf (full-width 2x2 panel)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from aac.viz.style import (  # noqa: E402
    METHOD_COLORS,
    TMLR_FULL_WIDTH,
    setup_style,
)

DEFAULT_GRAPHS = ("NY", "BAY", "COL", "FLA")
GRAPH_LABELS = {
    "NY":  "(a) NY (264K nodes)",
    "BAY": "(b) BAY (321K nodes)",
    "COL": "(c) COL (436K nodes)",
    "FLA": "(d) FLA (1.07M nodes)",
}


def _load_timing(csv_path: Path, graph: str) -> dict[str, dict[str, float]]:
    """Return {'aac': {...}, 'alt': {...}} for the named graph."""
    out: dict[str, dict[str, float]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["graph"] != graph:
                continue
            out[row["method"]] = {
                "p50_ms": float(row["p50_ms"]),
                "p95_ms": float(row["p95_ms"]),
                "preprocess_total_sec": float(row["preprocess_total_sec"]),
            }
    if "aac" not in out or "alt" not in out:
        raise RuntimeError(
            f"timing CSV {csv_path} is missing aac/alt rows for {graph!r}; "
            f"re-run scripts/run_timing_and_stats.py to regenerate it."
        )
    return out


def _amortized(N: np.ndarray, offline_s: float, per_query_ms: float) -> np.ndarray:
    return offline_s + N * (per_query_ms / 1000.0)


def _plot_panel(ax, csv_path: Path, graph: str, title: str) -> tuple[float, float]:
    """Plot one panel; return (breakeven_N, ylim_top) for the panel."""
    timing = _load_timing(csv_path, graph)
    aac_off = timing["aac"]["preprocess_total_sec"]
    aac_p50 = timing["aac"]["p50_ms"]
    alt_off = timing["alt"]["preprocess_total_sec"]
    alt_p50 = timing["alt"]["p50_ms"]

    N = np.logspace(0, 5, 200)
    aac_cost = _amortized(N, aac_off, aac_p50)
    alt = _amortized(N, alt_off, alt_p50)

    if alt_p50 > aac_p50:
        N_break = (aac_off - alt_off) / ((alt_p50 - aac_p50) / 1000.0)
    else:
        N_break = float("inf")

    aac_color = METHOD_COLORS["aac"]
    alt_color = METHOD_COLORS["alt"]
    ax.plot(N, aac_cost, color=aac_color, linewidth=1.6, label="AAC")
    ax.plot(N, alt, color=alt_color, linewidth=1.6, label="FPS-ALT")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10)

    # Per-panel inset: offline + p50 numbers (replaces the wordy legend
    # entries from the old single-panel version).
    ax.text(
        0.04, 0.96,
        (f"AAC: $T_\\mathrm{{off}}{{=}}{aac_off:.0f}$ s, "
         f"$p_{{50}}{{=}}{aac_p50:.1f}$ ms\n"
         f"ALT: $T_\\mathrm{{off}}{{=}}{alt_off:.1f}$ s, "
         f"$p_{{50}}{{=}}{alt_p50:.1f}$ ms"),
        transform=ax.transAxes, ha="left", va="top", fontsize=7,
        bbox=dict(facecolor="white", edgecolor="lightgray",
                  boxstyle="round,pad=0.25", linewidth=0.4),
    )

    if np.isfinite(N_break) and 1.0 <= N_break <= N[-1]:
        ax.axvline(N_break, color="gray", linestyle="--", linewidth=0.7)
        y_at_break = _amortized(np.array([N_break]), aac_off, aac_p50)[0]
        ax.annotate(
            f"$N_\\mathrm{{break}}{{\\approx}}{N_break:,.0f}$",
            xy=(N_break, y_at_break),
            xytext=(N_break * 1.5, y_at_break * 0.25),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )
    elif not np.isfinite(N_break):
        ax.text(
            0.97, 0.04,
            "ALT dominates at every $N$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray",
        )
    return N_break, max(aac_cost.max(), alt.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("results/dimacs/timing_p50_p95.csv"),
        help="Path to timing CSV produced by scripts/run_timing_and_stats.py.",
    )
    ap.add_argument(
        "--graphs",
        type=str,
        nargs="+",
        default=list(DEFAULT_GRAPHS),
        help="DIMACS graphs to plot in row-major order.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/amortized_cost.pdf"),
    )
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    setup_style()

    n = len(args.graphs)
    nrows = 2 if n > 2 else 1
    ncols = 2 if n > 1 else 1
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(TMLR_FULL_WIDTH, TMLR_FULL_WIDTH * 0.65),
        sharex=True, sharey=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    breakeven: dict[str, float] = {}
    for ax, graph in zip(axes_flat, args.graphs):
        title = GRAPH_LABELS.get(graph, graph)
        breakeven[graph], _ = _plot_panel(ax, args.csv, graph, title)
    for ax in axes_flat[len(args.graphs):]:
        ax.set_visible(False)

    # X-label on each bottom-row panel (avoids fig.supxlabel collision with
    # the outside-bottom-center legend); single shared y-label on the left
    # edge via fig.supylabel (constrained-layout reserves space cleanly).
    for ax in axes_flat[-ncols:]:
        ax.set_xlabel(r"Query workload $N$ (log scale)")
    fig.supylabel("Total wall-clock cost (s, log scale)", fontsize=9)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
    )
    fig.savefig(args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    for graph, N_b in breakeven.items():
        if np.isfinite(N_b):
            print(f"  {graph}: breakeven N = {N_b:,.0f} queries")
        else:
            print(f"  {graph}: no finite breakeven (ALT is at least as fast)")


if __name__ == "__main__":
    main()
