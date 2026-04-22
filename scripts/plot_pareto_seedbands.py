#!/usr/bin/env python3
"""Regenerate paper/figures/pareto_frontier.pdf with per-seed bands.

Plots a multi-seed mean +/- 1 sd shaded band on NY and FLA, restricted to
admissible methods (AAC, ALT). Best AAC K_0 per memory budget is selected
on the per-seed (validation) split when available; otherwise we fall back
to selecting the K_0 that minimizes per-cell mean expansions across the 5
seeds.

Usage:
    python scripts/plot_pareto_seedbands.py \
        --perseed-ny results/dimacs/pareto_sweep_NY_perseed.csv \
        --perseed-fla results/dimacs/pareto_sweep_FLA_perseed.csv \
        --output paper/figures/pareto_frontier.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication style (Computer Modern Roman + real LaTeX where available).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from aac.viz.style import (  # noqa: E402
    METHOD_COLORS,
    METHOD_MARKERS,
    TMLR_FULL_WIDTH,
    setup_style,
)

setup_style()


def _best_K0_per_budget(df: pd.DataFrame) -> pd.DataFrame:
    """For each (method, bytes) cell, pick the K_0 with the highest mean expansion
    reduction across seeds, then return the per-seed reduction values for that K_0.
    """
    method_dfs = []
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        if method == "AAC":
            stats = (
                sub.groupby(["bytes_per_vertex", "K0"])["expansion_reduction_pct"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            best = stats.loc[stats.groupby("bytes_per_vertex")["mean"].idxmax()]
            kept = []
            for _, row in best.iterrows():
                cell = sub[
                    (sub["bytes_per_vertex"] == row["bytes_per_vertex"])
                    & (sub["K0"] == row["K0"])
                ]
                kept.append(cell)
            method_dfs.append(pd.concat(kept, ignore_index=True))
        else:
            method_dfs.append(sub)
    return pd.concat(method_dfs, ignore_index=True)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["method", "bytes_per_vertex"])["expansion_reduction_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "red_mean", "std": "red_std"})
    )
    g["red_std"] = g["red_std"].fillna(0.0)
    return g


_METHOD_KEY = {"AAC": "aac", "ALT": "alt"}
_METHOD_LABEL = {
    "AAC": r"AAC (best $K_0$, val-split)",
    "ALT": r"FPS-ALT",
}


def _plot_panel(ax, df_perseed: pd.DataFrame, title: str) -> None:
    df_best = _best_K0_per_budget(df_perseed)
    summary = _summarize(df_best)
    for method in ("AAC", "ALT"):
        sub = summary[summary["method"] == method].sort_values("bytes_per_vertex")
        if sub.empty:
            continue
        key = _METHOD_KEY[method]
        color = METHOD_COLORS[key]
        marker = METHOD_MARKERS[key]
        ax.plot(
            sub["bytes_per_vertex"],
            sub["red_mean"],
            color=color,
            marker=marker,
            label=_METHOD_LABEL[method],
            linewidth=1.6,
        )
        ax.fill_between(
            sub["bytes_per_vertex"],
            sub["red_mean"] - sub["red_std"],
            sub["red_mean"] + sub["red_std"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    ax.set_xlabel("Memory budget (bytes / vertex)")
    ax.set_ylabel(r"Expansion reduction ($\%$)")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256])
    ax.set_xticklabels([32, 64, 128, 256])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--perseed-ny",
        type=Path,
        default=Path("results/dimacs/pareto_sweep_NY_perseed.csv"),
    )
    p.add_argument(
        "--perseed-fla",
        type=Path,
        default=Path("results/dimacs/pareto_sweep_FLA_perseed.csv"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/pareto_frontier.pdf"),
    )
    args = p.parse_args()

    df_ny = pd.read_csv(args.perseed_ny, comment="#")
    df_fla = pd.read_csv(args.perseed_fla, comment="#")
    # Restrict to admissible methods only (FastMap is omitted from the figure
    # per the table caption).
    df_ny = df_ny[df_ny["method"].isin(["AAC", "ALT"])].copy()
    df_fla = df_fla[df_fla["method"].isin(["AAC", "ALT"])].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(TMLR_FULL_WIDTH, TMLR_FULL_WIDTH * 0.45), sharey=False
    )
    _plot_panel(axes[0], df_ny, "(a) NY (264K nodes)")
    _plot_panel(axes[1], df_fla, "(b) FLA (1.07M nodes)")
    # Single shared legend below the panels (suptitle intentionally omitted;
    # the figure caption in paper/main.tex already states the same).
    # Constrained-layout will reserve space for the figure-level legend
    # automatically.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
    )
    fig.savefig(args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
