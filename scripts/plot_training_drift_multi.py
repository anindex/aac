"""Plot training-drift multi-graph sweep into paper/figures/training_drift.pdf.

Reads results/training_drift_multi/drift_{graph}_B{budget}.csv and produces a
2x2 grid (rows = graph, cols = budget). Each panel plots
``forced-first-m  -  AAC(epoch)`` (the gap to the architectural ceiling), in
percentage points, with the area between the curve and y=0 shaded as
``headroom not captured.''  All four panels share the same y-range, making
the drift story (``trained selector never closes the gap'') a single visual
gestalt instead of four panels with mutually-incompatible y-ranges.

The forced-first-m and FPS-ALT K=m references coincide algebraically (the
forced-first-m identity in Section 5.7); the new y=0 baseline IS that
reference, so they no longer need separate legend entries.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "training_drift_multi"
OUT_PDF = ROOT / "paper" / "figures" / "training_drift.pdf"

# Adopt the publication style (Computer Modern Roman, real LaTeX rendering
# when available). All paper figures share this style for consistency.
sys.path.insert(0, str(ROOT / "src"))
from aac.viz.style import (  # noqa: E402
    METHOD_COLORS,
    OKABE_ITO,
    TMLR_FULL_WIDTH,
    setup_style,
)

setup_style()

GRAPHS = ["sbm", "ba"]
BUDGETS = [32, 64]
GRAPH_LABEL = {"sbm": r"SBM ($5{\times}2000$)", "ba": r"BA ($10{,}000$, $m{=}5$)"}


def load(graph: str, budget: int):
    path = DATA_DIR / f"drift_{graph}_B{budget}.csv"
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    by_epoch = defaultdict(list)
    for r in rows:
        by_epoch[int(r["epochs"])].append(float(r["expansion_reduction_pct"]))
    epochs = sorted(by_epoch)
    means = np.array([np.mean(by_epoch[e]) for e in epochs])
    stds = np.array([np.std(by_epoch[e], ddof=1) if len(by_epoch[e]) > 1 else 0.0 for e in epochs])
    forced = float(rows[0]["forced_first_m_pct"])
    alt = float(rows[0]["alt_ref_pct"])
    return np.array(epochs), means, stds, forced, alt


def main():
    # Tight panel spacing -- with sharex+sharey the inner ticks/labels are
    # suppressed, so wspace/hspace can be very small without crowding.
    fig, axes = plt.subplots(
        len(GRAPHS), len(BUDGETS),
        figsize=(TMLR_FULL_WIDTH, TMLR_FULL_WIDTH * 0.52),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.06, "hspace": 0.18},
    )
    aac_color = METHOD_COLORS["aac"]
    headroom_color = OKABE_ITO["vermillion"]
    aac_handle = headroom_handle = None
    for i, g in enumerate(GRAPHS):
        for j, b in enumerate(BUDGETS):
            ax = axes[i, j]
            ep, mu, sd, forced, _alt = load(g, b)
            # Reframing: gap = forced-first-m  -  AAC(epoch).  Positive values
            # mean ``below the architectural ceiling''; the trained selector
            # is consistently above zero across all 4 panels.
            gap_mean = forced - mu
            gap_sd = sd  # std propagates one-to-one under translation by a constant.
            (line,) = ax.plot(
                ep, gap_mean, "-o", color=aac_color, lw=1.4, ms=4,
                label="AAC gap to ceiling",
            )
            ax.fill_between(ep, gap_mean - gap_sd, gap_mean + gap_sd,
                            color=aac_color, alpha=0.18)
            # ``Headroom not captured'' shading between the gap curve and y=0
            # baseline (negative direction = below ceiling = wasted budget).
            zero = np.zeros_like(gap_mean)
            headroom_poly = ax.fill_between(
                ep, zero, gap_mean,
                where=gap_mean > 0,
                interpolate=True,
                color=headroom_color, alpha=0.10, linewidth=0,
                label="Headroom not captured",
            )
            # Architectural ceiling (forced-first-m = FPS-ALT $K{=}m$) lives
            # at gap = 0 by construction; mark it with a thin black line.
            ax.axhline(0.0, color="black", lw=0.7, alpha=0.6)
            # Per-panel callout of the absolute matched-memory reference,
            # so the gap is interpretable in original units.
            # ``ceiling = 89.95'' (no %% sign because the y-axis is in
            # percentage-points already; the absolute reference value is
            # what matters for interpreting the gap-to-ceiling magnitude).
            # Pinned to the top-left interior corner so it does not collide
            # with the AAC trajectory or the headroom shading.
            ax.text(
                0.03, 0.95,
                f"ceiling = {forced:.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color="black", alpha=0.7,
                bbox=dict(facecolor="white", edgecolor="lightgray",
                          boxstyle="round,pad=0.2", linewidth=0.3),
            )
            ax.set_title(f"{GRAPH_LABEL[g]}, $B{{=}}{b}$ bytes/v", fontsize=9)
            if i == len(GRAPHS) - 1:
                ax.set_xlabel("Training epochs")
            if j == 0:
                ax.set_ylabel(r"Gap to ceiling (pp)")
            if aac_handle is None:
                aac_handle = line
                headroom_handle = headroom_poly
    # Symmetric y-range so all 4 panels share the same gestalt.  The drift
    # story (``trained selector never closes the gap'') reads as a single
    # picture: every curve is above zero throughout.
    for ax in axes.ravel():
        ax.set_ylim(-0.5, 2.0)
    # Single shared legend at the bottom replaces 4 redundant per-panel ones.
    fig.legend(
        [aac_handle, headroom_handle],
        [r"AAC trained (mean $\pm$ 1 sd over seeds)",
         "Headroom not captured (gap above ceiling)"],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
