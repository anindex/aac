#!/usr/bin/env python
"""Two-panel figure: gap-to-teacher vs covering radius on the toy P_7.

Reads ``results/toy_p7/highlight.csv`` produced by
``scripts/run_toy_p7_gap_vs_covering.py`` and renders a TMLR full-width
2-panel figure to ``paper/figures/toy_p7_divergence.pdf``:

  Panel A "Covering view" -- Two stacked copies of the path P_7. Selected
          landmark vertices are filled in their set color; the union of
          per-landmark coverage balls of radius r_2 is shaded as a
          single rectangle per row (avoids alpha-stacking artifacts
          where balls overlap). Row label and r_2 carry white bboxes
          so they remain readable when the band extends past the path
          range (S_gap's union spans [-3, 9]). S_cov wins (smaller r_2).
  Panel B "Gap view"      -- Same two stacked path copies. For S_cov, the
          single failing query class (1,5)/(5,1) is drawn as an arc
          floating above the path, with thin connector ticks down to
          the (s,t) vertices to anchor the arc visually. For S_gap, no
          arc appears; a short italic caption between the rows reports
          "every interior query is exact" (any l outside [s,t] gives
          the ALT bound exactly d(s,t) on a 1D path -- the geometric
          reason is given in the figure caption in paper/main.tex).
          E[gap] is annotated on the right. S_gap wins (smaller E[gap]).

Color mapping mirrors the canonical Okabe-Ito palette
(``src/aac/viz/style.py``): S_cov uses AAC-blue (#0072B2), S_gap uses
ALT-vermillion (#D55E00). These hues are perceptually aligned with the
``s1`` slate-blue and ``s2`` terracotta of ``paper/method_diagram.tex``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from aac.viz.style import OKABE_ITO, TMLR_FULL_WIDTH, setup_style  # noqa: E402

DEFAULT_INPUT_DIR = _PROJECT_ROOT / "results" / "toy_p7"
DEFAULT_OUTPUT = _PROJECT_ROOT / "paper" / "figures" / "toy_p7_divergence.pdf"

N_VERTICES = 7
QUERY_SUPPORT = tuple(range(1, 6))  # {1..5}

COLOR_COV = OKABE_ITO["blue"]        # #0072B2 -- matches AAC color
COLOR_GAP = OKABE_ITO["vermillion"]  # #D55E00 -- matches ALT color
COLOR_INK = "#1A1A1A"
COLOR_MUTE = "#6B6B6B"


# ---------------------------------------------------------------------
# Data helpers (closed-form on P_7; same logic as the data generator)
# ---------------------------------------------------------------------

def _load_highlight(path: Path) -> dict[str, dict]:
    """Returns {'S_cov': {...}, 'S_gap': {...}} from highlight.csv."""
    out: dict[str, dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["name"]] = {
                "subset": (int(row["l1"]), int(row["l2"])),
                "r_cov": int(row["r_cov"]),
                "exp_gap": float(row["exp_gap"]),
                "exp_gap_num": int(row["exp_gap_num"]),
                "exp_gap_den": int(row["exp_gap_den"]),
                "n_exact": int(row["n_exact_queries"]),
            }
    return out


def _alt_heuristic(S: tuple[int, int], s: int, t: int) -> int:
    return max(abs(abs(s - l) - abs(l - t)) for l in S)


def _covering_radius(S: tuple[int, int]) -> int:
    return max(min(abs(v - l) for l in S) for v in range(N_VERTICES))


def _coverage_union(S: tuple[int, int], r: int) -> list[tuple[float, float]]:
    """Returns sorted, merged list of (start, end) coverage intervals.

    The union of per-landmark balls ``[l - r, l + r]`` collapsed into
    minimal disjoint intervals, so a single rectangle per merged piece
    can be drawn without alpha-stacking artifacts.
    """
    intervals = sorted((l - r, l + r) for l in S)
    merged: list[tuple[float, float]] = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _failing_queries(S: tuple[int, int]) -> list[tuple[int, int, int]]:
    """List of (s, t, gap) for every interior query with gap > 0."""
    out: list[tuple[int, int, int]] = []
    for s in QUERY_SUPPORT:
        for t in QUERY_SUPPORT:
            if s == t:
                continue
            gap = abs(s - t) - _alt_heuristic(S, s, t)
            if gap > 0:
                out.append((s, t, gap))
    return out


def _format_egap(meta: dict) -> str:
    """E[gap] string with both fraction and decimal where the value is nonzero."""
    if meta["exp_gap_num"] == 0:
        return r"$\mathbb{E}[\mathrm{gap}] = 0$"
    return (rf"$\mathbb{{E}}[\mathrm{{gap}}] = "
            rf"\frac{{{meta['exp_gap_num']}}}{{{meta['exp_gap_den']}}}$")


# ---------------------------------------------------------------------
# Path-graph drawing primitive
# ---------------------------------------------------------------------

def _draw_path(ax, y: float, selected: set[int], fill_color: str,
               *, label_vertices: bool = False) -> None:
    """Draw P_7 horizontally at vertical position y.

    Selected vertices are filled in ``fill_color`` (large markers).
    Unselected vertices are white-filled circles (small markers).
    Edges are drawn as a single thick black line UNDER the vertices
    (zorder=1) but we use the ``solid_capstyle="round"`` to make the
    end nubs look intentional even if the markers extend past them.
    """
    ax.plot(range(N_VERTICES), [y] * N_VERTICES,
            color=COLOR_INK, linewidth=1.6,
            solid_capstyle="round", zorder=1)
    for v in range(N_VERTICES):
        if v in selected:
            ax.scatter([v], [y], s=170, color=fill_color,
                       edgecolor=COLOR_INK, linewidth=0.9, zorder=3)
        else:
            ax.scatter([v], [y], s=46, facecolor="white",
                       edgecolor=COLOR_INK, linewidth=0.8, zorder=2)
    if label_vertices:
        for v in range(N_VERTICES):
            ax.text(v, y - 0.45, str(v), ha="center", va="top",
                    fontsize=7.5, color=COLOR_INK)


# ---------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------

def _panel_coverage(ax, s_cov: tuple[int, int], s_gap: tuple[int, int]) -> None:
    """Panel (a): coverage balls of radius r_2 around each landmark."""
    ax.set_title(r"(a) Covering view: $S_{\mathrm{cov}}$ wins ($r_2$ smaller)",
                 loc="left", pad=4, fontsize=9.5)

    y_cov = 1.10
    y_gap = 0.0
    r_cov = _covering_radius(s_cov)
    r_gap = _covering_radius(s_gap)

    half_h = 0.22  # half-height of the coverage shading band.
    # Draw a single rectangle per merged interval to avoid alpha-stacking
    # artifacts where two per-landmark balls overlap (e.g., S_cov balls of
    # radius 2 around 2 and 4 overlap in [2, 4]; S_gap balls of radius 3
    # around 0 and 6 meet at 3). The caption explains the union notation.
    for s, e in _coverage_union(s_cov, r_cov):
        ax.add_patch(mpatches.Rectangle(
            (s, y_cov - half_h), e - s, 2 * half_h,
            facecolor=COLOR_COV, alpha=0.18, edgecolor="none", zorder=0))
    for s, e in _coverage_union(s_gap, r_gap):
        ax.add_patch(mpatches.Rectangle(
            (s, y_gap - half_h), e - s, 2 * half_h,
            facecolor=COLOR_GAP, alpha=0.18, edgecolor="none", zorder=0))

    _draw_path(ax, y_cov, set(s_cov), COLOR_COV, label_vertices=False)
    _draw_path(ax, y_gap, set(s_gap), COLOR_GAP, label_vertices=True)

    # The merged coverage bands extend past the path range (S_gap reaches
    # x in [-3, 9]); white bboxes behind the row/r_2 labels keep the
    # vermillion / blue text readable where it overlaps a same-colored
    # band.
    label_bbox = dict(facecolor="white", edgecolor="none", pad=1.0)
    ax.text(-0.5, y_cov, r"$S_{\mathrm{cov}}=\{2,4\}$",
            ha="right", va="center", fontsize=9, color=COLOR_COV,
            bbox=label_bbox, zorder=5)
    ax.text(-0.5, y_gap, r"$S_{\mathrm{gap}}=\{0,6\}$",
            ha="right", va="center", fontsize=9, color=COLOR_GAP,
            bbox=label_bbox, zorder=5)

    ax.text(N_VERTICES - 0.4, y_cov, rf"$r_2={r_cov}$",
            ha="left", va="center", fontsize=9, color=COLOR_COV,
            bbox=label_bbox, zorder=5)
    ax.text(N_VERTICES - 0.4, y_gap, rf"$r_2={r_gap}$",
            ha="left", va="center", fontsize=9, color=COLOR_GAP,
            bbox=label_bbox, zorder=5)

    ax.text((N_VERTICES - 1) / 2, -0.95,
            r"shaded bands $=$ coverage balls $\bigcup_{l\in S}[l-r_2,\,l+r_2]$",
            ha="center", va="top", fontsize=7.5, color=COLOR_MUTE,
            style="italic")

    ax.set_xlim(-2.5, N_VERTICES + 0.7)
    ax.set_ylim(-1.3, 1.85)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _panel_gap(ax, s_cov: tuple[int, int], s_gap: tuple[int, int],
               cov_meta: dict, gap_meta: dict) -> None:
    """Panel (b): failing-query arcs above each path copy.

    For S_cov: a single arc above the path traces the symmetric failing
    query class (1,5)/(5,1), labeled with its gap value.

    For S_gap: no failing arcs; the row carries a short caption
    "every interior query exact" and the figure caption explains why
    (any landmark outside [s,t] gives the ALT bound exactly d on a 1D
    path).
    """
    ax.set_title(r"(b) Gap view: $S_{\mathrm{gap}}$ wins "
                 r"($\mathbb{E}[\mathrm{gap}]$ smaller)",
                 loc="left", pad=4, fontsize=9.5)

    y_cov = 1.10
    y_gap = 0.0

    _draw_path(ax, y_cov, set(s_cov), COLOR_COV, label_vertices=False)
    _draw_path(ax, y_gap, set(s_gap), COLOR_GAP, label_vertices=True)

    # Failing queries for S_cov: symmetric pair (1,5)/(5,1) with gap 2 each.
    cov_failures = _failing_queries(s_cov)
    if cov_failures:
        worst_s, worst_t, worst_gap = min(cov_failures,
                                          key=lambda r: (r[0], r[1]))
        if worst_t < worst_s:
            worst_s, worst_t = worst_t, worst_s

        # Lift the arc base slightly off the path so the endpoints don't
        # look like vertical "ticks" sticking out of the (s,t) vertices.
        arc_base = y_cov + 0.18
        arc_height = 0.45
        arc = mpatches.Arc(
            ((worst_s + worst_t) / 2, arc_base),
            width=worst_t - worst_s,
            height=2 * arc_height,
            angle=0, theta1=0, theta2=180,
            color=COLOR_COV, linewidth=1.4, zorder=4)
        ax.add_patch(arc)
        # Short thin connectors from arc endpoints down to the path
        # vertices, so the arc visibly anchors at (worst_s, t).
        for v in (worst_s, worst_t):
            ax.plot([v, v], [y_cov + 0.06, arc_base],
                    color=COLOR_COV, linewidth=1.0, alpha=0.7, zorder=4)
        ax.text((worst_s + worst_t) / 2, arc_base + arc_height + 0.04,
                rf"failing queries $({worst_s},{worst_t})$ and "
                rf"$({worst_t},{worst_s})$: gap $={worst_gap}$ each",
                ha="center", va="bottom", fontsize=7.5, color=COLOR_COV)

    # For S_gap, draw a faint check-mark style marker above the path
    # rather than failing arcs, to visually communicate "all exact".
    ax.text((N_VERTICES - 1) / 2, y_gap + 0.50,
            r"every interior query is exact "
            r"(all 20 have gap $= 0$)",
            ha="center", va="bottom", fontsize=7.5, color=COLOR_GAP,
            style="italic")

    ax.text(-0.5, y_cov, r"$S_{\mathrm{cov}}=\{2,4\}$",
            ha="right", va="center", fontsize=9, color=COLOR_COV)
    ax.text(-0.5, y_gap, r"$S_{\mathrm{gap}}=\{0,6\}$",
            ha="right", va="center", fontsize=9, color=COLOR_GAP)

    ax.text(N_VERTICES - 0.4, y_cov, _format_egap(cov_meta),
            ha="left", va="center", fontsize=9, color=COLOR_COV)
    ax.text(N_VERTICES - 0.4, y_gap, _format_egap(gap_meta),
            ha="left", va="center", fontsize=9, color=COLOR_GAP)

    ax.text((N_VERTICES - 1) / 2, -0.95,
            r"queries uniform on $\{1,\dots,5\}^2 \setminus \mathrm{diag}$",
            ha="center", va="top", fontsize=7.5, color=COLOR_MUTE,
            style="italic")

    ax.set_xlim(-2.5, N_VERTICES + 1.2)
    ax.set_ylim(-1.3, 1.95)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    highlight = _load_highlight(args.input_dir / "highlight.csv")
    s_cov = highlight["S_cov"]["subset"]
    s_gap = highlight["S_gap"]["subset"]

    setup_style(use_latex=False)

    fig = plt.figure(figsize=(TMLR_FULL_WIDTH, 2.25))
    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.18,
                          left=0.04, right=0.99, top=0.92, bottom=0.04)
    ax_cov = fig.add_subplot(gs[0, 0])
    ax_gap = fig.add_subplot(gs[0, 1])

    _panel_coverage(ax_cov, s_cov, s_gap)
    _panel_gap(ax_gap, s_cov, s_gap,
               highlight["S_cov"], highlight["S_gap"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    try:
        rel = args.output.relative_to(_PROJECT_ROOT)
        print(f"Wrote {rel}")
    except ValueError:
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
