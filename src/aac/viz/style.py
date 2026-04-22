"""Consolidated publication style for all AAC figures.

Single source of truth for matplotlib rcParams, the colorblind-safe palette,
method colors / markers, and TMLR figure sizing.  Every visualization script
and ``experiments/reporting/figures.py`` should import from here.

Palette
-------
The canonical palette is **Okabe-Ito** (Okabe & Ito, 2008), the de-facto
standard colorblind-safe qualitative palette in scientific publishing
(Nature Methods recommended, used across ML / TMLR papers).  All eight hues
are exported as :data:`OKABE_ITO`; :data:`PALETTE` aliases an ordered tuple
for legacy callers that index positionally.

Rendering
---------
:func:`setup_style` defaults to ``use_latex=False``: matplotlib's mathtext
engine with the Computer Modern font set is visually indistinguishable from
``\\usepackage{cm}`` LaTeX at body-text sizes, and avoids the silent ``%``
truncation footgun that ``text.usetex=True`` introduces.  Set
``use_latex=True`` explicitly to opt in to the real LaTeX pipeline (requires
a TeX distribution on PATH).
"""

from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette
# ---------------------------------------------------------------------------
#
# Reference: Okabe, M. & Ito, K. (2008). "Color Universal Design (CUD): How
# to make figures and presentations that are friendly to colorblind people."
# Each hex value is verifiable against the original specification.

OKABE_ITO: dict[str, str] = {
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "green":      "#009E73",
    "purple":     "#CC79A7",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "yellow":     "#F0E442",
    "black":      "#000000",
}

# Ordered tuple for legacy callers that index ``PALETTE[i]``.  Order chosen so
# the first three entries (blue, vermillion, green) carry the highest
# luminance contrast for the most common 2- and 3-method comparisons.
PALETTE: tuple[str, ...] = (
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["green"],
    OKABE_ITO["purple"],
    OKABE_ITO["orange"],
    OKABE_ITO["skyblue"],
    OKABE_ITO["yellow"],
    OKABE_ITO["black"],
)

METHOD_COLORS: dict[str, str] = {
    "aac":        OKABE_ITO["blue"],
    "alt":        OKABE_ITO["vermillion"],
    "fastmap":    OKABE_ITO["green"],
    "datasp":     OKABE_ITO["purple"],
    "hybrid":     OKABE_ITO["orange"],
    "dijkstra":   OKABE_ITO["black"],
    "greedy_max": OKABE_ITO["skyblue"],
}

METHOD_MARKERS: dict[str, str] = {
    "aac":        "o",
    "alt":        "s",
    "fastmap":    "D",
    "dijkstra":   "x",
    "datasp":     "P",
    "hybrid":     "^",
    "greedy_max": "v",
}

METHOD_LABELS: dict[str, str] = {
    "aac":        "AAC",
    "alt":        "ALT",
    "fastmap":    "FastMap",
    "dijkstra":   "Dijkstra",
    "datasp":     "DataSP",
    "hybrid":     "max(AAC, ALT)",
    "greedy_max": "Greedy-Max",
}

# Minimum font size for print readability (pts).
MIN_FONT_SIZE: int = 7

# ---------------------------------------------------------------------------
# TMLR figure dimensions (inches)
# ---------------------------------------------------------------------------

TMLR_COLUMN_WIDTH: float = 3.25
"""Single-column figure width for TMLR (inches)."""

TMLR_FULL_WIDTH: float = 6.75
"""Full-width (two-column) figure width for TMLR (inches)."""

DEFAULT_FIGSIZE: tuple[float, float] = (6.0, 4.0)
"""Default figure size used when no explicit override is given."""


def setup_style(*, use_latex: bool = False) -> None:
    """Configure matplotlib for publication-quality TMLR figures.

    Defaults to mathtext with the Computer Modern font set, which renders
    indistinguishably from real LaTeX at body-text sizes and avoids the
    ``text.usetex=True`` ``%``-comment footgun.  Pass ``use_latex=True`` to
    opt in to the LaTeX pipeline (requires ``latex`` on PATH).

    Parameters
    ----------
    use_latex : bool, default False
        Render text via the LaTeX subprocess (Computer Modern Roman).  Falls
        back to mathtext silently if no TeX distribution is found.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)

    rc: dict[str, object] = {
        # Typography (TMLR-friendly, slightly smaller than seaborn defaults so
        # 4-panel grids stay readable at single-column reproduction).
        "font.family":          "serif",
        "font.size":            10,
        "axes.titlesize":       10,
        "axes.labelsize":       9,
        "legend.fontsize":      8,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        # Math: Computer Modern via mathtext is visually identical to real
        # LaTeX at body-text sizes and works without a TeX distribution.
        "mathtext.fontset":     "cm",
        "mathtext.default":     "regular",
        # Layout & figure I/O.
        "figure.figsize":       DEFAULT_FIGSIZE,
        "figure.constrained_layout.use": True,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
        # Axes spines / grid (lighter than seaborn defaults to keep data ink
        # ratio high when many panels share a figure).
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "grid.linewidth":       0.4,
        "grid.alpha":           0.45,
        "lines.linewidth":      1.4,
        "lines.markersize":     4.5,
    }

    if use_latex and shutil.which("latex") is not None:
        rc["text.usetex"] = True
        rc["font.serif"] = ["Computer Modern Roman"]
    else:
        rc["text.usetex"] = False

    plt.rcParams.update(rc)
