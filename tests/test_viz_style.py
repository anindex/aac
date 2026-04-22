"""Tests for the consolidated visualization style module (aac.viz.style)."""

import matplotlib.pyplot as plt
import pytest


class TestVizStyle:
    def test_setup_style_sets_rcparams(self):
        """setup_style() sets expected rcParams for TMLR publication quality.

        Typography: 10pt body /
        9pt axes labels / 8pt ticks & legend, with mathtext.cm so we never
        need ``text.usetex=True`` and the silent ``%``-truncation footgun
        is structurally avoided.
        """
        from aac.viz.style import setup_style

        setup_style(use_latex=False)
        assert plt.rcParams["font.family"] == ["serif"]
        assert plt.rcParams["font.size"] == 10.0
        assert plt.rcParams["axes.titlesize"] == 10.0
        assert plt.rcParams["axes.labelsize"] == 9.0
        assert plt.rcParams["legend.fontsize"] == 8.0
        assert plt.rcParams["xtick.labelsize"] == 8.0
        assert plt.rcParams["ytick.labelsize"] == 8.0
        assert plt.rcParams["mathtext.fontset"] == "cm"
        assert plt.rcParams["figure.constrained_layout.use"] is True
        assert plt.rcParams["savefig.dpi"] == 300.0
        assert plt.rcParams["savefig.bbox"] == "tight"
        assert plt.rcParams["savefig.pad_inches"] == 0.05

    def test_setup_style_default_is_mathtext(self):
        """Default setup_style() uses mathtext, not text.usetex.

        Avoids silent ``%``-truncation under text.usetex; mathtext with
        Computer Modern is visually indistinguishable at body-text sizes.
        """
        from aac.viz.style import setup_style

        setup_style()
        assert plt.rcParams["text.usetex"] is False

    def test_setup_style_latex_detection(self):
        """setup_style uses shutil.which('latex') to gate LaTeX rendering."""
        import shutil

        from aac.viz.style import setup_style

        has_latex = shutil.which("latex") is not None
        setup_style(use_latex=True)
        assert plt.rcParams["text.usetex"] == has_latex
        # With use_latex=False, text.usetex should not be set to True
        setup_style(use_latex=False)
        assert plt.rcParams["text.usetex"] is False or plt.rcParams["text.usetex"] == False

    def test_method_colors_markers(self):
        """METHOD_COLORS and METHOD_MARKERS contain expected method keys."""
        from aac.viz.style import METHOD_COLORS, METHOD_MARKERS

        expected_keys = {
            "aac", "alt", "fastmap", "dijkstra", "datasp", "hybrid", "greedy_max",
        }
        assert set(METHOD_COLORS.keys()) == expected_keys
        assert set(METHOD_MARKERS.keys()) == expected_keys
        # Markers are single characters
        for v in METHOD_MARKERS.values():
            assert isinstance(v, str) and len(v) == 1

    def test_okabe_ito_palette(self):
        """OKABE_ITO is the canonical colorblind-safe palette."""
        from aac.viz.style import OKABE_ITO, METHOD_COLORS, PALETTE

        # Original Okabe & Ito 2008 spec; verifiable against the publication.
        assert OKABE_ITO["blue"] == "#0072B2"
        assert OKABE_ITO["vermillion"] == "#D55E00"
        assert OKABE_ITO["green"] == "#009E73"
        # METHOD_COLORS draws from OKABE_ITO so AAC / ALT match the palette.
        assert METHOD_COLORS["aac"] == OKABE_ITO["blue"]
        assert METHOD_COLORS["alt"] == OKABE_ITO["vermillion"]
        # Legacy positional access (PALETTE[i]) still works.
        assert PALETTE[0] == OKABE_ITO["blue"]
        assert PALETTE[1] == OKABE_ITO["vermillion"]

    def test_method_labels(self):
        """METHOD_LABELS maps method keys to human-readable display names."""
        from aac.viz.style import METHOD_LABELS

        assert METHOD_LABELS["aac"] == "AAC"
        assert METHOD_LABELS["alt"] == "ALT"
        assert METHOD_LABELS["fastmap"] == "FastMap"
        assert METHOD_LABELS["dijkstra"] == "Dijkstra"
        assert METHOD_LABELS["datasp"] == "DataSP"
        assert METHOD_LABELS["greedy_max"] == "Greedy-Max"

    def test_tmlr_width_constants(self):
        """TMLR width constants are correct inch values."""
        from aac.viz.style import TMLR_COLUMN_WIDTH, TMLR_FULL_WIDTH

        assert TMLR_COLUMN_WIDTH == 3.25
        assert TMLR_FULL_WIDTH == 6.75

    def test_setup_style_idempotent(self):
        """Calling setup_style twice produces the same rcParams."""
        from aac.viz.style import setup_style

        setup_style(use_latex=False)
        first = dict(plt.rcParams)
        setup_style(use_latex=False)
        second = dict(plt.rcParams)
        # Check key params are identical
        for key in ["font.size", "axes.labelsize", "savefig.dpi", "savefig.bbox"]:
            assert first[key] == second[key]
