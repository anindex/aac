"""Tests for experimental results visualization functions (aac.viz.results)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from aac.viz.style import METHOD_COLORS, PALETTE  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pareto_df() -> pd.DataFrame:
    """Create synthetic Pareto DataFrame with 3 methods and known values."""
    rows = [
        # AAC: two K0 configs at bytes_per_vertex=64, one at 32 and 128
        {"method": "AAC", "bytes_per_vertex": 32, "expansion_reduction_pct": 78.0},
        {"method": "AAC", "bytes_per_vertex": 64, "expansion_reduction_pct": 85.0},
        {"method": "AAC", "bytes_per_vertex": 64, "expansion_reduction_pct": 89.0},
        {"method": "AAC", "bytes_per_vertex": 128, "expansion_reduction_pct": 93.0},
        # ALT: one point per bytes_per_vertex
        {"method": "ALT", "bytes_per_vertex": 64, "expansion_reduction_pct": 87.0},
        {"method": "ALT", "bytes_per_vertex": 128, "expansion_reduction_pct": 94.0},
        {"method": "ALT", "bytes_per_vertex": 256, "expansion_reduction_pct": 96.0},
        # FastMap: one point per bytes_per_vertex
        {"method": "FastMap", "bytes_per_vertex": 32, "expansion_reduction_pct": 22.0},
        {"method": "FastMap", "bytes_per_vertex": 64, "expansion_reduction_pct": 25.0},
        {"method": "FastMap", "bytes_per_vertex": 128, "expansion_reduction_pct": 27.0},
    ]
    return pd.DataFrame(rows)


def _make_ablation_df() -> pd.DataFrame:
    """Create synthetic ablation DataFrame with 4 methods and known cost_regret."""
    rows = [
        {"method": "full_e2e", "cost_regret": 0.145},
        {"method": "frozen_compressor", "cost_regret": 0.097},
        {"method": "frozen_encoder", "cost_regret": 1.012},
        {"method": "datasp_published", "cost_regret": 0.173},
    ]
    return pd.DataFrame(rows)


def _make_timing_df() -> pd.DataFrame:
    """Create synthetic timing DataFrame with 2 graphs and 2 methods."""
    rows = [
        {"graph": "NY", "method": "aac", "p50_ms": 90.9},
        {"graph": "NY", "method": "alt", "p50_ms": 57.9},
        {"graph": "BAY", "method": "aac", "p50_ms": 118.3},
        {"graph": "BAY", "method": "alt", "p50_ms": 114.8},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestPlotParetoFrontier
# ---------------------------------------------------------------------------


class TestPlotParetoFrontier:
    def test_plot_pareto_frontier_renders(self):
        """Pareto frontier renders without error and axes has lines."""
        from aac.viz.results import plot_pareto_frontier

        fig, ax = plt.subplots()
        df = _make_pareto_df()
        plot_pareto_frontier(ax, df)
        # At least 2 lines: AAC envelope + ALT or FastMap
        assert len(ax.lines) >= 2, f"Expected >= 2 lines, got {len(ax.lines)}"
        plt.close(fig)

    def test_plot_pareto_frontier_aac_envelope(self):
        """AAC data uses max reduction at each bytes_per_vertex (Pareto envelope)."""
        from aac.viz.results import plot_pareto_frontier

        fig, ax = plt.subplots()
        df = _make_pareto_df()
        plot_pareto_frontier(ax, df)

        # Find AAC line (should be first plotted)
        # At bytes_per_vertex=64, there are two AAC rows (85 and 89).
        # The envelope should use 89 (the max).
        aac_line = ax.lines[0]
        y_data = aac_line.get_ydata()
        # The AAC envelope has 3 points: (32, 78), (64, 89), (128, 93)
        # At index 1 (bytes_per_vertex=64), the value should be 89, not 85
        assert 89.0 in y_data, f"Expected 89.0 in AAC envelope y-data, got {y_data}"
        assert 85.0 not in y_data, f"85.0 should not be in envelope, got {y_data}"
        plt.close(fig)

    def test_plot_pareto_frontier_log_scale(self):
        """After plotting, x-axis uses log scale."""
        from aac.viz.results import plot_pareto_frontier

        fig, ax = plt.subplots()
        df = _make_pareto_df()
        plot_pareto_frontier(ax, df)
        assert ax.get_xscale() == "log", f"Expected log scale, got {ax.get_xscale()}"
        plt.close(fig)

    def test_all_functions_use_style_colors(self):
        """plot_pareto_frontier uses METHOD_COLORS values from style.py."""
        from aac.viz.results import plot_pareto_frontier

        fig, ax = plt.subplots()
        df = _make_pareto_df()
        plot_pareto_frontier(ax, df)

        # Collect colors from plotted lines (may be hex strings or RGBA tuples)
        line_colors = []
        for line in ax.lines:
            c = line.get_color()
            if isinstance(c, str):
                line_colors.append(c.upper())
            else:
                line_colors.append(matplotlib.colors.to_hex(c).upper())

        # At least one line should use a PALETTE entry
        palette_hex = {c.upper() for c in PALETTE[:4]}
        found = any(lc in palette_hex for lc in line_colors)
        assert found, f"No line colors match PALETTE entries. Colors: {line_colors}"
        plt.close(fig)


# ---------------------------------------------------------------------------
# TestPlotAblationBar
# ---------------------------------------------------------------------------


class TestPlotAblationBar:
    def test_plot_ablation_bar_renders(self):
        """Ablation bar renders 4 bars (patches) without error."""
        from aac.viz.results import plot_ablation_bar

        fig, ax = plt.subplots()
        df = _make_ablation_df()
        plot_ablation_bar(ax, df)
        patches = [p for p in ax.patches]
        assert len(patches) == 4, f"Expected 4 bars, got {len(patches)}"
        plt.close(fig)

    def test_plot_ablation_bar_with_errors(self):
        """Ablation bar with error dict produces 4 bars without error."""
        from aac.viz.results import plot_ablation_bar

        fig, ax = plt.subplots()
        df = _make_ablation_df()
        plot_ablation_bar(ax, df, errors={"full_e2e": 0.05})
        patches = [p for p in ax.patches]
        assert len(patches) == 4, f"Expected 4 bars, got {len(patches)}"
        plt.close(fig)

    def test_plot_ablation_bar_value_annotations(self):
        """After plotting, axes has text annotations with bar values."""
        from aac.viz.results import plot_ablation_bar

        fig, ax = plt.subplots()
        df = _make_ablation_df()
        plot_ablation_bar(ax, df)

        # Collect text content from axes
        texts = [t.get_text() for t in ax.texts]
        # Should contain formatted values like "0.145", "0.097", "1.012", "0.173"
        assert any("0.145" in t for t in texts), (
            f"Expected '0.145' in annotations, got {texts}"
        )
        assert any("0.097" in t for t in texts), (
            f"Expected '0.097' in annotations, got {texts}"
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# TestPlotLatencyBars
# ---------------------------------------------------------------------------


class TestPlotLatencyBars:
    def test_plot_latency_bars_renders(self):
        """Latency bars renders grouped bars: 2 graphs x 2 methods = 4 bars."""
        from aac.viz.results import plot_latency_bars

        fig, ax = plt.subplots()
        df = _make_timing_df()
        plot_latency_bars(ax, df)
        patches = [p for p in ax.patches]
        assert len(patches) == 4, f"Expected 4 bars, got {len(patches)}"
        plt.close(fig)

    def test_plot_latency_bars_memory_annotation(self):
        """With memory_budget, axes has text annotations for memory context."""
        from aac.viz.results import plot_latency_bars

        fig, ax = plt.subplots()
        df = _make_timing_df()
        plot_latency_bars(
            ax, df, memory_budget={"aac": "64B/v", "alt": "128B/v"}
        )
        # Should have at least some text/legend entries mentioning memory
        # Check legend text
        legend = ax.get_legend()
        assert legend is not None, "Expected a legend"
        legend_texts = [t.get_text() for t in legend.get_texts()]
        found_memory = any("B/v" in t for t in legend_texts)
        assert found_memory, (
            f"Expected memory annotations in legend, got {legend_texts}"
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# TestPlotResultsDashboard
# ---------------------------------------------------------------------------


class TestPlotResultsDashboard:
    def test_plot_results_dashboard_renders(self):
        """Dashboard returns Figure with at least 4 subplots."""
        from aac.viz.results import plot_results_dashboard

        df_ny = _make_pareto_df()
        df_fla = _make_pareto_df()
        df_ablation = _make_ablation_df()
        df_timing = _make_timing_df()

        fig = plot_results_dashboard(df_ny, df_fla, df_ablation, df_timing)
        axes = fig.get_axes()
        assert len(axes) >= 4, f"Expected >= 4 axes, got {len(axes)}"
        plt.close(fig)

    def test_plot_results_dashboard_figsize(self):
        """Dashboard figure width equals TMLR_FULL_WIDTH (6.75 inches)."""
        from aac.viz.results import plot_results_dashboard
        from aac.viz.style import TMLR_FULL_WIDTH

        df_ny = _make_pareto_df()
        df_fla = _make_pareto_df()
        df_ablation = _make_ablation_df()
        df_timing = _make_timing_df()

        fig = plot_results_dashboard(df_ny, df_fla, df_ablation, df_timing)
        fig_width = fig.get_figwidth()
        assert abs(fig_width - TMLR_FULL_WIDTH) < 0.01, (
            f"Expected width {TMLR_FULL_WIDTH}, got {fig_width}"
        )
        plt.close(fig)
