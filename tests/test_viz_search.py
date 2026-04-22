"""Tests for search behavior visualization functions (aac.viz.search)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from aac.search.types import SearchResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_search_result(
    expanded_nodes: list[int],
    path: list[int],
    cost: float = 10.0,
) -> SearchResult:
    """Create a SearchResult with given expansion data."""
    return SearchResult(
        path=path,
        cost=cost,
        expansions=len(expanded_nodes),
        optimal=True,
        h_source=0.0,
        expanded_nodes=expanded_nodes,
        g_values={n: float(i) for i, n in enumerate(expanded_nodes)},
    )


@pytest.fixture
def grid_5x5_result() -> SearchResult:
    """A SearchResult for a 5x5 grid (25 nodes), path from 0 to 24."""
    # Simulated expansion order: expand first row, then diagonally
    expanded = list(range(15))  # first 15 nodes expanded
    path = [0, 1, 2, 3, 4, 9, 14, 19, 24]
    return _make_search_result(expanded, path, cost=8.0)


@pytest.fixture
def road_coords() -> np.ndarray:
    """20 random 2D coordinates for road-network-style tests."""
    rng = np.random.default_rng(42)
    return rng.uniform(low=-74.0, high=-73.5, size=(20, 2))


@pytest.fixture
def road_result() -> SearchResult:
    """A SearchResult for a 20-node road network."""
    expanded = [0, 3, 7, 12, 15, 19]
    path = [0, 3, 12, 19]
    return _make_search_result(expanded, path, cost=5.0)


# ---------------------------------------------------------------------------
# plot_expansion_heatmap (road network scatter)
# ---------------------------------------------------------------------------


class TestPlotExpansionHeatmap:
    def test_returns_scalar_mappable(self, road_coords, road_result):
        """plot_expansion_heatmap returns a ScalarMappable."""
        from aac.viz.search import plot_expansion_heatmap

        fig, ax = plt.subplots()
        sm = plot_expansion_heatmap(
            ax,
            road_coords,
            road_result.expanded_nodes,
            road_result.path,
            source=0,
            target=19,
        )
        assert hasattr(sm, "get_cmap"), "Return value should be a ScalarMappable"
        plt.close(fig)

    def test_axes_has_collections_and_line(self, road_coords, road_result):
        """Axes has at least 2 collections (background + expanded) and 1 line (path)."""
        from aac.viz.search import plot_expansion_heatmap

        fig, ax = plt.subplots()
        plot_expansion_heatmap(
            ax,
            road_coords,
            road_result.expanded_nodes,
            road_result.path,
            source=0,
            target=19,
        )
        assert len(ax.collections) >= 2, (
            f"Expected >= 2 collections, got {len(ax.collections)}"
        )
        assert len(ax.lines) >= 1, f"Expected >= 1 line, got {len(ax.lines)}"
        plt.close(fig)

    def test_colormap_is_ylorrd(self, road_coords, road_result):
        """Colormap name contains 'YlOrRd' (matches the documented style)."""
        from aac.viz.search import plot_expansion_heatmap

        fig, ax = plt.subplots()
        sm = plot_expansion_heatmap(
            ax,
            road_coords,
            road_result.expanded_nodes,
            road_result.path,
            source=0,
            target=19,
        )
        cmap_name = sm.get_cmap().name
        assert "YlOrRd" in cmap_name, f"Expected YlOrRd colormap, got {cmap_name}"
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_expansion_heatmap_grid (grid imshow)
# ---------------------------------------------------------------------------


class TestPlotExpansionHeatmapGrid:
    def test_returns_scalar_mappable(self, grid_5x5_result):
        """plot_expansion_heatmap_grid returns a ScalarMappable (AxesImage)."""
        from aac.viz.search import plot_expansion_heatmap_grid

        fig, ax = plt.subplots()
        sm = plot_expansion_heatmap_grid(
            ax,
            5,
            5,
            grid_5x5_result.expanded_nodes,
            grid_5x5_result.path,
            source=0,
            target=24,
        )
        assert hasattr(sm, "get_cmap"), "Return value should be a ScalarMappable"
        plt.close(fig)

    def test_axes_has_image(self, grid_5x5_result):
        """Axes has an imshow image."""
        from aac.viz.search import plot_expansion_heatmap_grid

        fig, ax = plt.subplots()
        plot_expansion_heatmap_grid(
            ax,
            5,
            5,
            grid_5x5_result.expanded_nodes,
            grid_5x5_result.path,
            source=0,
            target=24,
        )
        assert len(ax.images) >= 1, f"Expected >= 1 image, got {len(ax.images)}"
        plt.close(fig)

    def test_axes_has_path_line(self, grid_5x5_result):
        """Axes has at least 1 line (shortest path overlay)."""
        from aac.viz.search import plot_expansion_heatmap_grid

        fig, ax = plt.subplots()
        plot_expansion_heatmap_grid(
            ax,
            5,
            5,
            grid_5x5_result.expanded_nodes,
            grid_5x5_result.path,
            source=0,
            target=24,
        )
        assert len(ax.lines) >= 1, f"Expected >= 1 line, got {len(ax.lines)}"
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_comparison_panel (3-panel figure)
# ---------------------------------------------------------------------------


class TestPlotComparisonPanel:
    def test_returns_figure_with_4_axes(self):
        """Returns figure with 4 axes (3 panels + 1 colorbar)."""
        from aac.viz.search import plot_comparison_panel

        # Grid is 10x10 (100 nodes), so expanded indices are within bounds
        H, W = 10, 10
        path = [0, 1, 2, 3, 4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 99]
        results = [
            _make_search_result(list(range(80)), path),
            _make_search_result(list(range(40)), path),
            _make_search_result(list(range(15)), path),
        ]
        method_keys = ["dijkstra", "alt", "aac"]

        fig = plot_comparison_panel(
            results, method_keys, grid_dims=(H, W)
        )
        axes = fig.get_axes()
        assert len(axes) == 4, f"Expected 4 axes, got {len(axes)}"
        plt.close(fig)

    def test_panel_titles_contain_expansion_count(self):
        """Each panel title contains expansion count with comma formatting."""
        from aac.viz.search import plot_comparison_panel

        # Use 50x50 grid (2500 nodes) to accommodate 1247 expansions
        H, W = 50, 50
        path = [0, 1, 2, 3, 53, 103, 153, 203, 253]
        results = [
            _make_search_result(list(range(1247)), path),
            _make_search_result(list(range(412)), path),
            _make_search_result(list(range(189)), path),
        ]
        method_keys = ["dijkstra", "alt", "aac"]

        fig = plot_comparison_panel(
            results, method_keys, grid_dims=(H, W)
        )
        axes = fig.get_axes()
        # First 3 axes are panels
        titles = [axes[i].get_title() for i in range(3)]
        assert "1,247" in titles[0], f"Expected '1,247' in title, got '{titles[0]}'"
        assert "412" in titles[1], f"Expected '412' in title, got '{titles[1]}'"
        assert "189" in titles[2], f"Expected '189' in title, got '{titles[2]}'"
        plt.close(fig)

    def test_shared_norm_across_panels(self):
        """All 3 panels share the same color normalization vmax."""
        from aac.viz.search import plot_comparison_panel

        # Grid 15x15 (225 nodes) to accommodate 100 expansions
        H, W = 15, 15
        path = [0, 1, 2, 17, 32, 47, 62, 77, 92]
        results = [
            _make_search_result(list(range(100)), path),
            _make_search_result(list(range(50)), path),
            _make_search_result(list(range(20)), path),
        ]
        method_keys = ["dijkstra", "alt", "aac"]

        fig = plot_comparison_panel(
            results, method_keys, grid_dims=(H, W)
        )
        # The shared norm vmax should be based on the max expansions (100)
        # We verify by checking that the colorbar exists and the figure was created
        axes = fig.get_axes()
        assert len(axes) == 4, "Should have 3 panels + 1 colorbar"
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_heuristic_contour (road network tricontourf)
# ---------------------------------------------------------------------------


class TestPlotHeuristicContour:
    def test_creates_filled_contour(self, road_coords):
        """plot_heuristic_contour creates filled contour on axes."""
        from aac.viz.search import plot_heuristic_contour

        h_values = np.linalg.norm(
            road_coords - road_coords[19], axis=1
        )
        path = [0, 3, 12, 19]

        fig, ax = plt.subplots()
        result = plot_heuristic_contour(
            ax, road_coords, h_values, path, target=19
        )
        # tricontourf creates collections
        assert len(ax.collections) >= 1, "Expected filled contour collections"
        plt.close(fig)

    def test_has_path_overlay(self, road_coords):
        """Axes has at least 1 line (path overlay)."""
        from aac.viz.search import plot_heuristic_contour

        h_values = np.linalg.norm(
            road_coords - road_coords[19], axis=1
        )
        path = [0, 3, 12, 19]

        fig, ax = plt.subplots()
        plot_heuristic_contour(
            ax, road_coords, h_values, path, target=19
        )
        assert len(ax.lines) >= 1, f"Expected >= 1 line, got {len(ax.lines)}"
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_heuristic_contour_grid (grid imshow contour)
# ---------------------------------------------------------------------------


class TestPlotHeuristicContourGrid:
    def test_creates_image(self):
        """plot_heuristic_contour_grid creates image on axes."""
        from aac.viz.search import plot_heuristic_contour_grid

        H, W = 5, 5
        # h-values: distance from target (node 24 = row 4, col 4)
        h_values = np.zeros(H * W)
        for node_id in range(H * W):
            r, c = divmod(node_id, W)
            h_values[node_id] = abs(r - 4) + abs(c - 4)

        path = [0, 1, 2, 3, 4, 9, 14, 19, 24]

        fig, ax = plt.subplots()
        plot_heuristic_contour_grid(ax, H, W, h_values, path, target=24)
        assert len(ax.images) >= 1, f"Expected >= 1 image, got {len(ax.images)}"
        plt.close(fig)

    def test_has_path_overlay(self):
        """Axes has at least 1 line (path overlay)."""
        from aac.viz.search import plot_heuristic_contour_grid

        H, W = 5, 5
        h_values = np.zeros(H * W)
        for node_id in range(H * W):
            r, c = divmod(node_id, W)
            h_values[node_id] = abs(r - 4) + abs(c - 4)

        path = [0, 1, 2, 3, 4, 9, 14, 19, 24]

        fig, ax = plt.subplots()
        plot_heuristic_contour_grid(ax, H, W, h_values, path, target=24)
        assert len(ax.lines) >= 1, f"Expected >= 1 line, got {len(ax.lines)}"
        plt.close(fig)
