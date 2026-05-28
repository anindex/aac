"""Tests for graph format loaders (DIMACS, MovingAI, OSMnx, Warcraft, PBF)."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

FIXTURES = Path(__file__).parent / "fixtures"


try:
    import osmium  # noqa: F401

    HAS_OSMIUM = True
except ImportError:
    HAS_OSMIUM = False


# --- DIMACS tests ---


class TestDimacs:
    """Tests for DIMACS .gr/.co loader."""

    def test_dimacs_node_edge_count(self):
        """Load tiny_dimacs.gr, check num_nodes==5, num_edges==7."""
        from aac.graphs.loaders.dimacs import load_dimacs

        g = load_dimacs(FIXTURES / "tiny_dimacs.gr")
        assert g.num_nodes == 5
        assert g.num_edges == 7

    def test_dimacs_zero_indexed(self):
        """All col_indices are in [0, 4]."""
        from aac.graphs.loaders.dimacs import load_dimacs

        g = load_dimacs(FIXTURES / "tiny_dimacs.gr")
        assert g.col_indices.min().item() >= 0
        assert g.col_indices.max().item() <= 4

    def test_dimacs_coordinates(self):
        """Load with .co file, check coordinates shape is (5, 2)."""
        from aac.graphs.loaders.dimacs import load_dimacs

        g = load_dimacs(FIXTURES / "tiny_dimacs.gr", co_path=FIXTURES / "tiny_dimacs.co")
        assert g.coordinates is not None
        assert g.coordinates.shape == (5, 2)

    def test_dimacs_directed_fp64(self):
        """graph.is_directed is True, graph.values.dtype is torch.float64."""
        from aac.graphs.loaders.dimacs import load_dimacs

        g = load_dimacs(FIXTURES / "tiny_dimacs.gr")
        assert g.is_directed is True
        assert g.values.dtype == torch.float64


# --- MovingAI tests ---


class TestMovingAI:
    """Tests for MovingAI .map/.scen loader."""

    def test_movingai_passable_cells(self):
        """Load tiny map, check num_nodes == 13 (16 - 3 blocked)."""
        from aac.graphs.loaders.movingai import load_movingai_map

        g = load_movingai_map(FIXTURES / "tiny_movingai.map")
        assert g.num_nodes == 13

    def test_movingai_diagonal_cost(self):
        """Find a diagonal edge and verify weight is approximately sqrt(2)."""
        from aac.graphs.loaders.movingai import load_movingai_map

        g = load_movingai_map(FIXTURES / "tiny_movingai.map")
        # In our 4x4 grid with some blocked cells, there should be diagonal edges
        # with weight sqrt(2). Check that at least one edge has this weight.
        has_diagonal = torch.any(torch.isclose(g.values, torch.tensor(math.sqrt(2), dtype=torch.float64)))
        assert has_diagonal, f"No diagonal edge with sqrt(2) cost found. Unique weights: {g.values.unique()}"

    def test_movingai_cardinal_cost(self):
        """Find a cardinal edge and verify weight is 1.0."""
        from aac.graphs.loaders.movingai import load_movingai_map

        g = load_movingai_map(FIXTURES / "tiny_movingai.map")
        has_cardinal = torch.any(torch.isclose(g.values, torch.tensor(1.0, dtype=torch.float64)))
        assert has_cardinal, f"No cardinal edge with 1.0 cost found. Unique weights: {g.values.unique()}"

    def test_movingai_scenario_parsing(self):
        """Load .scen, check first entry has correct start/goal/cost."""
        from aac.graphs.loaders.movingai import load_movingai_scenarios

        scenarios = load_movingai_scenarios(FIXTURES / "tiny_movingai.scen")
        assert len(scenarios) == 1
        start, goal, cost = scenarios[0]
        assert start == (0, 0), f"Expected start (0, 0), got {start}"
        assert goal == (3, 3), f"Expected goal (3, 3), got {goal}"
        assert abs(cost - 4.24264069) < 1e-5, f"Expected cost ~4.243, got {cost}"

    def test_movingai_coordinates(self):
        """MovingAI coordinates use (col, row) convention mapped to node indices."""
        from aac.graphs.loaders.movingai import load_movingai_map

        g = load_movingai_map(FIXTURES / "tiny_movingai.map")
        assert g.coordinates is not None
        assert g.coordinates.shape == (13, 2)
        # First passable cell is (0,0) -> coordinates should be (0, 0)
        assert g.coordinates[0, 0].item() == 0  # col
        assert g.coordinates[0, 1].item() == 0  # row

    def test_movingai_undirected(self):
        """MovingAI graph should be undirected."""
        from aac.graphs.loaders.movingai import load_movingai_map

        g = load_movingai_map(FIXTURES / "tiny_movingai.map")
        assert g.is_directed is False


# --- OSMnx tests ---


class TestOSMnx:
    """Tests for OSMnx graph loader (using mock NetworkX graphs)."""

    def _make_simple_digraph(self):
        """Create a small NetworkX DiGraph with 5 nodes, 8 edges with 'length' attr."""
        import networkx as nx

        G = nx.DiGraph()
        # Nodes with x (lon) and y (lat) attributes matching OSMnx convention
        G.add_node(100, x=-73.9, y=40.7)
        G.add_node(200, x=-73.8, y=40.8)
        G.add_node(300, x=-73.7, y=40.9)
        G.add_node(400, x=-73.6, y=40.6)
        G.add_node(500, x=-73.5, y=40.5)

        edges = [
            (100, 200, 150.0),
            (200, 300, 200.0),
            (300, 400, 180.0),
            (400, 500, 120.0),
            (200, 100, 155.0),
            (300, 200, 205.0),
            (500, 100, 300.0),
            (100, 400, 250.0),
        ]
        for u, v, length in edges:
            G.add_edge(u, v, length=length)

        return G

    def test_osmnx_mock_graph(self):
        """Convert a mock NetworkX DiGraph using the internal helper."""
        from aac.graphs.loaders.osmnx import _networkx_digraph_to_graph

        G = self._make_simple_digraph()
        g = _networkx_digraph_to_graph(G)

        assert g.num_nodes == 5
        assert g.is_directed is True
        assert g.values.dtype == torch.float64
        # All col_indices should be 0-indexed in [0, 4]
        assert g.col_indices.min().item() >= 0
        assert g.col_indices.max().item() <= 4
        # Coordinates should exist
        assert g.coordinates is not None
        assert g.coordinates.shape == (5, 2)

    def test_osmnx_no_parallel_edges(self):
        """Create MultiDiGraph with parallel edges, verify only min-weight kept."""
        import networkx as nx

        from aac.graphs.loaders.osmnx import _networkx_digraph_to_graph, _resolve_parallel_edges

        G = nx.MultiDiGraph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=1.0, y=1.0)
        G.add_node(3, x=2.0, y=2.0)

        # Add parallel edges between nodes 1 and 2
        G.add_edge(1, 2, key=0, length=100.0)
        G.add_edge(1, 2, key=1, length=50.0)   # shorter
        G.add_edge(1, 2, key=2, length=200.0)
        G.add_edge(2, 3, key=0, length=80.0)

        # Convert MultiDiGraph to simple DiGraph keeping min-weight edges
        simple_G = _resolve_parallel_edges(G)
        g = _networkx_digraph_to_graph(simple_G)

        assert g.num_nodes == 3
        # Should have exactly 2 edges: (1->2, min=50) and (2->3, 80)
        assert g.num_edges == 2
        # The min weight from node 0 to node 1 should be 50.0
        assert 50.0 in g.values.tolist()

    def test_osmnx_coordinates_remap_to_node_order(self):
        """Coordinates are remapped so each row matches the 0-indexed node order."""
        from aac.graphs.loaders.osmnx import _networkx_digraph_to_graph

        G = self._make_simple_digraph()
        g = _networkx_digraph_to_graph(G)

        # Nodes were created in order 100, 200, 300, 400, 500 with monotonic
        # x = -73.9, -73.8, ... and we expect row i to correspond to the i-th
        # node in that insertion order.
        assert g.coordinates is not None
        expected = torch.tensor(
            [[-73.9, 40.7], [-73.8, 40.8], [-73.7, 40.9], [-73.6, 40.6], [-73.5, 40.5]],
            dtype=torch.float64,
        )
        assert torch.allclose(g.coordinates, expected, atol=1e-12)


# --- Warcraft tests ---


class TestWarcraft:
    """Tests for Warcraft terrain map loader."""

    def test_warcraft_synthetic(self):
        """Create a small synthetic .npz with 4x4 cost map, verify graph structure."""
        from aac.graphs.loaders.warcraft import load_warcraft

        # Create a 4x4 cost map with values 1-5
        cost_map = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 2.0],
            [4.0, 3.0, 2.0, 1.0],
        ], dtype=np.float64)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, cost_map=cost_map)
            g, cost_tensor = load_warcraft(f.name)

        assert g.num_nodes == 16  # 4x4 grid
        assert g.is_directed is False
        assert g.values.dtype == torch.float64
        # All edge weights should be positive
        assert (g.values > 0).all()
        # Cost tensor should match input
        assert cost_tensor.shape == (4, 4)

    def test_warcraft_edge_weights_positive(self):
        """Verify all edge weights are strictly positive."""
        from aac.graphs.loaders.warcraft import load_warcraft

        cost_map = np.ones((3, 3), dtype=np.float64) * 2.0
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, cost_map=cost_map)
            g, _ = load_warcraft(f.name)

        assert (g.values > 0).all()


# --- PBF tests (formerly tests/test_pbf_loader.py) ---


def _write_synthetic_pbf(path: str) -> None:
    """Write a small synthetic PBF file with a mix of road types and oneway settings.

    Layout (approximate, lon/lat):
        Nodes 0-9 placed along a line for simplicity.
        Ways:
          - Way 1: residential, bidirectional, nodes [1, 2, 3]
          - Way 2: motorway, oneway=yes (forward), nodes [3, 4, 5]
          - Way 3: footway (non-driving), nodes [5, 6, 7] -- should be excluded
          - Way 4: primary, oneway=-1 (reverse), nodes [7, 8, 9]
          - Way 5: cycleway (non-driving), nodes [1, 9] -- should be excluded
    """
    import osmium

    writer = osmium.SimpleWriter(path)

    # Create nodes with distinct coordinates (lon, lat) = (i * 0.01, 50.0 + i * 0.001)
    for i in range(1, 10):
        node = osmium.osm.mutable.Node(
            id=i,
            location=osmium.osm.Location(i * 0.01, 50.0 + i * 0.001),
        )
        writer.add_node(node)

    way1 = osmium.osm.mutable.Way(
        id=100, nodes=[1, 2, 3], tags={"highway": "residential"},
    )
    writer.add_way(way1)

    way2 = osmium.osm.mutable.Way(
        id=101, nodes=[3, 4, 5], tags={"highway": "motorway", "oneway": "yes"},
    )
    writer.add_way(way2)

    way3 = osmium.osm.mutable.Way(
        id=102, nodes=[5, 6, 7], tags={"highway": "footway"},
    )
    writer.add_way(way3)

    way4 = osmium.osm.mutable.Way(
        id=103, nodes=[7, 8, 9], tags={"highway": "primary", "oneway": "-1"},
    )
    writer.add_way(way4)

    way5 = osmium.osm.mutable.Way(
        id=104, nodes=[1, 9], tags={"highway": "cycleway"},
    )
    writer.add_way(way5)

    writer.close()


@pytest.mark.skipif(not HAS_OSMIUM, reason="pyosmium not installed")
class TestPbf:
    """Tests for the PBF road network loader."""

    @pytest.fixture
    def synthetic_pbf(self, tmp_path: Path) -> str:
        """Create a synthetic PBF file and return its path."""
        pbf_path = str(tmp_path / "test.osm.pbf")
        _write_synthetic_pbf(pbf_path)
        return pbf_path

    def test_load_produces_directed_graph(self, synthetic_pbf: str) -> None:
        """load_pbf_road_graph produces a directed Graph."""
        from aac.graphs.loaders.pbf import load_pbf_road_graph
        from aac.graphs.types import Graph

        graph = load_pbf_road_graph(synthetic_pbf)
        assert isinstance(graph, Graph)
        assert graph.is_directed is True
        assert graph.num_nodes > 0
        assert graph.num_edges > 0

    def test_all_edge_weights_positive(self, synthetic_pbf: str) -> None:
        """All edge weights are positive (no zero-length edges)."""
        from aac.graphs.loaders.pbf import load_pbf_road_graph

        graph = load_pbf_road_graph(synthetic_pbf)
        assert (graph.values > 0).all(), "All edge weights must be positive"

    def test_node_ids_contiguous(self, synthetic_pbf: str) -> None:
        """Node IDs are contiguous 0-indexed (no gaps)."""
        from aac.graphs.loaders.pbf import load_pbf_road_graph

        graph = load_pbf_road_graph(synthetic_pbf)
        # Check CSR structure: crow_indices should be (V+1,)
        assert graph.crow_indices.shape[0] == graph.num_nodes + 1
        # col_indices should be in [0, V)
        assert graph.col_indices.min() >= 0
        assert graph.col_indices.max() < graph.num_nodes

    def test_coordinates_shape_and_finite(self, synthetic_pbf: str) -> None:
        """Coordinates tensor has shape (V, 2) and values are finite."""
        from aac.graphs.loaders.pbf import load_pbf_road_graph

        graph = load_pbf_road_graph(synthetic_pbf)
        assert graph.coordinates is not None
        assert graph.coordinates.shape == (graph.num_nodes, 2)
        assert torch.isfinite(graph.coordinates).all()

    def test_oneway_asymmetric_edges(self, synthetic_pbf: str) -> None:
        """Oneway roads produce asymmetric edge sets; bidirectional produce both."""
        from aac.graphs.loaders.pbf import load_pbf_road_graph

        graph = load_pbf_road_graph(synthetic_pbf)
        rows, cols, _ = graph.to_coo()

        edge_set = set(zip(rows.tolist(), cols.tolist()))

        # Some edges should be one-directional (from oneway ways)
        reverse_count = sum(1 for u, v in edge_set if (v, u) in edge_set)
        forward_only = len(edge_set) - reverse_count
        assert forward_only > 0, "Oneway roads should produce asymmetric edges"

    def test_non_driving_excluded(self, synthetic_pbf: str) -> None:
        """Non-driving highway types (footway, cycleway) are excluded."""
        from aac.graphs.loaders.pbf import load_pbf_road_graph

        graph = load_pbf_road_graph(synthetic_pbf)

        # Driving ways use nodes {1,2,3,4,5,7,8,9} = 8 nodes; node 6 is only in
        # the footway, so it should be excluded.
        assert graph.num_nodes == 8, (
            f"Expected 8 driving nodes but got {graph.num_nodes}. "
            "Non-driving highway types should be excluded."
        )
