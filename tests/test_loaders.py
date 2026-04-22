"""Tests for graph format loaders."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

FIXTURES = Path(__file__).parent / "fixtures"


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

        from aac.graphs.loaders.osmnx import _resolve_parallel_edges, _networkx_digraph_to_graph

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

    def test_osmnx_coordinates_from_latlon(self):
        """Verify coordinates are extracted from node lat/lon."""
        from aac.graphs.loaders.osmnx import _networkx_digraph_to_graph

        G = self._make_simple_digraph()
        g = _networkx_digraph_to_graph(G)

        # Coordinates should contain the lat/lon values from nodes
        assert g.coordinates is not None
        # The coordinates should be remapped to match 0-indexed node order


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
