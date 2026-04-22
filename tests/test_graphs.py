"""Tests for graph data structures, conversion, and validation."""

import pytest
import scipy.sparse
import torch

from aac.graphs.convert import edges_to_graph, graph_to_scipy, scipy_to_graph
from aac.graphs.types import Graph
from aac.graphs.validate import validate_graph


class TestGraphCreation:
    """Test Graph dataclass creation and basic properties."""

    def test_graph_creation(self, small_directed_graph: Graph) -> None:
        """Graph created from edge list has correct num_nodes, num_edges, and tensor shapes."""
        g = small_directed_graph
        assert g.num_nodes == 5
        assert g.num_edges == 7
        assert g.crow_indices.shape == (5 + 1,)
        assert g.col_indices.shape == (7,)
        assert g.values.shape == (7,)

    def test_graph_device_transfer(self, small_directed_graph: Graph, device: torch.device) -> None:
        """Graph.to(device) moves all tensors to the target device."""
        g = small_directed_graph.to(device)
        assert g.crow_indices.device.type == device.type
        assert g.col_indices.device.type == device.type
        assert g.values.device.type == device.type

    def test_graph_with_coordinates(self) -> None:
        """Graph with coordinates stores (V, 2) coordinate tensor."""
        coords = torch.rand(5, 2, dtype=torch.float64)
        g = Graph(
            crow_indices=torch.tensor([0, 2, 4, 6, 7, 7], dtype=torch.int64),
            col_indices=torch.tensor([1, 2, 2, 3, 3, 4, 4], dtype=torch.int64),
            values=torch.tensor([2.0, 5.0, 1.0, 6.0, 2.0, 7.0, 1.0], dtype=torch.float64),
            num_nodes=5,
            num_edges=7,
            is_directed=True,
            coordinates=coords,
        )
        assert g.coordinates is not None
        assert g.coordinates.shape == (5, 2)


class TestCOOConversion:
    """Test CSR <-> COO conversion."""

    def test_coo_roundtrip(self, small_directed_graph: Graph) -> None:
        """Graph.to_coo() returns (row_indices, col_indices, values) with correct shapes."""
        g = small_directed_graph
        row, col, val = g.to_coo()
        assert row.shape == (g.num_edges,)
        assert col.shape == (g.num_edges,)
        assert val.shape == (g.num_edges,)
        # Check known edge: (0, 1, 2.0)
        mask = (row == 0) & (col == 1)
        assert mask.sum() == 1
        assert val[mask].item() == pytest.approx(2.0)


class TestEdgesToGraph:
    """Test edges_to_graph conversion function."""

    def test_directed_graph_from_edges(self) -> None:
        """edges_to_graph creates correct directed graph from edge lists."""
        sources = torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.int64)
        targets = torch.tensor([1, 2, 2, 3, 3, 4, 4], dtype=torch.int64)
        weights = torch.tensor([2.0, 5.0, 1.0, 6.0, 2.0, 7.0, 1.0], dtype=torch.float64)
        g = edges_to_graph(sources, targets, weights, num_nodes=5, is_directed=True)
        assert g.num_nodes == 5
        assert g.num_edges == 7
        assert g.is_directed is True

    def test_undirected_doubles_edges(self) -> None:
        """edges_to_graph with is_directed=False doubles edge count and produces symmetric adjacency."""
        sources = torch.tensor([0, 0, 1], dtype=torch.int64)
        targets = torch.tensor([1, 2, 2], dtype=torch.int64)
        weights = torch.tensor([1.0, 4.0, 2.0], dtype=torch.float64)
        g = edges_to_graph(sources, targets, weights, num_nodes=3, is_directed=False)
        # 3 edges * 2 (both directions) = 6
        assert g.num_edges == 6
        assert g.is_directed is False
        # Check symmetry: dense adjacency should be symmetric
        dense = g.to_dense(sentinel=0.0)
        # Ignore diagonal
        dense.fill_diagonal_(0.0)
        assert torch.allclose(dense, dense.T)


class TestScipyConversion:
    """Test Graph <-> SciPy sparse conversion."""

    def test_graph_to_scipy(self, small_directed_graph: Graph) -> None:
        """graph_to_scipy produces scipy.sparse.csr_matrix with same nnz and values."""
        g = small_directed_graph
        sp = graph_to_scipy(g)
        assert isinstance(sp, scipy.sparse.csr_matrix)
        assert sp.nnz == g.num_edges
        assert sp.shape == (g.num_nodes, g.num_nodes)

    def test_scipy_roundtrip(self, small_directed_graph: Graph) -> None:
        """scipy_to_graph round-trips correctly: torch -> scipy -> torch gives same graph."""
        g = small_directed_graph
        sp = graph_to_scipy(g)
        g2 = scipy_to_graph(sp, is_directed=True)
        sp2 = graph_to_scipy(g2)
        # Compare scipy matrices
        diff = (sp - sp2).nnz
        assert diff == 0


class TestValidation:
    """Test graph validation checks."""

    def test_validate_negative_weights(self) -> None:
        """validate_graph raises ValueError on graph with negative weights."""
        g = Graph(
            crow_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            col_indices=torch.tensor([1, 0], dtype=torch.int64),
            values=torch.tensor([-1.0, 2.0], dtype=torch.float64),
            num_nodes=2,
            num_edges=2,
            is_directed=True,
        )
        with pytest.raises(ValueError, match="negative edge weights"):
            validate_graph(g)

    def test_validate_asymmetric(self) -> None:
        """validate_graph raises ValueError on undirected graph with asymmetric adjacency."""
        # Undirected graph but missing reverse edge: (0,1) present but not (1,0)
        g = Graph(
            crow_indices=torch.tensor([0, 1, 1], dtype=torch.int64),
            col_indices=torch.tensor([1], dtype=torch.int64),
            values=torch.tensor([1.0], dtype=torch.float64),
            num_nodes=2,
            num_edges=1,
            is_directed=False,
        )
        with pytest.raises(ValueError, match="asymmetric adjacency"):
            validate_graph(g)

    def test_validate_passes_on_valid_directed(self, small_directed_graph: Graph) -> None:
        """validate_graph passes on valid directed graph."""
        validate_graph(small_directed_graph)  # Should not raise

    def test_validate_passes_on_valid_undirected(self, small_undirected_graph: Graph) -> None:
        """validate_graph passes on valid undirected graph."""
        validate_graph(small_undirected_graph)  # Should not raise
