"""Tests for graph data structures, conversion, validation, and NPZ I/O."""

import pickle
import zipfile

import numpy as np
import pytest
import scipy.sparse
import torch

from aac.graphs.convert import edges_to_graph, graph_to_scipy, scipy_to_graph
from aac.graphs.io import load_graph_npz, save_graph_npz
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


# ---------------------------------------------------------------------------
# NPZ serialization (formerly tests/test_io.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_with_coordinates() -> Graph:
    """5-node directed graph with (V, 2) coordinates."""
    s = torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.int64)
    t = torch.tensor([1, 2, 2, 3, 3, 4, 4], dtype=torch.int64)
    w = torch.tensor([2.0, 5.0, 1.0, 6.0, 2.0, 7.0, 1.0], dtype=torch.float64)
    coords = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]],
        dtype=torch.float64,
    )
    return edges_to_graph(s, t, w, 5, is_directed=True, coordinates=coords)


class TestNpzIO:
    """Round-trip tests for save_graph_npz / load_graph_npz."""

    def test_npz_roundtrip_directed(self, tmp_path, small_directed_graph) -> None:
        """Save directed graph to NPZ and load back. All fields must match exactly."""
        path = tmp_path / "graph.npz"
        save_graph_npz(small_directed_graph, path)
        loaded = load_graph_npz(path)

        assert torch.equal(loaded.crow_indices, small_directed_graph.crow_indices)
        assert torch.equal(loaded.col_indices, small_directed_graph.col_indices)
        assert torch.equal(loaded.values, small_directed_graph.values)
        assert loaded.num_nodes == small_directed_graph.num_nodes
        assert loaded.num_edges == small_directed_graph.num_edges
        assert loaded.is_directed is True
        assert loaded.coordinates is None

    def test_npz_roundtrip_undirected(self, tmp_path, small_undirected_graph) -> None:
        """Save undirected graph to NPZ and load back. is_directed=False must round-trip."""
        path = tmp_path / "graph.npz"
        save_graph_npz(small_undirected_graph, path)
        loaded = load_graph_npz(path)

        assert torch.equal(loaded.crow_indices, small_undirected_graph.crow_indices)
        assert torch.equal(loaded.col_indices, small_undirected_graph.col_indices)
        assert torch.equal(loaded.values, small_undirected_graph.values)
        assert loaded.num_nodes == small_undirected_graph.num_nodes
        assert loaded.num_edges == small_undirected_graph.num_edges
        assert loaded.is_directed is False

    def test_npz_roundtrip_with_coordinates(self, tmp_path, graph_with_coordinates) -> None:
        """Graph with coordinates=(V, 2). Loaded coordinates must match exactly."""
        path = tmp_path / "graph.npz"
        save_graph_npz(graph_with_coordinates, path)
        loaded = load_graph_npz(path)

        assert loaded.coordinates is not None
        assert torch.equal(loaded.coordinates, graph_with_coordinates.coordinates)

    def test_npz_roundtrip_no_coordinates(self, tmp_path, small_directed_graph) -> None:
        """Graph without coordinates. Loaded graph must have coordinates=None."""
        path = tmp_path / "graph.npz"
        save_graph_npz(small_directed_graph, path)
        loaded = load_graph_npz(path)

        assert loaded.coordinates is None

    def test_npz_dtypes_preserved(self, tmp_path, small_directed_graph) -> None:
        """CSR arrays must preserve their dtypes after round-trip."""
        path = tmp_path / "graph.npz"
        save_graph_npz(small_directed_graph, path)
        loaded = load_graph_npz(path)

        assert loaded.crow_indices.dtype == torch.int64
        assert loaded.col_indices.dtype == torch.int64
        assert loaded.values.dtype == torch.float64

    def test_npz_load_rejects_pickle(self, tmp_path) -> None:
        """np.load must be called with allow_pickle=False -- pickle-based files must fail."""
        path = tmp_path / "malicious.npz"
        with zipfile.ZipFile(str(path), "w") as zf:
            # Write a pickle-based .npy entry
            buf = pickle.dumps({"malicious": True})
            zf.writestr("crow_indices.npy", buf)

        with pytest.raises((ValueError, Exception)):
            load_graph_npz(path)

    def test_npz_load_missing_keys(self, tmp_path) -> None:
        """NPZ file missing required keys must raise ValueError with descriptive message."""
        path = tmp_path / "incomplete.npz"
        np.savez(str(path), crow_indices=np.array([0, 1, 2]))

        with pytest.raises(ValueError, match="missing required keys"):
            load_graph_npz(path)

    def test_npz_suffix_handling(self, tmp_path, small_directed_graph) -> None:
        """Saving to a path without .npz suffix must still create a loadable file."""
        path = tmp_path / "graph_no_suffix"
        save_graph_npz(small_directed_graph, path)

        # np.savez automatically appends .npz if not present
        actual_path = tmp_path / "graph_no_suffix.npz"
        loaded = load_graph_npz(actual_path)

        assert torch.equal(loaded.crow_indices, small_directed_graph.crow_indices)
        assert loaded.num_nodes == small_directed_graph.num_nodes
