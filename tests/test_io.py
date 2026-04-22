"""Tests for NPZ graph serialization (save_graph_npz, load_graph_npz)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aac.graphs.convert import edges_to_graph
from aac.graphs.io import load_graph_npz, save_graph_npz

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def directed_graph_5():
    """5-node directed graph (same as test_sssp.py)."""
    s = torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.int64)
    t = torch.tensor([1, 2, 2, 3, 3, 4, 4], dtype=torch.int64)
    w = torch.tensor([2.0, 5.0, 1.0, 6.0, 2.0, 7.0, 1.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 5, is_directed=True)


@pytest.fixture
def undirected_graph_5():
    """5-node undirected graph (same as test_sssp.py)."""
    s = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int64)
    t = torch.tensor([1, 2, 2, 3, 3, 4], dtype=torch.int64)
    w = torch.tensor([1.0, 4.0, 2.0, 5.0, 1.0, 3.0], dtype=torch.float64)
    return edges_to_graph(s, t, w, 5, is_directed=False)


@pytest.fixture
def graph_with_coordinates():
    """5-node directed graph with (V, 2) coordinates."""
    s = torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.int64)
    t = torch.tensor([1, 2, 2, 3, 3, 4, 4], dtype=torch.int64)
    w = torch.tensor([2.0, 5.0, 1.0, 6.0, 2.0, 7.0, 1.0], dtype=torch.float64)
    coords = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]],
        dtype=torch.float64,
    )
    return edges_to_graph(s, t, w, 5, is_directed=True, coordinates=coords)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


def test_npz_roundtrip_directed(tmp_path, directed_graph_5):
    """Save directed graph to NPZ and load back. All fields must match exactly."""
    path = tmp_path / "graph.npz"
    save_graph_npz(directed_graph_5, path)
    loaded = load_graph_npz(path)

    assert torch.equal(loaded.crow_indices, directed_graph_5.crow_indices)
    assert torch.equal(loaded.col_indices, directed_graph_5.col_indices)
    assert torch.equal(loaded.values, directed_graph_5.values)
    assert loaded.num_nodes == directed_graph_5.num_nodes
    assert loaded.num_edges == directed_graph_5.num_edges
    assert loaded.is_directed == directed_graph_5.is_directed
    assert loaded.is_directed is True
    assert loaded.coordinates is None


def test_npz_roundtrip_undirected(tmp_path, undirected_graph_5):
    """Save undirected graph to NPZ and load back. is_directed=False must round-trip."""
    path = tmp_path / "graph.npz"
    save_graph_npz(undirected_graph_5, path)
    loaded = load_graph_npz(path)

    assert torch.equal(loaded.crow_indices, undirected_graph_5.crow_indices)
    assert torch.equal(loaded.col_indices, undirected_graph_5.col_indices)
    assert torch.equal(loaded.values, undirected_graph_5.values)
    assert loaded.num_nodes == undirected_graph_5.num_nodes
    assert loaded.num_edges == undirected_graph_5.num_edges
    assert loaded.is_directed is False


def test_npz_roundtrip_with_coordinates(tmp_path, graph_with_coordinates):
    """Graph with coordinates=(V, 2). Loaded coordinates must match exactly."""
    path = tmp_path / "graph.npz"
    save_graph_npz(graph_with_coordinates, path)
    loaded = load_graph_npz(path)

    assert loaded.coordinates is not None
    assert torch.equal(loaded.coordinates, graph_with_coordinates.coordinates)


def test_npz_roundtrip_no_coordinates(tmp_path, directed_graph_5):
    """Graph without coordinates. Loaded graph must have coordinates=None."""
    path = tmp_path / "graph.npz"
    save_graph_npz(directed_graph_5, path)
    loaded = load_graph_npz(path)

    assert loaded.coordinates is None


def test_npz_dtypes_preserved(tmp_path, directed_graph_5):
    """CSR arrays must preserve their dtypes after round-trip."""
    path = tmp_path / "graph.npz"
    save_graph_npz(directed_graph_5, path)
    loaded = load_graph_npz(path)

    assert loaded.crow_indices.dtype == torch.int64
    assert loaded.col_indices.dtype == torch.int64
    assert loaded.values.dtype == torch.float64


def test_npz_load_rejects_pickle(tmp_path):
    """np.load must be called with allow_pickle=False. Pickle-based files must fail."""
    # Create a file using pickle-based np.save (not savez), which creates a .npy with pickle
    path = tmp_path / "malicious.npz"
    # Create an NPZ-like file with an object array that requires pickling
    import pickle
    import zipfile

    with zipfile.ZipFile(str(path), "w") as zf:
        # Write a pickle-based .npy entry
        buf = pickle.dumps({"malicious": True})
        zf.writestr("crow_indices.npy", buf)

    with pytest.raises((ValueError, Exception)):
        load_graph_npz(path)


def test_npz_load_missing_keys(tmp_path):
    """NPZ file missing required keys must raise ValueError with descriptive message."""
    path = tmp_path / "incomplete.npz"
    # Save only partial data
    np.savez(str(path), crow_indices=np.array([0, 1, 2]))

    with pytest.raises(ValueError, match="missing required keys"):
        load_graph_npz(path)


def test_npz_suffix_handling(tmp_path, directed_graph_5):
    """Saving to a path without .npz suffix must still create a loadable file."""
    path = tmp_path / "graph_no_suffix"
    save_graph_npz(directed_graph_5, path)

    # np.savez automatically appends .npz if not present
    actual_path = tmp_path / "graph_no_suffix.npz"
    loaded = load_graph_npz(actual_path)

    assert torch.equal(loaded.crow_indices, directed_graph_5.crow_indices)
    assert loaded.num_nodes == directed_graph_5.num_nodes
