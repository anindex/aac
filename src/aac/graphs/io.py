"""NPZ serialization for Graph objects.

Provides save_graph_npz and load_graph_npz for efficient graph persistence
that bypasses NetworkX and directly stores/loads CSR arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from aac.graphs.types import Graph

_REQUIRED_KEYS = frozenset(
    {"crow_indices", "col_indices", "values", "num_nodes", "num_edges", "is_directed"}
)


def save_graph_npz(graph: Graph, path: str | Path) -> None:
    """Save a Graph to NPZ format.

    Stores CSR arrays (crow_indices, col_indices, values), scalar metadata
    (num_nodes, num_edges, is_directed), and optional coordinates as numpy
    arrays inside an uncompressed NPZ file.

    Args:
        graph: Graph object to serialize.
        path: Destination file path. The .npz suffix is appended by numpy
            if not already present.
    """
    arrays: dict[str, np.ndarray] = {
        "crow_indices": graph.crow_indices.cpu().numpy(),
        "col_indices": graph.col_indices.cpu().numpy(),
        "values": graph.values.cpu().numpy(),
        "num_nodes": np.int64(graph.num_nodes),
        "num_edges": np.int64(graph.num_edges),
        "is_directed": np.bool_(graph.is_directed),
    }
    if graph.coordinates is not None:
        arrays["coordinates"] = graph.coordinates.cpu().numpy()

    np.savez(str(path), **arrays)


def load_graph_npz(path: str | Path) -> Graph:
    """Load a Graph from an NPZ file.

    Security: Uses ``allow_pickle=False`` to prevent arbitrary code
    execution from crafted NPZ files.

    Args:
        path: Path to the NPZ file.

    Returns:
        Reconstructed Graph object with original dtypes and metadata.

    Raises:
        ValueError: If the NPZ file is missing required keys.
    """
    data = np.load(str(path), allow_pickle=False)

    present = set(data.files)
    missing = _REQUIRED_KEYS - present
    if missing:
        raise ValueError(
            f"NPZ file missing required keys: {sorted(missing)}. "
            f"Found: {sorted(present)}"
        )

    coordinates = (
        torch.tensor(data["coordinates"]) if "coordinates" in data else None
    )

    return Graph(
        crow_indices=torch.tensor(data["crow_indices"], dtype=torch.int64),
        col_indices=torch.tensor(data["col_indices"], dtype=torch.int64),
        values=torch.tensor(data["values"], dtype=torch.float64),
        num_nodes=int(data["num_nodes"].item()),
        num_edges=int(data["num_edges"].item()),
        is_directed=bool(data["is_directed"].item()),
        coordinates=coordinates,
    )
