"""DIMACS .gr/.co format parser and loader.

Parses the 9th DIMACS Implementation Challenge format:
- .gr files: graph edges with `p sp <nodes> <edges>` header and `a <u> <v> <w>` arc lines (1-indexed)
- .co files: coordinates with `v <id> <x> <y>` lines (1-indexed)

All vertex IDs are converted from 1-indexed (DIMACS) to 0-indexed (internal).
"""

from __future__ import annotations

from pathlib import Path

import torch

from aac.graphs.convert import edges_to_graph
from aac.graphs.types import Graph


def load_dimacs(
    gr_path: str | Path,
    co_path: str | Path | None = None,
) -> Graph:
    """Load a DIMACS .gr file (and optional .co coordinates) into a Graph.

    Args:
        gr_path: Path to the .gr file with edge definitions.
        co_path: Optional path to the .co file with vertex coordinates.

    Returns:
        Directed Graph with fp64 edge weights and optional (V, 2) coordinates.

    Raises:
        ValueError: If the parsed edge count does not match the header.
    """
    num_nodes = 0
    num_edges_header = 0
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []

    with open(gr_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if parts[0] == "p":
                # p sp <nodes> <edges>
                num_nodes = int(parts[2])
                num_edges_header = int(parts[3])
            elif parts[0] == "a":
                # a <u> <v> <weight> (1-indexed)
                u = int(parts[1]) - 1  # convert to 0-indexed
                v = int(parts[2]) - 1
                w = float(parts[3])
                sources.append(u)
                targets.append(v)
                weights.append(w)

    if len(sources) != num_edges_header:
        raise ValueError(
            f"Expected {num_edges_header} edges from header, got {len(sources)}"
        )

    # Parse optional coordinate file
    coordinates = None
    if co_path is not None:
        coordinates = _parse_coordinates(co_path, num_nodes)

    src_t = torch.tensor(sources, dtype=torch.int64)
    tgt_t = torch.tensor(targets, dtype=torch.int64)
    wgt_t = torch.tensor(weights, dtype=torch.float64)

    return edges_to_graph(
        sources=src_t,
        targets=tgt_t,
        weights=wgt_t,
        num_nodes=num_nodes,
        is_directed=True,
        coordinates=coordinates,
    )


def _parse_coordinates(co_path: str | Path, num_nodes: int) -> torch.Tensor:
    """Parse a DIMACS .co coordinate file.

    Args:
        co_path: Path to the .co file.
        num_nodes: Expected number of vertices.

    Returns:
        (V, 2) fp64 tensor of (x, y) coordinates.
    """
    coords = torch.zeros(num_nodes, 2, dtype=torch.float64)

    with open(co_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c") or line.startswith("p"):
                continue
            parts = line.split()
            if parts[0] == "v":
                # v <id> <x> <y> (1-indexed)
                vid = int(parts[1]) - 1  # convert to 0-indexed
                x = float(parts[2])
                y = float(parts[3])
                coords[vid, 0] = x
                coords[vid, 1] = y

    return coords
