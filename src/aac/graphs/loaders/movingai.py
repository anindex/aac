"""MovingAI .map/.scen format parser for grid-world benchmarks.

Parses the MovingAI benchmark format:
- .map files: ASCII grid with terrain characters (`.` passable, `@` blocked, etc.)
- .scen files: scenario files with start/goal coordinates and optimal costs

Grid graphs use 8-connected neighbors with cardinal cost 1.0 and diagonal cost sqrt(2).
Coordinates follow (col, row) convention matching MovingAI's (x, y) = (col, row).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from aac.graphs.convert import edges_to_graph
from aac.graphs.types import Graph

# Passable terrain characters per MovingAI spec
_PASSABLE = frozenset("." "G" "S")

# 8-connected neighbor offsets: (drow, dcol)
_NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

_SQRT2 = math.sqrt(2)


def load_movingai_map(map_path: str | Path) -> Graph:
    """Load a MovingAI .map file into an 8-connected grid graph.

    Passable terrain characters: `.`, `G`, `S`.
    Blocked terrain: `@`, `O`, `T`, `W`.

    Args:
        map_path: Path to the .map file.

    Returns:
        Undirected Graph with:
        - Cardinal edges (cost 1.0) and diagonal edges (cost sqrt(2))
        - Coordinates as (col, row) for each passable cell
        - Contiguous 0-indexed node IDs for passable cells only
    """
    height, width, grid = _parse_map_file(map_path)

    # Assign contiguous IDs to passable cells
    cell_to_node: dict[tuple[int, int], int] = {}
    coords: list[tuple[float, float]] = []
    node_id = 0

    for r in range(height):
        for c in range(width):
            if grid[r][c] in _PASSABLE:
                cell_to_node[(r, c)] = node_id
                coords.append((float(c), float(r)))  # (col, row) convention
                node_id += 1

    num_nodes = node_id

    # Build edges between passable neighbors
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []

    for (r, c), src_id in cell_to_node.items():
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_to_node:
                tgt_id = cell_to_node[(nr, nc)]
                is_diagonal = dr != 0 and dc != 0
                cost = _SQRT2 if is_diagonal else 1.0
                sources.append(src_id)
                targets.append(tgt_id)
                weights.append(cost)

    src_t = torch.tensor(sources, dtype=torch.int64)
    tgt_t = torch.tensor(targets, dtype=torch.int64)
    wgt_t = torch.tensor(weights, dtype=torch.float64)
    coord_t = torch.tensor(coords, dtype=torch.float64)

    # edges_to_graph will add reverse edges for undirected, but since we already
    # enumerate both directions (src->tgt and tgt->src via symmetric neighbor iteration),
    # we should pass this as directed and set is_directed=False on the result.
    # Actually, the neighbor loop already produces both (u,v) and (v,u) since we iterate
    # all passable cells. So we need to pass as directed to avoid doubling.
    # But edges_to_graph for undirected adds reverse edges. Since we already have them,
    # pass as directed to build CSR, then set is_directed=False.

    # We already have symmetric edges from the neighbor iteration. To avoid
    # edges_to_graph doubling them, we only emit one direction and let
    # edges_to_graph add the reverse.
    # Actually, simpler: just emit one direction per pair and use is_directed=False.

    # The neighbor loop visits each cell and each neighbor direction independently,
    # producing both (A,B) and (B,A). So we have all edges already.
    # Build CSR directly as directed to avoid doubling.
    # But Graph.is_directed should be False for undirected semantics.

    # Let's just deduplicate: only emit edge (src, tgt) where src < tgt,
    # then use is_directed=False so edges_to_graph adds reverse edges.
    dedup_sources: list[int] = []
    dedup_targets: list[int] = []
    dedup_weights: list[float] = []

    for i in range(len(sources)):
        if sources[i] < targets[i]:
            dedup_sources.append(sources[i])
            dedup_targets.append(targets[i])
            dedup_weights.append(weights[i])

    src_t = torch.tensor(dedup_sources, dtype=torch.int64)
    tgt_t = torch.tensor(dedup_targets, dtype=torch.int64)
    wgt_t = torch.tensor(dedup_weights, dtype=torch.float64)

    return edges_to_graph(
        sources=src_t,
        targets=tgt_t,
        weights=wgt_t,
        num_nodes=num_nodes,
        is_directed=False,
        coordinates=coord_t,
    )


def load_movingai_scenarios(
    scen_path: str | Path,
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """Parse a MovingAI .scen scenario file.

    Args:
        scen_path: Path to the .scen file.

    Returns:
        List of ((start_col, start_row), (goal_col, goal_row), optimal_cost) tuples.
        Coordinates use MovingAI convention: x = col, y = row.
    """
    scenarios: list[tuple[tuple[int, int], tuple[int, int], float]] = []

    with open(scen_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("version"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                # Try space-separated as fallback
                parts = line.split()
            if len(parts) < 9:
                continue

            # Format: bucket map_name map_width map_height start_x start_y goal_x goal_y optimal_length
            start_col = int(parts[4])
            start_row = int(parts[5])
            goal_col = int(parts[6])
            goal_row = int(parts[7])
            optimal_cost = float(parts[8])

            scenarios.append(
                ((start_col, start_row), (goal_col, goal_row), optimal_cost)
            )

    return scenarios


def _parse_map_file(map_path: str | Path) -> tuple[int, int, list[str]]:
    """Parse the header and grid from a MovingAI .map file.

    Returns:
        (height, width, grid) where grid is a list of row strings.
    """
    height = 0
    width = 0
    grid: list[str] = []

    with open(map_path, "r") as f:
        # Parse header
        for line in f:
            line = line.strip()
            if line.startswith("type"):
                continue
            elif line.startswith("height"):
                height = int(line.split()[1])
            elif line.startswith("width"):
                width = int(line.split()[1])
            elif line == "map":
                break

        # Parse grid rows
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line:
                grid.append(line)

    if len(grid) != height:
        raise ValueError(f"Expected {height} grid rows, got {len(grid)}")

    return height, width, grid
