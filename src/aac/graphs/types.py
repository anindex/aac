"""Core graph data structures in CSR format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class Graph:
    """Sparse weighted graph in CSR format.

    Attributes:
        crow_indices: (V+1,) int64 row pointer array for CSR format.
        col_indices: (E,) int64 column indices for CSR format.
        values: (E,) fp64 edge weights.
        num_nodes: Number of vertices V.
        num_edges: Number of stored edges E.
        is_directed: Whether the graph is directed.
        coordinates: Optional (V, 2) tensor for spatial graphs.
    """

    crow_indices: torch.Tensor
    col_indices: torch.Tensor
    values: torch.Tensor
    num_nodes: int
    num_edges: int
    is_directed: bool
    coordinates: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        """Return the device of the graph tensors."""
        return self.values.device

    def to(self, device: torch.device) -> Graph:
        """Move all tensors to the target device."""
        return Graph(
            crow_indices=self.crow_indices.to(device),
            col_indices=self.col_indices.to(device),
            values=self.values.to(device),
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            is_directed=self.is_directed,
            coordinates=self.coordinates.to(device) if self.coordinates is not None else None,
        )

    def to_coo(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (row_indices, col_indices, values) in COO format."""
        row_indices = torch.repeat_interleave(
            torch.arange(self.num_nodes, device=self.device),
            self.crow_indices[1:] - self.crow_indices[:-1],
        )
        return row_indices, self.col_indices, self.values

    def to_dense(self, sentinel: float = 1e18) -> torch.Tensor:
        """Convert to (V, V) dense adjacency matrix with sentinel for non-edges.

        Diagonal entries are set to 0 (distance from a node to itself).
        Note: self-loop edge weights, if present, are overwritten by 0.
        """
        V = self.num_nodes
        dense = torch.full((V, V), sentinel, dtype=self.values.dtype, device=self.device)
        row_indices, col_indices, values = self.to_coo()
        dense[row_indices, col_indices] = values
        # Set diagonal to 0 last -- self-distance is always 0 for shortest paths.
        # This intentionally overwrites any self-loop weights.
        dense.fill_diagonal_(0.0)
        return dense


@dataclass(frozen=True)
class SSSPResult:
    """Result of single-source shortest path computation."""

    distances: torch.Tensor
    predecessors: Optional[torch.Tensor] = None
    source: int = 0
    num_iterations: int = 0


@dataclass(frozen=True)
class TeacherLabels:
    """Distance labels from anchors to all vertices."""

    d_out: torch.Tensor
    d_in: torch.Tensor
    anchor_indices: torch.Tensor
    is_directed: bool


@dataclass(frozen=True)
class Embedding:
    """Hilbert or tropical embedding vectors."""

    phi: torch.Tensor
    kind: str
    is_directed: bool
    num_anchors: int
