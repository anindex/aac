"""Tests for PBF road network loader.

Uses pyosmium's SimpleWriter to create small synthetic PBF files
for testing the loader without external data dependencies.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

try:
    import osmium

    HAS_OSMIUM = True
except ImportError:
    HAS_OSMIUM = False

pytestmark = pytest.mark.skipif(not HAS_OSMIUM, reason="pyosmium not installed")


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
    writer = osmium.SimpleWriter(path)

    # Create nodes with distinct coordinates
    # Each node at (lon, lat) = (i * 0.01, 50.0 + i * 0.001)
    for i in range(1, 10):
        node = osmium.osm.mutable.Node(
            id=i,
            location=osmium.osm.Location(i * 0.01, 50.0 + i * 0.001),
        )
        writer.add_node(node)

    # Way 1: residential, bidirectional (default), nodes [1, 2, 3]
    way1 = osmium.osm.mutable.Way(
        id=100,
        nodes=[1, 2, 3],
        tags={"highway": "residential"},
    )
    writer.add_way(way1)

    # Way 2: motorway, oneway=yes, nodes [3, 4, 5]
    way2 = osmium.osm.mutable.Way(
        id=101,
        nodes=[3, 4, 5],
        tags={"highway": "motorway", "oneway": "yes"},
    )
    writer.add_way(way2)

    # Way 3: footway (non-driving), nodes [5, 6, 7]
    way3 = osmium.osm.mutable.Way(
        id=102,
        nodes=[5, 6, 7],
        tags={"highway": "footway"},
    )
    writer.add_way(way3)

    # Way 4: primary, oneway=-1 (reverse direction), nodes [7, 8, 9]
    way4 = osmium.osm.mutable.Way(
        id=103,
        nodes=[7, 8, 9],
        tags={"highway": "primary", "oneway": "-1"},
    )
    writer.add_way(way4)

    # Way 5: cycleway (non-driving), nodes [1, 9]
    way5 = osmium.osm.mutable.Way(
        id=104,
        nodes=[1, 9],
        tags={"highway": "cycleway"},
    )
    writer.add_way(way5)

    writer.close()


@pytest.fixture
def synthetic_pbf(tmp_path: Path) -> str:
    """Create a synthetic PBF file and return its path."""
    pbf_path = str(tmp_path / "test.osm.pbf")
    _write_synthetic_pbf(pbf_path)
    return pbf_path


def test_load_produces_directed_graph(synthetic_pbf: str) -> None:
    """Test 1: load_pbf_road_graph produces a directed Graph."""
    from aac.graphs.loaders.pbf import load_pbf_road_graph
    from aac.graphs.types import Graph

    graph = load_pbf_road_graph(synthetic_pbf)
    assert isinstance(graph, Graph)
    assert graph.is_directed is True
    assert graph.num_nodes > 0
    assert graph.num_edges > 0


def test_all_edge_weights_positive(synthetic_pbf: str) -> None:
    """Test 2: All edge weights are positive (no zero-length edges)."""
    from aac.graphs.loaders.pbf import load_pbf_road_graph

    graph = load_pbf_road_graph(synthetic_pbf)
    assert (graph.values > 0).all(), "All edge weights must be positive"


def test_node_ids_contiguous(synthetic_pbf: str) -> None:
    """Test 3: Node IDs are contiguous 0-indexed (no gaps)."""
    from aac.graphs.loaders.pbf import load_pbf_road_graph

    graph = load_pbf_road_graph(synthetic_pbf)
    # Check CSR structure: crow_indices should be (V+1,)
    assert graph.crow_indices.shape[0] == graph.num_nodes + 1
    # col_indices should be in [0, V)
    assert graph.col_indices.min() >= 0
    assert graph.col_indices.max() < graph.num_nodes


def test_coordinates_shape_and_finite(synthetic_pbf: str) -> None:
    """Test 4: Coordinates tensor has shape (V, 2) and values are finite."""
    from aac.graphs.loaders.pbf import load_pbf_road_graph

    graph = load_pbf_road_graph(synthetic_pbf)
    assert graph.coordinates is not None
    assert graph.coordinates.shape == (graph.num_nodes, 2)
    assert torch.isfinite(graph.coordinates).all()


def test_oneway_asymmetric_edges(synthetic_pbf: str) -> None:
    """Test 5: Oneway roads produce asymmetric edge sets; bidirectional produce both."""
    from aac.graphs.loaders.pbf import load_pbf_road_graph

    graph = load_pbf_road_graph(synthetic_pbf)
    rows, cols, _ = graph.to_coo()

    # Convert to set of (src, dst) for easy lookup
    edge_set = set(zip(rows.tolist(), cols.tolist()))

    # The graph should have more edges in one direction than the other
    # for oneway roads (edges are not symmetric overall)
    reverse_count = sum(1 for u, v in edge_set if (v, u) in edge_set)
    forward_only = len(edge_set) - reverse_count

    # Some edges should be one-directional (from oneway ways)
    assert forward_only > 0, "Oneway roads should produce asymmetric edges"


def test_non_driving_excluded(synthetic_pbf: str) -> None:
    """Test 6: Non-driving highway types (footway, cycleway) are excluded."""
    from aac.graphs.loaders.pbf import load_pbf_road_graph

    graph = load_pbf_road_graph(synthetic_pbf)

    # With only driving ways (residential [1,2,3], motorway [3,4,5], primary [7,8,9]):
    # Driving nodes used: {1,2,3,4,5,7,8,9} = 8 nodes
    # Node 6 is only in footway, so should be excluded
    # (mapped to contiguous IDs, so we just check count)
    assert graph.num_nodes == 8, (
        f"Expected 8 driving nodes but got {graph.num_nodes}. "
        "Non-driving highway types should be excluded."
    )
