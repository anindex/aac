"""PBF road network loader using pyosmium.

Downloads and parses OpenStreetMap PBF files to extract driving road networks.
Handles oneway tags, filters non-driving highway types, computes haversine
edge weights, and produces directed Graph instances with contiguous 0-indexed
node IDs and (lon, lat) coordinates.
"""

from __future__ import annotations

import logging
import math

import torch

from aac.graphs.convert import edges_to_graph
from aac.graphs.types import Graph

logger = logging.getLogger(__name__)

# Highway types suitable for driving (excludes pedestrian, cycling, service)
HIGHWAY_DRIVING = frozenset({
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "living_street",
})

# Earth radius in meters for haversine computation
_EARTH_RADIUS_M = 6_371_000.0


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute the great-circle distance between two (lon, lat) points in meters.

    Uses the haversine formula with R = 6371000 m.

    Args:
        lon1: Longitude of point 1 (degrees).
        lat1: Latitude of point 1 (degrees).
        lon2: Longitude of point 2 (degrees).
        lat2: Latitude of point 2 (degrees).

    Returns:
        Distance in meters. Always non-negative.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return _EARTH_RADIUS_M * c


def load_pbf_road_graph(pbf_path: str) -> Graph:
    """Load a driving road network from an OpenStreetMap PBF file.

    Extracts road ways with driving-suitable highway tags, computes haversine
    edge weights in meters, handles oneway roads, and produces a directed Graph
    with contiguous 0-indexed node IDs and (lon, lat) coordinates.

    Requires the ``osmium`` package (``pip install osmium``).

    Args:
        pbf_path: Path to the .osm.pbf file.

    Returns:
        Directed Graph with fp64 edge weights (meters) and (V, 2) coordinates.

    Raises:
        ImportError: If pyosmium is not installed.
    """
    try:
        import osmium
    except ImportError as e:
        raise ImportError(
            "pyosmium is required for load_pbf_road_graph. "
            "Install with: pip install osmium"
        ) from e

    class RoadNetworkHandler(osmium.SimpleHandler):
        """Extracts driving road network from PBF via two-pass or single-pass approach."""

        def __init__(self) -> None:
            super().__init__()
            self.node_coords: dict[int, tuple[float, float]] = {}
            self.edges: list[tuple[int, int, float]] = []

        def node(self, n: osmium.osm.Node) -> None:
            """Store node coordinates (lon, lat)."""
            if n.location.valid():
                self.node_coords[n.id] = (n.location.lon, n.location.lat)

        def way(self, w: osmium.osm.Way) -> None:
            """Extract edges from driving-suitable ways with haversine weights."""
            highway = w.tags.get("highway", "")
            if highway not in HIGHWAY_DRIVING:
                return

            # Determine oneway behavior
            oneway_tag = w.tags.get("oneway", "no")
            is_forward_only = oneway_tag in ("yes", "true", "1")
            is_reverse_only = oneway_tag == "-1"

            # Extract node refs
            refs = [n.ref for n in w.nodes]

            # Create edges between consecutive node pairs
            for i in range(len(refs) - 1):
                from_id = refs[i]
                to_id = refs[i + 1]

                if from_id not in self.node_coords or to_id not in self.node_coords:
                    continue

                lon1, lat1 = self.node_coords[from_id]
                lon2, lat2 = self.node_coords[to_id]

                dist = _haversine_m(lon1, lat1, lon2, lat2)
                if dist <= 0:
                    continue

                if is_reverse_only:
                    # Reverse oneway: only to_id -> from_id
                    self.edges.append((to_id, from_id, dist))
                elif is_forward_only:
                    # Forward oneway: only from_id -> to_id
                    self.edges.append((from_id, to_id, dist))
                else:
                    # Bidirectional: both directions
                    self.edges.append((from_id, to_id, dist))
                    self.edges.append((to_id, from_id, dist))

    handler = RoadNetworkHandler()
    handler.apply_file(pbf_path, locations=True)

    if not handler.edges:
        raise ValueError(f"No driving road edges found in {pbf_path}")

    # Collect all used node IDs from edges, create contiguous mapping
    used_nodes: set[int] = set()
    for src, tgt, _ in handler.edges:
        used_nodes.add(src)
        used_nodes.add(tgt)

    sorted_nodes = sorted(used_nodes)
    node_map = {osm_id: idx for idx, osm_id in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)

    # Build tensors
    sources = torch.tensor([node_map[e[0]] for e in handler.edges], dtype=torch.int64)
    targets = torch.tensor([node_map[e[1]] for e in handler.edges], dtype=torch.int64)
    weights = torch.tensor([e[2] for e in handler.edges], dtype=torch.float64)

    # Build coordinates tensor (lon, lat) for mapped nodes
    coordinates = torch.tensor(
        [handler.node_coords[osm_id] for osm_id in sorted_nodes],
        dtype=torch.float64,
    )

    # Validate coordinates are finite (T-18-01 mitigation)
    if not torch.isfinite(coordinates).all():
        logger.warning("Non-finite coordinates detected, replacing with 0.0")
        coordinates = torch.where(
            torch.isfinite(coordinates),
            coordinates,
            torch.zeros_like(coordinates),
        )

    logger.info(
        "PBF loader: %d nodes, %d edges from %s",
        num_nodes,
        len(handler.edges),
        pbf_path,
    )

    return edges_to_graph(
        sources=sources,
        targets=targets,
        weights=weights,
        num_nodes=num_nodes,
        is_directed=True,
        coordinates=coordinates,
    )
