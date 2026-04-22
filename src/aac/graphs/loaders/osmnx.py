"""OSMnx road network downloader and converter.

Downloads city/region road networks via OSMnx and converts them to Graph instances.
Handles MultiDiGraph cleanup: self-loop removal, parallel edge deduplication (keep min weight),
and node ID remapping to contiguous 0-indexed integers.

OSMnx uses y=lat, x=lon convention for node coordinates.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import torch

from aac.graphs.convert import edges_to_graph
from aac.graphs.types import Graph


def load_osmnx(
    place: str,
    network_type: str = "drive",
) -> Graph:
    """Download an OSMnx road network and convert to a Graph.

    Requires the ``osmnx`` package (optional dependency).

    Args:
        place: Place name for OSMnx query (e.g., "Piedmont, California, USA").
        network_type: OSMnx network type ("drive", "walk", "bike", "all").

    Returns:
        Directed Graph with fp64 edge weights (meters) and (V, 2) coordinates (lon, lat).
    """
    try:
        import osmnx as ox
    except ImportError as e:
        raise ImportError(
            "osmnx is required for load_osmnx. Install with: pip install osmnx"
        ) from e

    # simplify=True is the default in OSMnx 2.x, so graph is already simplified
    G = ox.graph_from_place(place, network_type=network_type, simplify=True)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Resolve parallel edges (MultiDiGraph -> simple DiGraph)
    G = _resolve_parallel_edges(G)

    return _networkx_digraph_to_graph(G)


def _resolve_parallel_edges(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Convert a MultiDiGraph to a simple DiGraph keeping min-weight parallel edges.

    For each (u, v) pair with multiple edges, keeps the edge with the minimum
    "length" attribute. Falls back to the first edge if "length" is missing.

    Args:
        G: NetworkX MultiDiGraph (possibly with parallel edges).

    Returns:
        Simple NetworkX DiGraph with no parallel edges.
    """
    simple = nx.DiGraph()

    # Copy all nodes with their attributes
    for node, data in G.nodes(data=True):
        simple.add_node(node, **data)

    # For each (u, v) pair, keep the edge with minimum length
    for u, v, data in G.edges(data=True):
        if simple.has_edge(u, v):
            existing_length = simple[u][v].get("length", float("inf"))
            new_length = data.get("length", float("inf"))
            if new_length < existing_length:
                simple[u][v].update(data)
        else:
            simple.add_edge(u, v, **data)

    return simple


def _networkx_digraph_to_graph(
    G: nx.DiGraph,
    weight_attr: str = "length",
    default_weight: float = 1.0,
) -> Graph:
    """Convert a NetworkX DiGraph to a Graph with contiguous 0-indexed node IDs.

    Args:
        G: NetworkX DiGraph (simple, no self-loops, no parallel edges).
        weight_attr: Edge attribute to use as weight.
        default_weight: Default weight if attribute is missing.

    Returns:
        Directed Graph with fp64 weights and optional (V, 2) coordinates.
    """
    # Remove any remaining self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Build node ID mapping: original -> contiguous 0-indexed
    node_list = sorted(G.nodes())
    node_map = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)

    # Extract edges
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []

    for u, v, data in G.edges(data=True):
        sources.append(node_map[u])
        targets.append(node_map[v])
        weights.append(float(data.get(weight_attr, default_weight)))

    # Extract coordinates (lon, lat) from node attributes
    coordinates: Optional[torch.Tensor] = None
    if node_list and "x" in G.nodes[node_list[0]] and "y" in G.nodes[node_list[0]]:
        coords = []
        for n in node_list:
            lon = float(G.nodes[n]["x"])
            lat = float(G.nodes[n]["y"])
            coords.append((lon, lat))
        coordinates = torch.tensor(coords, dtype=torch.float64)

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
