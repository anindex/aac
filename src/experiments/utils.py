"""Experiment utilities: seeding, query generation, memory accounting."""

from __future__ import annotations

import os
import random

import numpy as np
import scipy.sparse.csgraph
import torch

from aac.graphs.convert import graph_to_scipy
from aac.graphs.types import Graph

# Prefer igraph for directed SCCs when available. In practice it is much faster
# than SciPy's connected-components path even on mid-size road graphs such as
# DIMACS NY, not only on the multi-million-node cases.
_IGRAPH_THRESHOLD = 0


def _strong_cc_labels(sp: "scipy.sparse.csr_matrix", V: int) -> tuple[int, np.ndarray]:
    """Compute strong CC labels, using igraph for large graphs.

    Returns (n_components, labels) matching SciPy convention.
    """
    if V > _IGRAPH_THRESHOLD:
        try:
            import igraph

            rows, cols = sp.nonzero()
            g = igraph.Graph(
                n=V, edges=list(zip(rows.tolist(), cols.tolist())), directed=True,
            )
            membership = g.connected_components(mode="strong").membership
            labels = np.array(membership, dtype=np.int32)
            return int(labels.max() + 1), labels
        except ImportError:
            pass  # Fall through to SciPy

    return scipy.sparse.csgraph.connected_components(
        sp, directed=True, connection="strong",
    )


def seed_everything(seed: int) -> None:
    """Set all RNG states for full reproducibility.

    Sets Python, NumPy, PyTorch CPU/CUDA seeds, and enables
    deterministic CUDA operations.

    Args:
        seed: Integer seed for all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_queries(
    graph: Graph,
    num_queries: int,
    seed: int = 42,
    mode: str = "uniform",
    hotspot_nodes: int = 8,
    hotspot_mix: float = 0.85,
    powerlaw_alpha: float = 1.5,
) -> list[tuple[int, int]]:
    """Generate query pairs from the largest connected component.

    Extracts connected components via SciPy/igraph, restricts sampling to
    the largest (strong) component, and ensures source != target for each
    pair.

    Supported modes:
    - ``uniform``: uniform endpoint sampling over the largest component
      (the default; preserves backward compatibility with callers that do
      not pass ``mode``).
    - ``hotspot``: most endpoints are sampled from a small hotspot set,
      with a uniform backoff controlled by ``hotspot_mix``.
    - ``powerlaw``: endpoints are sampled from a degree-weighted
      power-law-like distribution, emphasizing high-degree traffic hubs.

    The ``hotspot`` and ``powerlaw`` modes are required by
    ``scripts/run_query_distribution_experiments.py`` (Section 5.6 of the
    paper, Table~\\ref{tab:query-distribution}); they were previously
    served from a duplicate ``experiments/utils.py`` at the repo root,
    which is now consolidated here.

    Args:
        graph: Graph in CSR format.
        num_queries: Number of query pairs to generate.
        seed: RNG seed for reproducible sampling.
        mode: Query-distribution mode: ``uniform``, ``hotspot``, or
            ``powerlaw``.
        hotspot_nodes: Number of hotspot endpoints when ``mode='hotspot'``.
        hotspot_mix: Probability of sampling from the hotspot set instead
            of the full component when ``mode='hotspot'``.
        powerlaw_alpha: Exponent applied to degree weights when
            ``mode='powerlaw'``.

    Returns:
        List of (source, target) tuples.
    """
    sp = graph_to_scipy(graph)
    is_directed = getattr(graph, "is_directed", False)
    # Use strong CC for directed graphs so that queries match the LCC used
    # by AAC/ALT preprocessing (which restricts landmarks to the strong
    # LCC). Without this, queries can span SCC boundaries, resulting in
    # sentinel-heavy heuristics and unfair evaluation.
    if is_directed:
        n_components, labels = _strong_cc_labels(sp, graph.num_nodes)
    else:
        n_components, labels = scipy.sparse.csgraph.connected_components(
            sp, directed=False, connection="weak",
        )

    # Find the largest connected component
    component_sizes = np.bincount(labels)
    largest_component = int(np.argmax(component_sizes))
    component_nodes = np.where(labels == largest_component)[0]

    rng = np.random.RandomState(seed)

    if mode == "uniform":
        component_probs = None
        hotspot_set = None
    elif mode == "hotspot":
        num_hotspots = max(2, min(int(hotspot_nodes), len(component_nodes)))
        hotspot_set = rng.choice(component_nodes, size=num_hotspots, replace=False)
        component_probs = None
    elif mode == "powerlaw":
        out_degree = np.diff(sp.indptr)[component_nodes].astype(np.float64)
        in_degree = np.bincount(sp.indices, minlength=graph.num_nodes)[
            component_nodes
        ].astype(np.float64)
        degree_weights = (
            np.maximum(out_degree + in_degree, 1.0) ** float(powerlaw_alpha)
        )
        component_probs = degree_weights / degree_weights.sum()
        hotspot_set = None
    else:
        raise ValueError(
            f"Unsupported query mode {mode!r}. Expected one of: "
            "uniform, hotspot, powerlaw"
        )

    def sample_endpoint() -> int:
        if mode == "hotspot" and rng.rand() < hotspot_mix:
            return int(rng.choice(hotspot_set))
        if component_probs is None:
            return int(rng.choice(component_nodes))
        return int(rng.choice(component_nodes, p=component_probs))

    queries: list[tuple[int, int]] = []
    while len(queries) < num_queries:
        s = sample_endpoint()
        t = sample_endpoint()
        if s != t:
            queries.append((s, t))

    return queries


def compute_strong_lcc(graph: Graph) -> tuple[np.ndarray, int]:
    """Compute largest strongly/weakly connected component.

    Uses strong CC for directed graphs, weak CC for undirected.
    Falls back to igraph for large directed graphs (>500K nodes).

    Args:
        graph: Graph in CSR format.

    Returns:
        (lcc_nodes, seed_vertex) where lcc_nodes is a sorted int array
        and seed_vertex is the first node in the LCC.
    """
    sp = graph_to_scipy(graph)
    is_directed = getattr(graph, "is_directed", False)
    if is_directed:
        _, labels = _strong_cc_labels(sp, graph.num_nodes)
    else:
        _, labels = scipy.sparse.csgraph.connected_components(
            sp, directed=False, connection="weak",
        )
    sizes = np.bincount(labels)
    largest = int(np.argmax(sizes))
    lcc_nodes = np.where(labels == largest)[0]
    return lcc_nodes, int(lcc_nodes[0])


def memory_bytes_per_vertex(num_dims: int, dtype_size: int = 4) -> int:
    """Compute memory bytes per vertex for embedding storage.

    For ALT, caller passes 2*K (forward + backward labels).
    For FastMap or AAC, caller passes m (embedding dimensions).

    Args:
        num_dims: Number of dimensions/values stored per vertex.
        dtype_size: Bytes per element. Default 4 (float32) per METR-05.

    Returns:
        Total bytes per vertex: num_dims * dtype_size.
    """
    return num_dims * dtype_size
