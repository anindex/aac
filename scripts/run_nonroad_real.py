#!/usr/bin/env python
"""Non-road real-graph experiment: AAC vs ALT vs Hybrid on OGB-arXiv citation graph.

A real non-road graph at >=50K nodes. OGB-arXiv is a ~170K-node citation
network -- naturally directed but we symmetrize and add random uniform[1,10]
edge weights for parity with the synthetic SBM/BA experiments.

This experiment is the source of the pre-registered prediction reported in
the paper. The verbatim prediction text (filed prior to evaluation) is in
``results/README.md`` (Pre-registration Record section).

Output:
    results/synthetic/ogbn_arxiv_results.csv
    results/synthetic/ogbn_arxiv_log.txt

Usage:
    python scripts/run_nonroad_real.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup (mirror run_synthetic_experiments.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import networkx as nx
import numpy as np
import torch

# Reuse the exact experiment runner / CSV writer from the synthetic script
# so the schema is identical and table-generation code can be reused.
from scripts.run_synthetic_experiments import (
    BUDGET_LEVELS,
    NUM_QUERIES,
    QUERY_SEED,
    SEEDS,
    nx_to_graph,
    run_experiment,
    write_csv,
)

OUTPUT_DIR = _PROJECT_ROOT / "results" / "synthetic"
GRAPH_SEED = 42


def load_ogbn_arxiv() -> nx.Graph:
    """Load the OGB-arXiv citation graph as an undirected networkx graph.

    The graph has ~170K nodes and ~1.17M edges after symmetrization. We
    restrict to the largest weakly-connected component and relabel nodes
    to a contiguous 0..N-1 range.
    """
    from ogb.nodeproppred import NodePropPredDataset

    # OGB's cached preprocessed file is a plain torch pickle, but newer
    # torch versions (>=2.6) default to weights_only=True which refuses
    # arbitrary pickled Python objects. Monkey-patch torch.load for the
    # loader call to allow the legacy format.
    import torch as _torch
    _orig_load = _torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)
    _torch.load = _patched_load
    try:
        print("  Loading OGB-arxiv via ogb.nodeproppred...")
        dataset = NodePropPredDataset(
            name="ogbn-arxiv", root=str(_PROJECT_ROOT / "data" / "ogb")
        )
    finally:
        _torch.load = _orig_load
    graph, _ = dataset[0]
    edge_index = graph["edge_index"]  # shape (2, E)
    num_nodes = int(graph["num_nodes"])
    print(f"  Raw OGB: {num_nodes:,} nodes, {edge_index.shape[1]:,} directed edges")

    # Build undirected networkx graph from edge_index
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    src = edge_index[0].tolist()
    tgt = edge_index[1].tolist()
    G.add_edges_from(zip(src, tgt))
    print(f"  After symmetrization: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    # Restrict to largest connected component so all queries are feasible
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        print(f"  {len(components)} CCs; taking largest = {len(largest):,} nodes")
        G = G.subgraph(largest).copy()

    # Relabel 0..N-1 contiguously
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    print(f"  Final LCC: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  OGB-ARXIV CITATION GRAPH (real non-road, ~170K nodes)")
    print("  Pre-registered prediction: see results/README.md (Pre-registration Record)")
    print(f"{'='*70}")

    G = load_ogbn_arxiv()
    print(f"  NetworkX: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    graph = nx_to_graph(G, weight_seed=GRAPH_SEED)
    print(f"  Graph: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges "
          f"(directed={graph.is_directed})")

    rows = run_experiment(
        graph_type="ogbn_arxiv",
        graph=graph,
        budget_levels=BUDGET_LEVELS,
        seeds=SEEDS,
        num_queries=NUM_QUERIES,
        query_seed=QUERY_SEED,
    )
    write_csv(rows, OUTPUT_DIR / "ogbn_arxiv_results.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
