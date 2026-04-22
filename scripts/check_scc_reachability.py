#!/usr/bin/env python
"""SCC reachability check for the paper's query protocol.

For each DIMACS road graph and OSMnx city graph, computes the fraction of
the standard 100-query sample (`generate_queries(graph, 100, seed=42)`)
whose endpoints lie in a common strongly connected component. By
construction (`generate_queries` already restricts to the strong LCC for
directed graphs), this fraction is 100% on every graph in the paper --
this script makes that guarantee auditable rather than implicit.

Output: results/scc_reachability.csv

Used in Sec.2.1 to back the one-sentence reachability statement.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import scipy.sparse.csgraph

from aac.graphs.convert import graph_to_scipy
from aac.graphs.io import load_graph_npz
from aac.graphs.loaders.dimacs import load_dimacs
from experiments.utils import _strong_cc_labels, generate_queries

DATA = _PROJECT_ROOT / "data"
OUT = _PROJECT_ROOT / "results" / "scc_reachability.csv"
NUM_QUERIES = 100
QUERY_SEED = 42

DIMACS = [
    ("NY", DATA / "dimacs" / "USA-road-d.NY.gr", DATA / "dimacs" / "USA-road-d.NY.co"),
    ("BAY", DATA / "dimacs" / "USA-road-d.BAY.gr", DATA / "dimacs" / "USA-road-d.BAY.co"),
    ("COL", DATA / "dimacs" / "USA-road-d.COL.gr", DATA / "dimacs" / "USA-road-d.COL.co"),
    ("FLA", DATA / "dimacs" / "USA-road-d.FLA.gr", DATA / "dimacs" / "USA-road-d.FLA.co"),
]
OSMNX = [
    ("modena", DATA / "osmnx" / "modena.npz"),
    ("manhattan", DATA / "osmnx" / "manhattan.npz"),
    ("berlin", DATA / "osmnx" / "berlin.npz"),
    ("los_angeles", DATA / "osmnx" / "los_angeles.npz"),
]


def fraction_in_common_scc(graph) -> tuple[float, int, int, int]:
    sp = graph_to_scipy(graph)
    is_directed = getattr(graph, "is_directed", False)
    if is_directed:
        n_components, labels = _strong_cc_labels(sp, graph.num_nodes)
    else:
        n_components, labels = scipy.sparse.csgraph.connected_components(
            sp, directed=False, connection="weak"
        )
    queries = generate_queries(graph, NUM_QUERIES, seed=QUERY_SEED)
    common = sum(1 for s, t in queries if labels[s] == labels[t])
    return common / len(queries), common, len(queries), n_components


def main() -> int:
    rows = []

    for name, gr, co in DIMACS:
        if not gr.exists():
            print(f"[skip] {name}: {gr} missing")
            continue
        print(f"[dimacs:{name}] loading ...")
        graph = load_dimacs(str(gr), str(co) if co.exists() else None)
        frac, n_common, n_total, n_comp = fraction_in_common_scc(graph)
        print(f"  V={graph.num_nodes:,} components={n_comp:,} "
              f"common-SCC fraction={frac*100:.1f}% ({n_common}/{n_total})")
        rows.append({
            "family": "DIMACS", "graph": name,
            "num_nodes": graph.num_nodes, "is_directed": graph.is_directed,
            "num_components": n_comp,
            "common_scc_fraction": frac,
            "common_scc_count": n_common, "num_queries": n_total,
        })

    for name, npz in OSMNX:
        if not npz.exists():
            print(f"[skip] {name}: {npz} missing")
            continue
        print(f"[osmnx:{name}] loading ...")
        graph = load_graph_npz(npz)
        frac, n_common, n_total, n_comp = fraction_in_common_scc(graph)
        print(f"  V={graph.num_nodes:,} components={n_comp:,} "
              f"common-SCC fraction={frac*100:.1f}% ({n_common}/{n_total})")
        rows.append({
            "family": "OSMnx", "graph": name,
            "num_nodes": graph.num_nodes, "is_directed": graph.is_directed,
            "num_components": n_comp,
            "common_scc_fraction": frac,
            "common_scc_count": n_common, "num_queries": n_total,
        })

    if not rows:
        print("No graphs available; nothing to do.")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT}")
    fracs = [r["common_scc_fraction"] for r in rows]
    print(f"Min common-SCC fraction across graphs: {min(fracs)*100:.1f}%")
    print(f"Max common-SCC fraction across graphs: {max(fracs)*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
