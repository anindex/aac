#!/usr/bin/env python
"""Generate LaTeX table_osmnx.tex from results/osmnx/large_scale_results.csv.

Reads per-seed results, aggregates, and produces the final table.
Bold: best admissible method per budget per graph.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("results/osmnx/large_scale_results.csv")
OUTPUT_PATH = Path("paper/table_osmnx.tex")

# Memory budget definitions
# ALT K landmarks = 2*K*4 bytes/vertex (fwd+bwd for directed)
# AAC m dimensions = m*4 bytes/vertex (for each of fwd,bwd, but stored as m floats)
BUDGETS = [32, 64, 128]

# For each budget, which ALT K and which AAC configs to consider
ALT_K_BY_BUDGET = {32: 4, 64: 8, 128: 16}
# AAC bytes_per_vertex is m*4 for undirected, or 2*m*4 for directed?
# Actually looking at the CSV: bytes_per_vertex for AAC m=8 is 32, m=16 is 64, etc.
# So it's m*4 (selecting best K0 per m budget)

GRAPH_ORDER = [
    ("Small-scale", ["modena", "manhattan"]),
    ("City-scale", ["berlin", "los_angeles"]),
    ("Country-scale", ["netherlands"]),
]

GRAPH_DISPLAY = {
    "modena": ("Modena", "30K"),
    "manhattan": ("Manhattan", "4.6K"),
    "berlin": ("Berlin", "28K"),
    "los_angeles": ("Los Angeles", "50K"),
    "netherlands": ("Netherlands", "4.5M"),
}


def best_aac_at_budget(df, graph, budget):
    """Get best AAC reduction at a given budget (best K0 for that m)."""
    mask = (df["graph"] == graph) & (df["method"] == "AAC") & (df["bytes_per_vertex"] == budget)
    subset = df[mask]
    if subset.empty:
        return None, None

    # Group by K0 and get mean reduction per seed
    per_config = subset.groupby("K0").agg(
        red_mean=("expansion_reduction_pct", "mean"),
        red_std=("expansion_reduction_pct", "std"),
    ).reset_index()

    best = per_config.loc[per_config["red_mean"].idxmax()]
    return best["red_mean"], best["red_std"]


def alt_at_budget(df, graph, budget):
    """Get ALT reduction at a given budget."""
    K = ALT_K_BY_BUDGET[budget]
    mask = (
        (df["graph"] == graph) & (df["method"] == "ALT")
        & (df["num_landmarks_or_dims"] == K)
    )
    subset = df[mask]
    if subset.empty:
        return None, None
    return subset["expansion_reduction_pct"].mean(), subset["expansion_reduction_pct"].std()


def fmt(mean, std, bold=False):
    if mean is None:
        return "---"
    std_val = std if std == std else 0.0  # handle NaN
    s = f"${mean:.1f} \\pm {std_val:.1f}$"
    if bold:
        s = f"$\\mathbf{{{mean:.1f} \\pm {std_val:.1f}}}$"
    return s


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")
    print(f"Graphs: {sorted(df['graph'].unique())}")

    lines = []
    lines.append(
        "%%% PAPER-TABLE-PROVENANCE\n"
        "%%% generator:  scripts/generate_table_osmnx.py\n"
        "%%% sources:    results/osmnx/large_scale_results.csv\n"
        "%%% paper-ref:  Table tab:osmnx; Sec 5.3 (OSMnx)\n"
        "%%% verify:     python scripts/check_paper_consistency.py paper/table_osmnx.tex\n"
        "%%% END-PROVENANCE"
    )
    lines.append(r"""\begin{table}[t]
\centering
\caption{OSMnx road network results across scales: expansion reduction (\%) at matched memory budgets.
AAC rows show the best $K_0$ per budget (descriptive Pareto, same protocol as Table~\ref{tab:main_results}).
Bold: best admissible method per budget per graph.
All configurations verified at 0 admissibility violations.
Mean $\pm$ std over 5 seeds (3 for Netherlands).
\emph{Cross-table note:} AAC cells use a per-budget best-$K_0$ retrospective protocol; values for the same nominal $(K_0, m)$ may differ from Tables~\ref{tab:selection-ablation} and~\ref{tab:compression-curve} by training stochasticity within the reported standard deviation (see Section~\ref{sec:setup}, ``Cross-table protocol'').}
\label{tab:osmnx}
\small
\begin{tabular}{llrccc}
\toprule
\textbf{Scale / Graph} & \textbf{Method} & \textbf{Nodes} & \textbf{32 B/v} & \textbf{64 B/v} & \textbf{128 B/v} \\""")

    for scale_name, graphs in GRAPH_ORDER:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{scale_name}}}}} \\")
        lines.append(r"\midrule")

        for gi, graph in enumerate(graphs):
            if graph not in df["graph"].unique():
                continue

            display_name, nodes = GRAPH_DISPLAY[graph]

            # Gather values
            alt_vals = {}
            aac_vals = {}
            for b in BUDGETS:
                alt_vals[b] = alt_at_budget(df, graph, b)
                aac_vals[b] = best_aac_at_budget(df, graph, b)

            # Determine bold (best per budget)
            alt_cells = []
            aac_cells = []
            for b in BUDGETS:
                a_m, a_s = alt_vals[b]
                h_m, h_s = aac_vals[b]
                if a_m is not None and h_m is not None:
                    alt_bold = a_m > h_m
                    aac_bold = h_m > a_m
                elif a_m is not None:
                    alt_bold, aac_bold = True, False
                elif h_m is not None:
                    alt_bold, aac_bold = False, True
                else:
                    alt_bold, aac_bold = False, False

                alt_cells.append(fmt(a_m, a_s, alt_bold))
                aac_cells.append(fmt(h_m, h_s, aac_bold))

            # AAC row first (AAC is the proposed method; aesthetic convention)
            aac_row = f"\\rowcolor{{aaccolor}} {display_name} & AAC & {nodes} & {' & '.join(aac_cells)} \\\\"
            lines.append(aac_row)
            # ALT row second
            alt_row = f"                            & ALT & {nodes} & {' & '.join(alt_cells)} \\\\"
            lines.append(alt_row)

            if gi < len(graphs) - 1:
                lines.append(r"\midrule")

    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")

    table = "\n".join(lines)
    OUTPUT_PATH.write_text(table + "\n")
    print(f"Written to {OUTPUT_PATH}")
    print()
    print(table)


if __name__ == "__main__":
    main()
