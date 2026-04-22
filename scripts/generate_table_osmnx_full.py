#!/usr/bin/env python
"""Generate LaTeX table_osmnx_full.tex from results/osmnx/large_scale_results.csv.

Produces the detailed per-configuration table for the appendix.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

INPUT_PATH = Path("results/osmnx/large_scale_results.csv")
OUTPUT_PATH = Path("paper/table_osmnx_full.tex")

GRAPH_ORDER = ["manhattan", "modena", "berlin", "los_angeles", "netherlands"]
GRAPH_DISPLAY = {
    "manhattan": "Manhattan (4.6K)",
    "modena": "Modena (30K)",
    "berlin": "Berlin (28K)",
    "los_angeles": "Los Angeles (50K)",
    "netherlands": "Netherlands (4.5M)",
}

ALT_K_VALUES = [4, 8, 16, 32]  # corresponds to 32, 64, 128, 256 B/v
HIT_CONFIGS = [(16, 8), (32, 8), (32, 16), (64, 16), (64, 32)]


def fmt(mean, std, bold=False):
    std_val = std if not np.isnan(std) else 0.0
    if bold:
        return f"$\\mathbf{{{mean:.1f} \\pm {std_val:.1f}}}$"
    return f"${mean:.1f} \\pm {std_val:.1f}$"


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    lines = []
    lines.append(
        "%%% PAPER-TABLE-PROVENANCE\n"
        "%%% generator:  scripts/generate_table_osmnx_full.py\n"
        "%%% sources:    results/osmnx/large_scale_results.csv\n"
        "%%% paper-ref:  Table tab:osmnx-full; Sec 5.3 (OSMnx full sweep)\n"
        "%%% verify:     python scripts/check_paper_consistency.py paper/table_osmnx_full.tex\n"
        "%%% END-PROVENANCE"
    )
    lines.append(r"""\begin{table*}[h!]
\centering
\caption{Full per-configuration OSMnx results. Mean $\pm$ std over 5 seeds (3 for Netherlands). Bold: best reduction per (graph, memory budget). All configurations verified at 0 admissibility violations.}
\label{tab:osmnx-full}
\small
\begin{tabular}{llrrrrr}
\toprule
\textbf{Graph} & \textbf{Method} & $K_0$ & $m$ & \textbf{B/v} & \textbf{Reduction (\%)} & \textbf{Latency (ms)} \\
\midrule""")

    available_graphs = [g for g in GRAPH_ORDER if g in df["graph"].unique()]

    for gi, graph in enumerate(available_graphs):
        gdf = df[df["graph"] == graph]
        display = GRAPH_DISPLAY.get(graph, graph)

        # Collect all configs with their budget and stats
        configs = []

        # ALT configs
        alt = gdf[gdf["method"] == "ALT"]
        for _, row in alt.groupby("num_landmarks_or_dims").agg(
            bpv=("bytes_per_vertex", "first"),
            red_mean=("expansion_reduction_pct", "mean"),
            red_std=("expansion_reduction_pct", "std"),
            lat_mean=("mean_query_latency_ms", "mean"),
        ).reset_index().iterrows():
            configs.append({
                "method": "ALT", "K0": "--", "m": "--",
                "bpv": int(row["bpv"]),
                "red_mean": row["red_mean"],
                "red_std": row["red_std"],
                "lat_mean": row["lat_mean"],
            })

        # AAC configs
        aac_rows = gdf[gdf["method"] == "AAC"]
        for k0, m in HIT_CONFIGS:
            mask = (aac_rows["K0"] == k0) & (aac_rows["m"] == m)
            sub = aac_rows[mask]
            if sub.empty:
                continue
            configs.append({
                "method": "AAC", "K0": k0, "m": m,
                "bpv": int(sub["bytes_per_vertex"].iloc[0]),
                "red_mean": sub["expansion_reduction_pct"].mean(),
                "red_std": sub["expansion_reduction_pct"].std(ddof=1),
                "lat_mean": sub["mean_query_latency_ms"].mean(),
            })

        # Determine best per budget
        budget_best = {}
        for c in configs:
            b = c["bpv"]
            if b not in budget_best or c["red_mean"] > budget_best[b]:
                budget_best[b] = c["red_mean"]

        # Generate rows
        # AAC rows first (AAC is the proposed method; aesthetic convention), then ALT
        alt_configs = [c for c in configs if c["method"] == "ALT"]
        aac_configs = [c for c in configs if c["method"] == "AAC"]

        first = True
        for c in sorted(aac_configs, key=lambda x: (x["bpv"], x["K0"])):
            bold = abs(c["red_mean"] - budget_best[c["bpv"]]) < 0.05
            prefix = f" {display}" if first else " "
            first = False
            red_str = fmt(c["red_mean"], c["red_std"], bold)
            lat = int(round(c["lat_mean"]))
            lines.append(f"\\rowcolor{{aaccolor}}   {prefix} & AAC & {c['K0']} & {c['m']} & {c['bpv']} & {red_str} & {lat} \\\\")

        for c in sorted(alt_configs, key=lambda x: x["bpv"]):
            bold = abs(c["red_mean"] - budget_best[c["bpv"]]) < 0.05
            red_str = fmt(c["red_mean"], c["red_std"], bold)
            lat = int(round(c["lat_mean"]))
            lines.append(f"   & ALT & -- & -- & {c['bpv']} & {red_str} & {lat} \\\\")

        if gi < len(available_graphs) - 1:
            lines.append(r"\midrule")

    lines.append(r"""\bottomrule
\end{tabular}
\end{table*}""")

    table = "\n".join(lines)
    OUTPUT_PATH.write_text(table + "\n")
    print(f"Written to {OUTPUT_PATH}")
    print()
    print(table)


if __name__ == "__main__":
    main()
