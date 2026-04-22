"""LaTeX table generation with bold best values per graph."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_comparison_table(
    df: pd.DataFrame,
    output_tex: str,
    metrics: list[str] | None = None,
    bold_best: bool = True,
    caption: str = "AAC vs baselines at equal bytes/vertex.",
    label: str = "tab:main_results",
) -> None:
    """Generate a LaTeX comparison table from aggregated results.

    For each metric column within each graph group, the minimum (best)
    value is wrapped in ``\\textbf{}``.

    Args:
        df: Aggregated results DataFrame with columns including
            graph, method, and the specified metrics.
        output_tex: Path for the output ``.tex`` file.
        metrics: List of metric columns to include. Defaults to
            ``["expansions_mean", "p50_ms", "memory_bytes_per_vertex",
            "preprocess_total_sec"]``.
        bold_best: Whether to bold the best (minimum) value per graph
            for each metric.
        caption: LaTeX table caption.
        label: LaTeX table label.
    """
    if metrics is None:
        metrics = [
            "expansions_mean",
            "p50_ms",
            "memory_bytes_per_vertex",
            "preprocess_total_sec",
        ]

    # Select columns that exist in the DataFrame
    display_cols = ["graph", "method"]
    for m in metrics:
        if m in df.columns:
            display_cols.append(m)

    table_df = df[display_cols].copy()

    if bold_best:
        table_df = _bold_best_values(table_df, metrics)

    # Pretty column names for LaTeX
    col_rename = {
        "graph": "Graph",
        "method": "Method",
        "expansions_mean": "Exp. (mean)",
        "expansions_median": "Exp. (median)",
        "p50_ms": "p50 (ms)",
        "p95_ms": "p95 (ms)",
        "memory_bytes_per_vertex": "Bytes/vtx",
        "preprocess_total_sec": "Preproc. (s)",
        "num_violations": "Violations",
    }
    table_df = table_df.rename(columns={k: v for k, v in col_rename.items() if k in table_df.columns})

    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)

    latex_str = table_df.to_latex(
        index=False,
        escape=False,
        float_format="%.1f",
        caption=caption,
        label=label,
    )

    with open(output_tex, "w") as f:
        f.write(latex_str)


def generate_preprocessing_table(
    df: pd.DataFrame,
    output_tex: str,
) -> None:
    """Generate a LaTeX preprocessing timing breakdown table.

    Columns: method, graph, anchor_selection_sec, sssp_sec, training_sec,
    total_sec. Bolds the minimum total per graph.

    Args:
        df: DataFrame with preprocessing timing columns.
        output_tex: Path for the output ``.tex`` file.
    """
    preproc_cols = [
        "method",
        "graph",
        "anchor_selection_sec",
        "sssp_sec",
        "training_sec",
        "total_sec",
    ]
    available = [c for c in preproc_cols if c in df.columns]
    table_df = df[available].copy()

    # Bold minimum total per graph
    if "total_sec" in table_df.columns and "graph" in table_df.columns:
        table_df = _bold_best_values(table_df, ["total_sec"])

    col_rename = {
        "method": "Method",
        "graph": "Graph",
        "anchor_selection_sec": "Anchor (s)",
        "sssp_sec": "SSSP (s)",
        "training_sec": "Training (s)",
        "total_sec": "Total (s)",
    }
    table_df = table_df.rename(columns={k: v for k, v in col_rename.items() if k in table_df.columns})

    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)
    latex_str = table_df.to_latex(
        index=False,
        escape=False,
        float_format="%.2f",
        caption="Preprocessing time breakdown.",
        label="tab:preprocessing",
    )

    with open(output_tex, "w") as f:
        f.write(latex_str)


def _bold_best_in_group(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_col: str,
    maximize: bool = True,
    fmt: str = "%.1f",
) -> pd.DataFrame:
    """Bold the best value per group, returning the column as formatted strings.

    Args:
        df: DataFrame to modify (in-place on a copy).
        group_cols: Columns defining groups.
        metric_col: The metric column to bold-best.
        maximize: If True, bold the maximum; if False, bold the minimum.
        fmt: Format string for float values.

    Returns:
        Modified DataFrame with metric_col as formatted strings.
    """
    result = df.copy()
    numeric_vals = pd.to_numeric(result[metric_col], errors="coerce")

    # Format the column as strings
    result[metric_col] = result[metric_col].apply(
        lambda v: fmt % v if isinstance(v, (int, float)) else str(v)
    )

    for _, group_idx in result.groupby(group_cols).groups.items():
        group_numeric = numeric_vals.loc[group_idx]
        if group_numeric.isna().all():
            continue
        best_idx = group_numeric.idxmax() if maximize else group_numeric.idxmin()
        result.loc[best_idx, metric_col] = (
            f"\\textbf{{{result.loc[best_idx, metric_col]}}}"
        )

    return result


def generate_dimacs_main_table(
    df_ny: pd.DataFrame,
    df_fla: pd.DataFrame,
    df_wilcoxon: pd.DataFrame,
    output_tex: str,
) -> None:
    """Generate main DIMACS comparison table: AAC vs ALT vs FastMap at equal memory.

    Shows methods at 32, 64, 128 bytes/vertex. For AAC, picks the best K0
    configuration at each memory level. Bolds best expansion_reduction_pct
    per (graph, bytes_per_vertex) group. Adds dagger significance marker
    for the AAC 64 bytes/vertex row when Wilcoxon two-sided p < 1e-5.

    Args:
        df_ny: Pareto sweep data for NY.
        df_fla: Pareto sweep data for FLA.
        df_wilcoxon: Wilcoxon test results.
        output_tex: Output .tex file path.
    """
    budgets = [32, 64, 128]
    rows: list[dict] = []

    for graph_name, df in [("NY", df_ny), ("FLA", df_fla)]:
        for budget in budgets:
            # AAC: best K0 config at this budget
            aac_at_budget = df[
                (df["method"] == "AAC") & (df["bytes_per_vertex"] == budget)
            ]
            if not aac_at_budget.empty:
                best_aac = aac_at_budget.loc[
                    aac_at_budget["expansion_reduction_pct"].idxmax()
                ]
                rows.append(
                    {
                        "Graph": graph_name,
                        "Method": "AAC",
                        "Bytes/V": budget,
                        "Exp. Red. (\\%)": best_aac["expansion_reduction_pct"],
                        "Mean Exp.": best_aac["mean_expansions"],
                        "Preproc. (s)": best_aac["preprocess_sec"],
                    }
                )

            # ALT at this budget
            alt_at_budget = df[
                (df["method"] == "ALT") & (df["bytes_per_vertex"] == budget)
            ]
            if not alt_at_budget.empty:
                alt_row = alt_at_budget.iloc[0]
                rows.append(
                    {
                        "Graph": graph_name,
                        "Method": "ALT",
                        "Bytes/V": budget,
                        "Exp. Red. (\\%)": alt_row["expansion_reduction_pct"],
                        "Mean Exp.": alt_row["mean_expansions"],
                        "Preproc. (s)": alt_row["preprocess_sec"],
                    }
                )

            # FastMap at this budget
            fm_at_budget = df[
                (df["method"] == "FastMap") & (df["bytes_per_vertex"] == budget)
            ]
            if not fm_at_budget.empty:
                fm_row = fm_at_budget.iloc[0]
                rows.append(
                    {
                        "Graph": graph_name,
                        "Method": "FastMap",
                        "Bytes/V": budget,
                        "Exp. Red. (\\%)": fm_row["expansion_reduction_pct"],
                        "Mean Exp.": fm_row["mean_expansions"],
                        "Preproc. (s)": fm_row["preprocess_sec"],
                    }
                )

    table_df = pd.DataFrame(rows)

    # Bold best expansion_reduction_pct per (Graph, Bytes/V) group
    table_df = _bold_best_in_group(
        table_df,
        group_cols=["Graph", "Bytes/V"],
        metric_col="Exp. Red. (\\%)",
        maximize=True,
    )

    # Format remaining numeric columns
    table_df["Mean Exp."] = table_df["Mean Exp."].apply(
        lambda v: f"{v:.0f}" if isinstance(v, (int, float)) else str(v)
    )
    table_df["Preproc. (s)"] = table_df["Preproc. (s)"].apply(
        lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else str(v)
    )

    # Add significance markers: dagger for AAC at 64 bytes/vertex where Wilcoxon
    # two-sided p < 1e-5 (the Wilcoxon test was run at K0=64 m=16 = 64 bytes/v)
    for graph_name in ["NY", "FLA"]:
        wilcoxon_row = df_wilcoxon[df_wilcoxon["graph"] == graph_name]
        if not wilcoxon_row.empty:
            p_two = wilcoxon_row["p_value_twosided"].values[0]
            if p_two < 1e-5:
                mask = (
                    (table_df["Graph"] == graph_name)
                    & (table_df["Method"] == "AAC")
                    & (table_df["Bytes/V"] == 64)
                )
                idx = table_df.index[mask]
                if len(idx) > 0:
                    current = table_df.loc[idx[0], "Exp. Red. (\\%)"]
                    table_df.loc[idx[0], "Exp. Red. (\\%)"] = (
                        current + "$^{\\dagger}$"
                    )

    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)

    latex_str = table_df.to_latex(
        index=False,
        escape=False,
        caption=(
            "AAC vs ALT and FastMap at equal memory budgets on DIMACS "
            "road networks. Best expansion reduction per (graph, memory) group "
            "in bold. FastMap achieves the highest raw expansion reduction but "
            "is inadmissible: 100\\% of FastMap-guided paths are suboptimal "
            "(see text). Among admissible methods, AAC and ALT guarantee "
            "optimal paths. $^{\\dagger}$: Wilcoxon two-sided $p < 10^{-5}$ "
            "(AAC vs ALT, admissible methods only)."
        ),
        label="tab:main_results",
    )

    with open(output_tex, "w") as f:
        f.write(latex_str)


def generate_timing_table(
    df_timing: pd.DataFrame,
    output_tex: str,
) -> None:
    """Generate LaTeX timing comparison table for AAC vs ALT.

    Columns: Graph, Method, optional offline preprocessing total, p50 (ms),
    p95 (ms). When preprocessing totals are present, bolds the minimum
    preprocessing total and minimum p50 per graph.

    Args:
        df_timing: Timing DataFrame with graph, method, p50_ms, p95_ms columns,
            optionally preprocess_total_sec.
        output_tex: Output .tex file path.
    """
    method_rename = {"aac": "AAC", "alt": "ALT"}
    display_cols = ["graph", "method"]
    if "preprocess_total_sec" in df_timing.columns:
        display_cols.append("preprocess_total_sec")
    display_cols.extend(["p50_ms", "p95_ms"])

    table_df = df_timing[display_cols].copy()
    table_df["method"] = table_df["method"].map(method_rename).fillna(table_df["method"])

    if "preprocess_total_sec" in table_df.columns:
        table_df = _bold_best_in_group(
            table_df,
            group_cols=["graph"],
            metric_col="preprocess_total_sec",
            maximize=False,
            fmt="%.1f",
        )

    # Bold minimum p50 per graph
    table_df = _bold_best_in_group(
        table_df,
        group_cols=["graph"],
        metric_col="p50_ms",
        maximize=False,
        fmt="%.1f",
    )

    # Format p95
    table_df["p95_ms"] = table_df["p95_ms"].apply(
        lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else str(v)
    )

    table_df = table_df.rename(
        columns={
            "graph": "Graph",
            "method": "Method",
            "preprocess_total_sec": "Offline (s)",
            "p50_ms": "p50 (ms)",
            "p95_ms": "p95 (ms)",
        }
    )

    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)

    latex_str = table_df.to_latex(
        index=False,
        escape=False,
        caption=(
            "Offline preprocessing total and query latency comparison (p50/p95) on DIMACS road networks. Offline time includes anchor selection and SSSP for both methods, plus compressor training for AAC when present in the source CSV."
        ),
        label="tab:timing",
    )

    with open(output_tex, "w") as f:
        f.write(latex_str)


def generate_pareto_detail_table(
    df_ny: pd.DataFrame,
    df_fla: pd.DataFrame,
    output_tex: str,
) -> None:
    """Generate full Pareto sweep detail table (supplementary material).

    Shows all AAC configs (K0, m) and ALT configs (K) for both NY and FLA.
    Bolds the best expansion reduction per graph.

    Args:
        df_ny: Pareto sweep data for NY.
        df_fla: Pareto sweep data for FLA.
        output_tex: Output .tex file path.
    """
    rows: list[dict] = []

    for graph_name, df in [("NY", df_ny), ("FLA", df_fla)]:
        for _, row in df.iterrows():
            method = row["method"]
            if method == "Dijkstra":
                continue  # Skip baseline

            if method == "AAC":
                config_str = f"K0={int(row['K0'])}, m={int(row['m'])}"
                display_method = "AAC"
            elif method == "ALT":
                config_str = f"K={int(row['num_landmarks_or_dims'])}"
                display_method = "ALT"
            elif method == "FastMap":
                config_str = f"d={int(row['num_landmarks_or_dims'])}"
                display_method = "FastMap"
            else:
                continue

            rows.append(
                {
                    "Graph": graph_name,
                    "Method": display_method,
                    "Config": config_str,
                    "Bytes/V": int(row["bytes_per_vertex"]),
                    "Reduction (\\%)": row["expansion_reduction_pct"],
                }
            )

    table_df = pd.DataFrame(rows)

    # Bold best reduction per graph
    table_df = _bold_best_in_group(
        table_df,
        group_cols=["Graph"],
        metric_col="Reduction (\\%)",
        maximize=True,
    )

    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)

    latex_str = table_df.to_latex(
        index=False,
        escape=False,
        caption=(
            "Full Pareto sweep detail (3-seed means). Best expansion reduction "
            "per graph in bold (across all methods and budgets). FastMap is "
            "inadmissible: 100\\% of FastMap-guided paths are suboptimal on "
            "these directed graphs. On FLA, ALT at 256 bytes/vertex achieves "
            "higher reduction than all FastMap configurations."
        ),
        label="tab:pareto_detail",
    )

    with open(output_tex, "w") as f:
        f.write(latex_str)


def _bold_best_values(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Bold the minimum value in each metric column per graph group.

    Converts metric columns to formatted strings first, then wraps the
    best (minimum) value with ``\\textbf{}``.
    """
    result = df.copy()

    group_col = "graph" if "graph" in result.columns else None

    for metric in metrics:
        if metric not in result.columns:
            continue

        # Remember original numeric values for comparison
        numeric_vals = pd.to_numeric(result[metric], errors="coerce")

        # Convert column to formatted strings
        result[metric] = result[metric].apply(
            lambda v: f"{v:.1f}" if isinstance(v, float) else str(v)
        )

        if group_col is not None:
            for _graph_name, group_idx in result.groupby(group_col).groups.items():
                group_numeric = numeric_vals.loc[group_idx]
                if group_numeric.isna().all():
                    continue
                best_idx = group_numeric.idxmin()
                result.loc[best_idx, metric] = (
                    f"\\textbf{{{result.loc[best_idx, metric]}}}"
                )
        else:
            if numeric_vals.isna().all():
                continue
            best_idx = numeric_vals.idxmin()
            result.loc[best_idx, metric] = (
                f"\\textbf{{{result.loc[best_idx, metric]}}}"
            )

    return result
