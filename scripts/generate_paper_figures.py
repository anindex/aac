#!/usr/bin/env python3
"""Generate all data-driven paper figures and tables from experimental CSVs.

Deterministic script: uses Agg backend, fixed random seeds, no interactive display.

Usage:
    python scripts/generate_paper_figures.py --output-dir results/paper/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable so `experiments` resolves to src/experiments/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

np.random.seed(42)

from experiments.reporting.figures import (
    plot_latency_comparison,
    plot_pareto_two_panel,
)
from experiments.reporting.latex_tables import (
    generate_dimacs_main_table,
    generate_pareto_detail_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all data-driven paper figures and tables."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/paper/",
        help="Directory for output files (default: results/paper/)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/dimacs/",
        help="Directory containing experimental CSVs (default: results/dimacs/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load CSVs ---
    print("Loading experimental data...")
    df_ny = pd.read_csv(data_dir / "pareto_sweep_NY.csv", comment="#")
    df_fla = pd.read_csv(data_dir / "pareto_sweep_FLA.csv", comment="#")
    df_timing = pd.read_csv(data_dir / "timing_p50_p95.csv")
    df_wilcoxon = pd.read_csv(data_dir / "wilcoxon_pvalues.csv")
    print(f"  pareto_sweep_NY: {len(df_ny)} rows")
    print(f"  pareto_sweep_FLA: {len(df_fla)} rows")
    print(f"  timing_p50_p95: {len(df_timing)} rows")
    print(f"  wilcoxon_pvalues: {len(df_wilcoxon)} rows")

    generated = []

    # --- Figure 1: Pareto frontier two-panel plot ---
    pareto_pdf = str(output_dir / "pareto_frontier.pdf")
    print("\nGenerating Pareto frontier plot...")
    plot_pareto_two_panel(df_ny, df_fla, pareto_pdf)
    generated.append(pareto_pdf)
    print(f"  -> {pareto_pdf}")

    # --- Figure 2: Latency comparison bar chart ---
    latency_pdf = str(output_dir / "latency_comparison.pdf")
    print("Generating latency comparison plot...")
    plot_latency_comparison(df_timing, latency_pdf)
    generated.append(latency_pdf)
    print(f"  -> {latency_pdf}")

    # --- Table 1: Main DIMACS comparison ---
    main_tex = str(output_dir / "table_main_dimacs.tex")
    print("Generating main DIMACS comparison table...")
    generate_dimacs_main_table(df_ny, df_fla, df_wilcoxon, main_tex)
    generated.append(main_tex)
    print(f"  -> {main_tex}")

    # The standalone timing table was folded into the inline `tab:latency`;
    # the generator below is retained for users who want the standalone form
    # but is no longer invoked from the reproducer pipeline.
    # generate_timing_table(df_timing, str(output_dir / "table_timing.tex"))

    # --- Table 3: Pareto detail (supplementary) ---
    pareto_tex = str(output_dir / "table_pareto_detail.tex")
    print("Generating Pareto detail table...")
    generate_pareto_detail_table(df_ny, df_fla, pareto_tex)
    generated.append(pareto_tex)
    print(f"  -> {pareto_tex}")

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print(f"Generated {len(generated)} files:")
    for f in generated:
        size = Path(f).stat().st_size
        print(f"  {f} ({size:,} bytes)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
