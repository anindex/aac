"""Tests for the experiments.reporting pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# CSV writer tests
# ---------------------------------------------------------------------------


def test_aggregate_results_function():
    """aggregate_results is importable and callable."""
    from experiments.reporting.csv_writer import aggregate_results

    assert callable(aggregate_results)


def test_merge_results_function():
    """merge_results is importable and callable."""
    from experiments.reporting.csv_writer import merge_results

    assert callable(merge_results)


def test_merge_results_concatenates():
    """merge_results concatenates and sorts DataFrames."""
    from experiments.reporting.csv_writer import merge_results

    df1 = pd.DataFrame(
        {
            "graph": ["NY"],
            "method": ["alt"],
            "memory_bytes_per_vertex": [64],
        }
    )
    df2 = pd.DataFrame(
        {
            "graph": ["NY"],
            "method": ["aac"],
            "memory_bytes_per_vertex": [32],
        }
    )
    merged = merge_results([df1, df2])
    assert len(merged) == 2
    # Sorted by memory ascending within same graph
    assert merged.iloc[0]["memory_bytes_per_vertex"] <= merged.iloc[1]["memory_bytes_per_vertex"]


# ---------------------------------------------------------------------------
# LaTeX table tests
# ---------------------------------------------------------------------------


def test_generate_comparison_table_function():
    """generate_comparison_table is importable and callable."""
    from experiments.reporting.latex_tables import generate_comparison_table

    assert callable(generate_comparison_table)


def test_generate_comparison_table_output():
    """generate_comparison_table produces .tex file with bold best values."""
    from experiments.reporting.latex_tables import generate_comparison_table

    df = pd.DataFrame(
        {
            "graph": ["NY", "NY", "BAY", "BAY"],
            "method": ["aac", "alt", "aac", "alt"],
            "expansions_mean": [100.0, 200.0, 150.0, 250.0],
            "p50_ms": [1.0, 2.0, 1.5, 2.5],
            "memory_bytes_per_vertex": [64.0, 64.0, 64.0, 64.0],
            "preprocess_total_sec": [10.0, 5.0, 12.0, 6.0],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "test_table.tex")
        generate_comparison_table(df, out_path)

        assert Path(out_path).exists()
        content = Path(out_path).read_text()
        assert "\\textbf" in content
        assert "\\begin{tabular}" in content


# ---------------------------------------------------------------------------
# Figure tests
# ---------------------------------------------------------------------------


def test_setup_plot_style():
    """setup_plot_style runs without error."""
    from experiments.reporting.figures import setup_plot_style

    setup_plot_style()  # Should not raise


def test_plot_pareto_frontier():
    """plot_pareto_frontier generates a valid PDF."""
    from experiments.reporting.figures import plot_pareto_frontier

    df = pd.DataFrame(
        {
            "graph": ["NY"] * 4,
            "method": ["aac", "aac", "alt", "dijkstra"],
            "m": [8, 16, 16, 0],
            "K0": [64, 64, 0, 0],
            "expansions_mean": [120.0, 80.0, 150.0, 500.0],
            "memory_bytes_per_vertex": [32, 64, 64, 0],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "pareto.pdf")
        plot_pareto_frontier(df, out_path, graph_name="NY")

        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0


def test_plot_compression_ratio():
    """plot_compression_ratio generates a valid PDF for AAC data."""
    from experiments.reporting.figures import plot_compression_ratio

    # Create mock data with varying m and K0
    rows = []
    for k0 in [64, 128]:
        for m in [4, 8, 16, 32]:
            rows.append(
                {
                    "graph": "NY",
                    "method": "aac",
                    "m": m,
                    "K0": k0,
                    "expansions_mean": 500.0 / m,  # decreases with m
                    "memory_bytes_per_vertex": m * 4,
                }
            )
    # Add Dijkstra baseline
    rows.append(
        {
            "graph": "NY",
            "method": "dijkstra",
            "m": 0,
            "K0": 0,
            "expansions_mean": 1000.0,
            "memory_bytes_per_vertex": 0,
        }
    )
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "compression.pdf")
        plot_compression_ratio(df, out_path, graph_name="NY")

        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0


# ---------------------------------------------------------------------------
# Reproduction-pipeline driver tests
# ---------------------------------------------------------------------------
#
# The single canonical reproduction entry point is `scripts/reproduce_paper.py`.
# These tests check that the reproducibility surface remains intact.


def test_reproduce_paper_script_exists():
    """`scripts/reproduce_paper.py` exists and contains a main entry point."""
    repro_path = Path("scripts/reproduce_paper.py")
    assert repro_path.exists(), (
        "scripts/reproduce_paper.py is the canonical reproduction driver; "
        "it must remain importable from the repo root."
    )
    content = repro_path.read_text()
    assert "def main" in content
    assert 'if __name__ == "__main__":' in content


def test_reproduce_paper_argument_parser():
    """`reproduce_paper.py` supports the documented CLI flags."""
    content = Path("scripts/reproduce_paper.py").read_text()
    # Flags advertised by README.md, AGENTS.md, and the script's own --help.
    assert "--track" in content
    assert "--tables-only" in content
    assert "--no-verify" in content
