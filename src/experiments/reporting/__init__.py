"""Reporting pipeline: CSV aggregation, LaTeX tables, publication-quality figures."""

from experiments.reporting.csv_writer import (
    aggregate_results,
    get_git_hash,
    merge_results,
    write_csv_metadata,
)
from experiments.reporting.figures import (
    plot_compression_ratio,
    plot_latency_comparison,
    plot_pareto_frontier,
    setup_plot_style,
)
from experiments.reporting.latex_tables import (
    generate_comparison_table,
    generate_preprocessing_table,
)

# Re-export consolidated style module for convenience
from aac.viz.style import setup_style  # noqa: F401

__all__ = [
    "aggregate_results",
    "get_git_hash",
    "merge_results",
    "write_csv_metadata",
    "generate_comparison_table",
    "generate_preprocessing_table",
    "plot_pareto_frontier",
    "plot_compression_ratio",
    "plot_latency_comparison",
    "setup_plot_style",
    "setup_style",
]
