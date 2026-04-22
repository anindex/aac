"""Visualization primitives for AAC figures."""

from aac.viz.compression import (
    landmark_overlay,
    selection_evolution_gif,
    weight_matrix_heatmap,
)
from aac.viz.embeddings import (
    compute_pca_projection,
    compute_tsne_projection,
    plot_embedding_projection,
)
from aac.viz.results import (
    plot_ablation_bar,
    plot_latency_bars,
    plot_pareto_frontier,
    plot_results_dashboard,
)
from aac.viz.search import (
    plot_comparison_panel,
    plot_expansion_heatmap,
    plot_expansion_heatmap_grid,
    plot_heuristic_contour,
    plot_heuristic_contour_grid,
)
from aac.viz.style import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    PALETTE,
    TMLR_COLUMN_WIDTH,
    TMLR_FULL_WIDTH,
    setup_style,
)

__all__ = [
    "PALETTE",
    "METHOD_COLORS",
    "METHOD_LABELS",
    "METHOD_MARKERS",
    "TMLR_COLUMN_WIDTH",
    "TMLR_FULL_WIDTH",
    "setup_style",
    "compute_pca_projection",
    "compute_tsne_projection",
    "landmark_overlay",
    "plot_ablation_bar",
    "plot_comparison_panel",
    "plot_embedding_projection",
    "plot_expansion_heatmap",
    "plot_expansion_heatmap_grid",
    "plot_heuristic_contour",
    "plot_heuristic_contour_grid",
    "plot_latency_bars",
    "plot_pareto_frontier",
    "plot_results_dashboard",
    "selection_evolution_gif",
    "weight_matrix_heatmap",
]
