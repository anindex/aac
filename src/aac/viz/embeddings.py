"""Embedding projection visualization for AAC.

Projects compressed label vectors into 2D using t-SNE or PCA, and plots
scatter colored by graph distance to show that nearby nodes in the graph
cluster in embedding space.

All functions use the consolidated style from ``aac.viz.style``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

from aac.viz.style import TMLR_COLUMN_WIDTH, TMLR_FULL_WIDTH  # noqa: F401

__all__ = [
    "compute_tsne_projection",
    "compute_pca_projection",
    "plot_embedding_projection",
]


def compute_tsne_projection(
    labels: np.ndarray,
    *,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Project label vectors to 2D using t-SNE.

    Args:
        labels: (V, m) compressed label vectors.
        perplexity: t-SNE perplexity parameter. Clamped to
            ``min(perplexity, V // 4)`` to avoid sklearn errors on
            small datasets.
        random_state: Random seed for reproducibility.

    Returns:
        (V, 2) array of 2D coordinates.

    Raises:
        ImportError: If scikit-learn is not installed, with install
            instructions.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "scikit-learn is required for embedding projection. "
            "Install with: pip install 'aac[experiments]'"
        ) from None

    safe_perplexity = min(perplexity, len(labels) // 4)
    # Ensure perplexity is at least 1
    safe_perplexity = max(safe_perplexity, 1.0)

    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        random_state=random_state,
    )
    return tsne.fit_transform(labels)


def compute_pca_projection(
    labels: np.ndarray,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """Project label vectors to 2D using PCA.

    Args:
        labels: (V, m) compressed label vectors.
        random_state: Random seed for reproducibility.

    Returns:
        (V, 2) array of 2D coordinates.

    Raises:
        ImportError: If scikit-learn is not installed, with install
            instructions.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError(
            "scikit-learn is required for embedding projection. "
            "Install with: pip install 'aac[experiments]'"
        ) from None

    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(labels)


def plot_embedding_projection(
    ax: plt.Axes,
    coords_2d: np.ndarray,
    color_values: np.ndarray,
    *,
    cmap: str = "viridis",
    title: str = "",
    colorbar_label: str = "Graph distance",
    marker_size: float = 5.0,
    alpha: float = 0.8,
) -> ScalarMappable:
    """Plot 2D projection of embeddings colored by a scalar quantity.

    Follows the established Axes-accepting pattern from
    :mod:`aac.viz.search` and :mod:`aac.viz.results`.

    Args:
        ax: Target matplotlib Axes.
        coords_2d: (V, 2) projected coordinates from t-SNE or PCA.
        color_values: (V,) scalar values for coloring (e.g., graph
            distance from a reference node).
        cmap: Matplotlib colormap name.
        title: Panel title. Set only if non-empty.
        colorbar_label: Label for a colorbar (caller creates colorbar
            using the returned ScalarMappable).
        marker_size: Scatter point size.
        alpha: Scatter point opacity.

    Returns:
        The PathCollection (ScalarMappable) from ``ax.scatter``, which
        the caller can pass to ``fig.colorbar()``.
    """
    sc = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=color_values,
        cmap=cmap,
        s=marker_size,
        alpha=alpha,
        edgecolors="none",
        rasterized=True,
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    return sc
