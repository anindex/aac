"""Tests for embedding projection visualization functions (aac.viz.embeddings)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# TestTsneProjection
# ---------------------------------------------------------------------------


class TestTsneProjection:
    def test_tsne_output_shape(self):
        """compute_tsne_projection with (50, 8) input returns (50, 2) array."""
        from aac.viz.embeddings import compute_tsne_projection

        rng = np.random.default_rng(42)
        labels = rng.standard_normal((50, 8))
        result = compute_tsne_projection(labels)
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 2)

    def test_tsne_perplexity_clamping(self):
        """compute_tsne_projection clamps perplexity so (20, 8) with perplexity=30 works."""
        from aac.viz.embeddings import compute_tsne_projection

        rng = np.random.default_rng(42)
        labels = rng.standard_normal((20, 8))
        # perplexity=30 > 20//4=5, should be clamped to 5 and not error
        result = compute_tsne_projection(labels, perplexity=30.0)
        assert result.shape == (20, 2)


# ---------------------------------------------------------------------------
# TestPcaProjection
# ---------------------------------------------------------------------------


class TestPcaProjection:
    def test_pca_output_shape(self):
        """compute_pca_projection with (50, 8) input returns (50, 2) array."""
        from aac.viz.embeddings import compute_pca_projection

        rng = np.random.default_rng(42)
        labels = rng.standard_normal((50, 8))
        result = compute_pca_projection(labels)
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 2)


# ---------------------------------------------------------------------------
# TestPlotEmbeddingProjection
# ---------------------------------------------------------------------------


class TestPlotEmbeddingProjection:
    def test_returns_scalar_mappable(self):
        """plot_embedding_projection returns a ScalarMappable (PathCollection)."""
        from matplotlib.cm import ScalarMappable

        from aac.viz.embeddings import plot_embedding_projection

        rng = np.random.default_rng(42)
        coords = rng.standard_normal((30, 2))
        colors = rng.standard_normal(30)

        fig, ax = plt.subplots()
        sc = plot_embedding_projection(ax, coords, colors)
        assert isinstance(sc, ScalarMappable)
        plt.close(fig)

    def test_scatter_point_count(self):
        """plot_embedding_projection creates scatter with correct number of points (V)."""
        from aac.viz.embeddings import plot_embedding_projection

        V = 45
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((V, 2))
        colors = rng.standard_normal(V)

        fig, ax = plt.subplots()
        plot_embedding_projection(ax, coords, colors)
        # The scatter creates a PathCollection; check offsets count
        collections = ax.collections
        assert len(collections) == 1
        offsets = collections[0].get_offsets()
        assert len(offsets) == V
        plt.close(fig)

    def test_title_set_when_provided(self):
        """plot_embedding_projection sets axes title when title is provided."""
        from aac.viz.embeddings import plot_embedding_projection

        rng = np.random.default_rng(42)
        coords = rng.standard_normal((20, 2))
        colors = rng.standard_normal(20)

        fig, ax = plt.subplots()
        plot_embedding_projection(ax, coords, colors, title="My Embedding")
        assert ax.get_title() == "My Embedding"
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test guarded import
# ---------------------------------------------------------------------------


def test_guarded_import_tsne():
    """Guarded import raises ImportError with 'pip install' in the message."""
    import builtins
    from unittest.mock import patch

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    # We need to import the module fresh so the guarded import triggers.
    # Patch builtins.__import__ to block sklearn.
    from aac.viz.embeddings import compute_tsne_projection

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="pip install"):
            compute_tsne_projection(np.zeros((10, 4)))
