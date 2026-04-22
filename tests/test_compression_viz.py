"""Tests for compression process visualization functions.

Tests use synthetic data (no external file dependencies).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from aac.compression.compressor import LinearCompressor
from aac.viz.compression import (
    landmark_overlay,
    selection_evolution_gif,
    weight_matrix_heatmap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_W() -> torch.Tensor:
    """Create a sample m x K weight logit matrix."""
    torch.manual_seed(42)
    # 4 compressed dims, 16 teacher landmarks
    W = torch.randn(4, 16, dtype=torch.float64)
    # Make it somewhat sparse: push most entries negative
    W -= 3.0
    # Set one strong entry per row
    for i in range(4):
        W[i, i * 4] = 5.0
    return W


@pytest.fixture
def sample_coords() -> np.ndarray:
    """Create sample 2D coordinates for a 10x10 grid."""
    return np.array(
        [(c, r) for r in range(10) for c in range(10)], dtype=np.float64
    )


@pytest.fixture
def sample_fps_ids() -> np.ndarray:
    """Create sample FPS landmark indices."""
    return np.array([0, 9, 90, 99, 44, 55, 33, 66], dtype=np.int64)


@pytest.fixture
def sample_snapshots(sample_W: torch.Tensor) -> list[tuple[float, torch.Tensor]]:
    """Create sample weight snapshots for animation."""
    snapshots = []
    for i in range(5):
        tau = 2.0 - i * 0.4  # tau from 2.0 to 0.4
        # Progressively sharpen the weights
        W_snap = sample_W.clone() * (1 + i * 0.5)
        snapshots.append((tau, W_snap))
    return snapshots


# ---------------------------------------------------------------------------
# Tests: weight_matrix_heatmap
# ---------------------------------------------------------------------------


class TestWeightMatrixHeatmap:
    """Tests for weight_matrix_heatmap function."""

    def test_creates_figure(self, sample_W: torch.Tensor) -> None:
        """Function returns a matplotlib Figure."""
        fig = weight_matrix_heatmap(sample_W)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_accepts_numpy(self, sample_W: torch.Tensor) -> None:
        """Function accepts numpy array input."""
        W_np = sample_W.numpy()
        fig = weight_matrix_heatmap(W_np)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_accepts_axes(self, sample_W: torch.Tensor) -> None:
        """Function draws on provided axes."""
        fig, ax = plt.subplots()
        result_fig = weight_matrix_heatmap(sample_W, ax=ax)
        assert result_fig is fig
        plt.close(fig)

    def test_custom_title(self, sample_W: torch.Tensor) -> None:
        """Function respects custom title."""
        fig = weight_matrix_heatmap(sample_W, title="Custom Title")
        ax = fig.axes[0]
        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_saves_to_pdf(self, sample_W: torch.Tensor) -> None:
        """Figure can be saved as PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = weight_matrix_heatmap(sample_W)
            path = os.path.join(tmpdir, "test.pdf")
            fig.savefig(path, format="pdf")
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            plt.close(fig)

    def test_small_matrix(self) -> None:
        """Function handles very small matrix (2x4)."""
        W = torch.randn(2, 4, dtype=torch.float64)
        fig = weight_matrix_heatmap(W)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_large_matrix(self) -> None:
        """Function handles larger matrix (16x64)."""
        W = torch.randn(16, 64, dtype=torch.float64)
        fig = weight_matrix_heatmap(W)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: landmark_overlay
# ---------------------------------------------------------------------------


class TestLandmarkOverlay:
    """Tests for landmark_overlay function."""

    def test_creates_figure(
        self,
        sample_coords: np.ndarray,
        sample_fps_ids: np.ndarray,
    ) -> None:
        """Function returns a matplotlib Figure with two panels."""
        fwd_ids = sample_fps_ids[:4]
        bwd_ids = sample_fps_ids[4:]
        fig = landmark_overlay(
            coords=sample_coords,
            fps_ids=sample_fps_ids,
            learned_fwd_ids=fwd_ids,
            learned_bwd_ids=bwd_ids,
        )
        assert isinstance(fig, plt.Figure)
        # Should have at least 2 axes (left + right panels)
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_accepts_custom_axes(
        self,
        sample_coords: np.ndarray,
        sample_fps_ids: np.ndarray,
    ) -> None:
        """Function draws on provided axes."""
        fig, (ax_l, ax_r) = plt.subplots(1, 2)
        fwd_ids = sample_fps_ids[:4]
        bwd_ids = sample_fps_ids[4:]
        result_fig = landmark_overlay(
            coords=sample_coords,
            fps_ids=sample_fps_ids,
            learned_fwd_ids=fwd_ids,
            learned_bwd_ids=bwd_ids,
            ax_left=ax_l,
            ax_right=ax_r,
        )
        assert result_fig is fig
        plt.close(fig)

    def test_saves_to_pdf(
        self,
        sample_coords: np.ndarray,
        sample_fps_ids: np.ndarray,
    ) -> None:
        """Figure can be saved as PDF."""
        fwd_ids = sample_fps_ids[:4]
        bwd_ids = sample_fps_ids[4:]
        fig = landmark_overlay(
            coords=sample_coords,
            fps_ids=sample_fps_ids,
            learned_fwd_ids=fwd_ids,
            learned_bwd_ids=bwd_ids,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "overlay.pdf")
            fig.savefig(path, format="pdf")
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        plt.close(fig)

    def test_infers_K0_and_m(
        self,
        sample_coords: np.ndarray,
        sample_fps_ids: np.ndarray,
    ) -> None:
        """Function infers K0 and m from input arrays."""
        fwd_ids = sample_fps_ids[:3]
        bwd_ids = sample_fps_ids[3:6]
        fig = landmark_overlay(
            coords=sample_coords,
            fps_ids=sample_fps_ids,
            learned_fwd_ids=fwd_ids,
            learned_bwd_ids=bwd_ids,
        )
        # Title should contain K0=8 and m=6
        ax_left = fig.axes[0]
        ax_right = fig.axes[1]
        assert "8" in ax_left.get_title()
        assert "6" in ax_right.get_title()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: selection_evolution_gif
# ---------------------------------------------------------------------------


class TestSelectionEvolutionGif:
    """Tests for selection_evolution_gif function."""

    def test_creates_gif(
        self,
        sample_snapshots: list[tuple[float, torch.Tensor]],
    ) -> None:
        """Function creates a GIF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = selection_evolution_gif(
                snapshots=sample_snapshots,
                output_path=os.path.join(tmpdir, "test.gif"),
            )
            assert os.path.exists(gif_path)
            assert os.path.getsize(gif_path) > 0
            assert str(gif_path).endswith(".gif")

    def test_single_frame(self, sample_W: torch.Tensor) -> None:
        """Function handles single-frame animation."""
        snapshots = [(1.0, sample_W)]
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = selection_evolution_gif(
                snapshots=snapshots,
                output_path=os.path.join(tmpdir, "single.gif"),
            )
            assert os.path.exists(gif_path)

    def test_empty_raises(self) -> None:
        """Function raises on empty snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="at least one frame"):
                selection_evolution_gif(
                    snapshots=[],
                    output_path=os.path.join(tmpdir, "empty.gif"),
                )

    def test_numpy_input(self) -> None:
        """Function accepts numpy arrays in snapshots."""
        W_np = np.random.randn(4, 8).astype(np.float32)
        snapshots = [(1.0, W_np), (0.5, W_np * 2)]
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = selection_evolution_gif(
                snapshots=snapshots,
                output_path=os.path.join(tmpdir, "numpy.gif"),
            )
            assert os.path.exists(gif_path)

    def test_creates_parent_dirs(
        self,
        sample_snapshots: list[tuple[float, torch.Tensor]],
    ) -> None:
        """Function creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = os.path.join(tmpdir, "a", "b", "c", "test.gif")
            gif_path = selection_evolution_gif(
                snapshots=sample_snapshots,
                output_path=deep_path,
            )
            assert os.path.exists(gif_path)

    def test_gif_size_reasonable(
        self,
        sample_snapshots: list[tuple[float, torch.Tensor]],
    ) -> None:
        """GIF file size is reasonable (< 5MB for small data)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = selection_evolution_gif(
                snapshots=sample_snapshots,
                output_path=os.path.join(tmpdir, "size.gif"),
            )
            size_mb = os.path.getsize(gif_path) / (1024 * 1024)
            assert size_mb < 5.0, f"GIF too large: {size_mb:.1f}MB"


# ---------------------------------------------------------------------------
# Integration: end-to-end with LinearCompressor
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests with actual LinearCompressor."""

    def test_weight_heatmap_from_compressor(self) -> None:
        """weight_matrix_heatmap works with LinearCompressor.W_fwd."""
        torch.manual_seed(42)
        comp = LinearCompressor(K=16, m=8, is_directed=True)
        fig = weight_matrix_heatmap(comp.W_fwd)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_full_pipeline_synthetic(self) -> None:
        """End-to-end: train small compressor, generate all visualizations."""
        torch.manual_seed(42)
        K, m = 8, 4
        comp = LinearCompressor(K=K, m=m, is_directed=True)

        # Quick training step (just to change weights)
        optimizer = torch.optim.Adam(comp.parameters(), lr=0.1)
        d_out = torch.randn(10, K, dtype=torch.float64)
        d_in = torch.randn(10, K, dtype=torch.float64)
        comp.train()
        y_fwd, y_bwd = comp(d_out, d_in, tau=1.0)
        loss = y_fwd.sum() + y_bwd.sum()
        loss.backward()
        optimizer.step()
        comp.eval()

        # Weight heatmap
        fig = weight_matrix_heatmap(comp.W_fwd)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Landmark overlay with dummy coords
        coords = np.random.randn(20, 2)
        fps_ids = np.arange(K)
        sel = comp.selected_landmarks()
        fwd_ids = np.array(sel["fwd"])
        bwd_ids = np.array(sel["bwd"])
        fig = landmark_overlay(
            coords=coords,
            fps_ids=fps_ids,
            learned_fwd_ids=fwd_ids,
            learned_bwd_ids=bwd_ids,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Evolution GIF
        snapshots = [(1.0, comp.W_fwd.data.clone())]
        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = selection_evolution_gif(
                snapshots=snapshots,
                output_path=os.path.join(tmpdir, "test.gif"),
            )
            assert os.path.exists(gif_path)
