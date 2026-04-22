"""Integration tests for Track 3 (Warcraft, Cabspotting) runners.

Tests runner instantiation, method support, path metric computation,
and Hydra config loading.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf


def _get_config_dir() -> str:
    """Return the absolute path to the canonical Hydra config directory.

    The Hydra configs live under `src/experiments/configs/`.
    """
    return str(
        Path(__file__).resolve().parent.parent / "src" / "experiments" / "configs"
    )


def _make_warcraft_cfg(
    method_name: str = "contextual",
    log_dir: str = "/tmp/test_tb_logs",
) -> DictConfig:
    """Create a minimal DictConfig for WarcraftRunner construction."""
    return OmegaConf.create(
        {
            "track": {
                "name": "warcraft",
                "data_dir": "data/warcraft_real/warcraft_shortest_path_oneskin",
                "grid_size": 12,
                "img_size": 96,
            },
            "method": {
                "name": method_name,
                "K": 5,
                "m": 8,
                "training": {
                    "num_epochs": 10,
                    "batch_size": 4,
                    "lr": 0.001,
                },
            },
            "log_dir": log_dir,
            "seed": 42,
            "output_dir": "/tmp/test_results",
            "num_queries": 10,
        }
    )


# --- TestWarcraftRunner ---


class TestWarcraftRunner:
    """Tests for WarcraftRunner instantiation, methods, and metrics."""

    def test_runner_instantiation(self) -> None:
        """Create WarcraftRunner with mock DictConfig, verify no errors."""
        from experiments.runners.warcraft_runner import WarcraftRunner

        cfg = _make_warcraft_cfg(method_name="contextual")
        runner = WarcraftRunner(cfg)
        assert runner is not None
        runner.close()

    def test_supported_methods(self) -> None:
        """Verify SUPPORTED_METHODS = ['contextual', 'datasp', 'dijkstra'].

        IMPORTANT: assert 'datasp' is in SUPPORTED_METHODS to validate
        the DataSP comparison is wired as specified.
        """
        from experiments.runners.warcraft_runner import WarcraftRunner

        assert WarcraftRunner.SUPPORTED_METHODS == ["contextual", "datasp", "dijkstra"]
        assert "datasp" in WarcraftRunner.SUPPORTED_METHODS
        assert "contextual" in WarcraftRunner.SUPPORTED_METHODS

    def test_unsupported_method_raises(self) -> None:
        """Creating runner with unsupported method raises ValueError."""
        from experiments.runners.warcraft_runner import WarcraftRunner

        with pytest.raises(ValueError, match="not supported"):
            cfg = _make_warcraft_cfg(method_name="alt")
            WarcraftRunner(cfg)

    def test_path_metrics_identical_paths(self) -> None:
        """Identical paths -> match=True, jaccard=1.0, cost_regret=0.0."""
        from experiments.runners.warcraft_runner import _compute_path_metrics

        path = [0, 1, 2, 3]
        result = _compute_path_metrics(path, path, 10.0, 10.0)
        assert result["match"] is True
        assert result["jaccard"] == 1.0
        assert result["cost_regret"] == 0.0

    def test_path_metrics_different_paths(self) -> None:
        """Completely different paths -> match=False, jaccard=0.0."""
        from experiments.runners.warcraft_runner import _compute_path_metrics

        pred = [0, 1, 2, 3]
        gt = [4, 5, 6, 7]
        result = _compute_path_metrics(pred, gt, 15.0, 10.0)
        assert result["match"] is False
        assert result["jaccard"] == 0.0
        assert result["cost_regret"] == pytest.approx(0.5)

    def test_path_metrics_partial_overlap(self) -> None:
        """Partial overlap -> 0 < jaccard < 1."""
        from experiments.runners.warcraft_runner import _compute_path_metrics

        pred = [0, 1, 2, 3]  # edges: (0,1), (1,2), (2,3)
        gt = [0, 1, 2, 4]  # edges: (0,1), (1,2), (2,4)
        result = _compute_path_metrics(pred, gt, 12.0, 10.0)
        assert result["match"] is False
        assert 0.0 < result["jaccard"] < 1.0
        # intersection: {(0,1), (1,2)} = 2, union: {(0,1),(1,2),(2,3),(2,4)} = 4
        assert result["jaccard"] == pytest.approx(2.0 / 4.0)
        assert result["cost_regret"] == pytest.approx(0.2)

    def test_path_metrics_empty_paths(self) -> None:
        """Empty paths -> jaccard=1.0 (both empty)."""
        from experiments.runners.warcraft_runner import _compute_path_metrics

        result = _compute_path_metrics([], [], 0.0, 0.0)
        assert result["jaccard"] == 1.0

    def test_dispatch_returns_warcraft_runner(self) -> None:
        """get_runner('warcraft') returns WarcraftRunner class."""
        from experiments.runners import get_runner
        from experiments.runners.warcraft_runner import WarcraftRunner

        assert get_runner("warcraft") is WarcraftRunner


# --- TestContextualConfigs ---


class TestContextualConfigs:
    """Tests for Hydra config loading of Track 3 and contextual method."""

    def test_warcraft_config_loads(self) -> None:
        """Load warcraft.yaml and verify required fields."""
        config_path = Path(_get_config_dir()) / "track" / "warcraft.yaml"
        cfg = OmegaConf.load(str(config_path))
        assert cfg.name == "warcraft"
        assert cfg.grid_size == 12
        assert cfg.img_size == 96
        # Real Pogancic dataset uses pre-split npy files (no manual split ratios)

    def test_contextual_method_loads(self) -> None:
        """Load contextual.yaml and verify training fields."""
        config_path = Path(_get_config_dir()) / "method" / "contextual.yaml"
        cfg = OmegaConf.load(str(config_path))
        assert cfg.name == "contextual"
        assert cfg.K == 5
        assert cfg.m == 8
        assert cfg.training.num_epochs == 100
        assert cfg.training.beta_init == 1.0
        assert cfg.training.beta_max == 30.0

    def test_warcraft_hydra_composition(self) -> None:
        """Warcraft track config composes with Hydra."""
        from hydra import compose, initialize_config_dir

        config_dir = _get_config_dir()
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(
                config_name="config",
                overrides=["track=warcraft", "method=contextual"],
            )
            assert cfg.track.name == "warcraft"
            assert cfg.method.name == "contextual"
            assert cfg.track.grid_size == 12


# --- TestContextualMetrics ---


class TestContextualMetrics:
    """Tests for METR-07 path accuracy metric computations."""

    def test_path_match_rate(self) -> None:
        """Compute path match rate over a batch of results."""
        from experiments.runners.warcraft_runner import _compute_path_metrics

        # 3 matches out of 5
        paths_pred = [
            [0, 1, 2],
            [0, 1, 2],
            [0, 3, 2],  # different
            [0, 1, 2],
            [0, 4, 2],  # different
        ]
        paths_gt = [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ]

        matches = []
        for pred, gt in zip(paths_pred, paths_gt):
            m = _compute_path_metrics(pred, gt, 10.0, 10.0)
            matches.append(m["match"])

        match_rate = sum(matches) / len(matches)
        assert 0.0 <= match_rate <= 1.0
        assert match_rate == pytest.approx(3.0 / 5.0)

    def test_jaccard_computation(self) -> None:
        """Test edge set overlap on small paths."""
        from experiments.runners.warcraft_runner import _compute_path_metrics

        # pred edges: {(0,1), (1,2)} -- gt edges: {(0,1), (1,3)}
        result = _compute_path_metrics([0, 1, 2], [0, 1, 3], 10.0, 10.0)
        # intersection = {(0,1)} = 1, union = {(0,1),(1,2),(1,3)} = 3
        assert result["jaccard"] == pytest.approx(1.0 / 3.0)

    def test_cost_regret_computation(self) -> None:
        """Verify (predicted - optimal) / optimal formula."""
        from experiments.runners.warcraft_runner import _compute_path_metrics

        result = _compute_path_metrics([0, 1], [0, 1], 15.0, 10.0)
        assert result["cost_regret"] == pytest.approx(0.5)

        result = _compute_path_metrics([0, 1], [0, 1], 10.0, 10.0)
        assert result["cost_regret"] == pytest.approx(0.0)

        result = _compute_path_metrics([0, 1], [0, 1], 20.0, 10.0)
        assert result["cost_regret"] == pytest.approx(1.0)


# --- TestCabspottingRunner ---


def _make_cabspotting_cfg(
    method_name: str = "contextual",
    log_dir: str = "/tmp/test_tb_logs",
) -> DictConfig:
    """Create a minimal DictConfig for CabspottingRunner construction."""
    return OmegaConf.create(
        {
            "track": {
                "name": "cabspotting",
                "data_dir": "data/cabspotting",
                "num_nodes": 355,
                "num_edges": 2178,
                "input_dim": 6,
                "train_split": 0.7,
                "beta_override": 30.0,
            },
            "method": {
                "name": method_name,
                "K": 5,
                "m": 8,
                "num_epochs": 10,
                "batch_size": 4,
                "lr": 0.0001,
                "patience": 4,
            },
            "log_dir": log_dir,
            "seed": 42,
        }
    )


class TestCabspottingRunner:
    """Tests for CabspottingRunner instantiation, methods, and metrics."""

    def test_runner_instantiation(self) -> None:
        """Create CabspottingRunner with mock DictConfig, verify no errors."""
        from experiments.runners.cabspotting_runner import CabspottingRunner

        cfg = _make_cabspotting_cfg(method_name="contextual")
        runner = CabspottingRunner(cfg)
        assert runner is not None
        runner.close()

    def test_supported_methods(self) -> None:
        """Verify SUPPORTED_METHODS = ['contextual', 'datasp', 'dijkstra'].

        IMPORTANT: assert 'datasp' is in SUPPORTED_METHODS to validate
        the DataSP comparison is wired as specified.
        """
        from experiments.runners.cabspotting_runner import CabspottingRunner

        assert CabspottingRunner.SUPPORTED_METHODS == [
            "contextual",
            "datasp",
            "dijkstra",
        ]
        assert "datasp" in CabspottingRunner.SUPPORTED_METHODS
        assert "contextual" in CabspottingRunner.SUPPORTED_METHODS
        assert "dijkstra" in CabspottingRunner.SUPPORTED_METHODS

    def test_path_metrics_identical_paths(self) -> None:
        """Identical paths -> path_match=1.0, jaccard=1.0, cost_regret=0.0."""
        from experiments.runners.cabspotting_runner import CabspottingRunner

        path = [0, 1, 2, 3]
        result = CabspottingRunner._compute_path_metrics(path, path, 10.0, 10.0)
        assert result["path_match"] == 1.0
        assert result["jaccard"] == 1.0
        assert result["cost_regret"] == 0.0

    def test_path_metrics_different_paths(self) -> None:
        """Different paths -> path_match=0.0, jaccard=0.0."""
        from experiments.runners.cabspotting_runner import CabspottingRunner

        pred = [0, 1, 2, 3]
        gt = [4, 5, 6, 7]
        result = CabspottingRunner._compute_path_metrics(pred, gt, 15.0, 10.0)
        assert result["path_match"] == 0.0
        assert result["jaccard"] == 0.0
        assert result["cost_regret"] == pytest.approx(0.5)

    def test_path_metrics_partial_overlap(self) -> None:
        """Partial overlap -> 0 < jaccard < 1."""
        from experiments.runners.cabspotting_runner import CabspottingRunner

        pred = [0, 1, 2, 3]  # edges: (0,1), (1,2), (2,3)
        gt = [0, 1, 2, 4]  # edges: (0,1), (1,2), (2,4)
        result = CabspottingRunner._compute_path_metrics(pred, gt, 12.0, 10.0)
        assert result["path_match"] == 0.0
        assert 0.0 < result["jaccard"] < 1.0
        # intersection: {(0,1), (1,2)} = 2, union: {(0,1),(1,2),(2,3),(2,4)} = 4
        assert result["jaccard"] == pytest.approx(2.0 / 4.0)
        assert result["cost_regret"] == pytest.approx(0.2)

    def test_data_loading_graceful_failure(self) -> None:
        """Verify _load_cabspotting_data with nonexistent dir returns None."""
        from experiments.runners.cabspotting_runner import CabspottingRunner

        cfg = _make_cabspotting_cfg()
        runner = CabspottingRunner(cfg)
        result = runner._load_cabspotting_data("/nonexistent/data/dir", 0.7)
        assert result is None
        runner.close()

    def test_unsupported_method_raises(self) -> None:
        """Creating runner with unsupported method raises ValueError."""
        from experiments.runners.cabspotting_runner import CabspottingRunner

        with pytest.raises(ValueError, match="not supported"):
            cfg = _make_cabspotting_cfg(method_name="alt")
            CabspottingRunner(cfg)

    def test_dispatch_returns_cabspotting_runner(self) -> None:
        """get_runner('cabspotting') returns CabspottingRunner class."""
        from experiments.runners import get_runner
        from experiments.runners.cabspotting_runner import CabspottingRunner

        assert get_runner("cabspotting") is CabspottingRunner


# --- TestCabspottingConfig ---


class TestCabspottingConfig:
    """Tests for Cabspotting Hydra config loading and composition."""

    def test_config_loads(self) -> None:
        """Load cabspotting.yaml and verify required fields."""
        config_path = Path(_get_config_dir()) / "track" / "cabspotting.yaml"
        cfg = OmegaConf.load(str(config_path))
        assert cfg.name == "cabspotting"
        assert cfg.data_dir == "data/cabspotting"
        assert cfg.input_dim == 6
        assert cfg.num_nodes == 355
        assert cfg.train_split == 0.7

    def test_beta_override(self) -> None:
        """Verify cabspotting.yaml has beta_override=30.0 matching DataSP."""
        config_path = Path(_get_config_dir()) / "track" / "cabspotting.yaml"
        cfg = OmegaConf.load(str(config_path))
        assert cfg.beta_override == 30.0

    def test_config_composable(self) -> None:
        """Verify cabspotting config can be merged with contextual method config."""
        config_path = Path(_get_config_dir()) / "track" / "cabspotting.yaml"
        track_cfg = OmegaConf.load(str(config_path))

        method_path = Path(_get_config_dir()) / "method" / "contextual.yaml"
        method_cfg = OmegaConf.load(str(method_path))

        # Merge should succeed without conflicts (different key namespaces)
        merged = OmegaConf.merge(
            {"track": track_cfg, "method": method_cfg}
        )
        assert merged.track.name == "cabspotting"
        assert merged.method.name == "contextual"
        assert merged.track.input_dim == 6
        assert merged.method.K == 5

    def test_num_edges_field(self) -> None:
        """Verify cabspotting config has num_edges field matching DataSP."""
        config_path = Path(_get_config_dir()) / "track" / "cabspotting.yaml"
        cfg = OmegaConf.load(str(config_path))
        assert cfg.num_edges == 2178
