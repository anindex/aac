"""Tests for Hydra configuration loading and dependency specification."""

from __future__ import annotations

from pathlib import Path

import pytest


def _get_config_dir() -> str:
    """Return the absolute path to the canonical Hydra config directory.

    The Hydra configs live under `src/experiments/configs/`.
    """
    return str(
        Path(__file__).resolve().parent.parent / "src" / "experiments" / "configs"
    )


class TestHydraConfig:
    """Tests for Hydra configuration composition."""

    def test_hydra_config_loads(self) -> None:
        """Root config loads with default track=dimacs and method=aac."""
        from hydra import compose, initialize_config_dir

        config_dir = _get_config_dir()
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="config")
            assert cfg.seed == 42
            assert cfg.track.name == "dimacs"
            assert cfg.method.name == "aac"
            assert cfg.num_queries == 1000

    def test_config_timing_section(self) -> None:
        """Timing configuration has expected defaults."""
        from hydra import compose, initialize_config_dir

        config_dir = _get_config_dir()
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="config")
            assert cfg.timing.warmup_runs == 10
            assert cfg.timing.num_runs == 100

    def test_config_batch_sizes(self) -> None:
        """Batch throughput sizes are [1, 8, 32, 128, 1024]."""
        from hydra import compose, initialize_config_dir

        config_dir = _get_config_dir()
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="config")
            assert list(cfg.batch_throughput_sizes) == [1, 8, 32, 128, 1024]


class TestMethodConfigs:
    """Tests for method-specific config groups."""

    @pytest.mark.parametrize(
        "method_name",
        ["aac", "alt", "fastmap", "dijkstra"],
    )
    def test_method_config_loads(self, method_name: str) -> None:
        """Each method config loads and has a 'name' field."""
        from hydra import compose, initialize_config_dir

        config_dir = _get_config_dir()
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(
                config_name="config",
                overrides=[f"method={method_name}"],
            )
            assert cfg.method.name == method_name


class TestTrackConfigs:
    """Tests for track-specific config groups."""

    @pytest.mark.parametrize("track_name", ["dimacs", "osmnx"])
    def test_track_config_loads(self, track_name: str) -> None:
        """Each track config loads and has a 'name' field."""
        from hydra import compose, initialize_config_dir

        config_dir = _get_config_dir()
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(
                config_name="config",
                overrides=[f"track={track_name}"],
            )
            assert cfg.track.name == track_name


class TestDependencies:
    """Tests for pyproject.toml dependency specification."""

    def test_deps_specified(self) -> None:
        """All required experiment dependencies appear in pyproject.toml."""
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        for dep in ["hydra-core", "tensorboard", "matplotlib", "seaborn", "pandas"]:
            assert dep in content, f"Missing dependency: {dep}"
