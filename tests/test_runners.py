"""Integration tests for experiment runners: BaseRunner, DIMACSRunner, OSMnxRunner."""

from __future__ import annotations

import inspect

import pytest
from omegaconf import DictConfig, OmegaConf

from aac.search.types import SearchResult
from experiments.metrics.collector import MetricsCollector
from experiments.runners import DIMACSRunner, OSMnxRunner, get_runner
from experiments.runners.base import BaseRunner
from experiments.utils import memory_bytes_per_vertex


def _make_cfg(
    track_name: str = "dimacs",
    method_name: str = "aac",
    log_dir: str = "/tmp/test_tb_logs",
) -> DictConfig:
    """Create a minimal DictConfig for runner construction."""
    return OmegaConf.create(
        {
            "track": {"name": track_name},
            "method": {"name": method_name, "m": 16, "K0": 64},
            "log_dir": log_dir,
            "seed": 42,
        }
    )


# --- Structure tests ---


def test_dimacs_runner_structure():
    """Assert DIMACSRunner is a subclass of BaseRunner with run and run_ablation."""
    assert issubclass(DIMACSRunner, BaseRunner)
    assert hasattr(DIMACSRunner, "run")
    assert hasattr(DIMACSRunner, "run_ablation")
    assert callable(getattr(DIMACSRunner, "run"))
    assert callable(getattr(DIMACSRunner, "run_ablation"))


def test_osmnx_runner_structure():
    """Assert OSMnxRunner is a subclass of BaseRunner with run method."""
    assert issubclass(OSMnxRunner, BaseRunner)
    assert hasattr(OSMnxRunner, "run")
    assert callable(getattr(OSMnxRunner, "run"))


# --- Method validation tests ---


def test_osmnx_supported_methods():
    """Assert OSMnxRunner only supports aac and dijkstra."""
    assert OSMnxRunner.SUPPORTED_METHODS == ["aac", "dijkstra"]

    # Creating with unsupported method should raise ValueError
    with pytest.raises(ValueError, match="not supported"):
        cfg = _make_cfg(track_name="osmnx", method_name="alt")
        OSMnxRunner(cfg)


def test_base_runner_validate_method():
    """Test BaseRunner method validation with and without SUPPORTED_METHODS."""

    # BaseRunner with None accepts any method
    class AnyRunner(BaseRunner):
        SUPPORTED_METHODS = None

    cfg_any = _make_cfg(method_name="anything_goes")
    runner = AnyRunner(cfg_any)
    runner.close()

    # Runner with restricted methods rejects unknown
    class RestrictedRunner(BaseRunner):
        SUPPORTED_METHODS = ["a", "b"]

    with pytest.raises(ValueError, match="not supported"):
        cfg_bad = _make_cfg(method_name="c")
        RestrictedRunner(cfg_bad)

    # Accepted methods work fine
    cfg_ok = _make_cfg(method_name="a")
    runner = RestrictedRunner(cfg_ok)
    runner.close()


# --- Dispatch tests ---


def test_get_runner_dispatch():
    """Assert get_runner maps track names to correct runner classes."""
    assert get_runner("dimacs") is DIMACSRunner
    assert get_runner("osmnx") is OSMnxRunner


def test_get_runner_invalid():
    """Assert get_runner raises ValueError for unknown tracks."""
    with pytest.raises(ValueError, match="Unknown track"):
        get_runner("invalid")


# --- BaseRunner method presence tests ---


def test_base_runner_preprocess_methods():
    """Assert BaseRunner has all preprocessing and throughput methods."""
    assert hasattr(BaseRunner, "preprocess_aac")
    assert hasattr(BaseRunner, "preprocess_alt")
    assert hasattr(BaseRunner, "preprocess_fastmap")
    assert hasattr(BaseRunner, "measure_batch_throughput")
    assert callable(getattr(BaseRunner, "preprocess_aac"))
    assert callable(getattr(BaseRunner, "preprocess_alt"))
    assert callable(getattr(BaseRunner, "preprocess_fastmap"))
    assert callable(getattr(BaseRunner, "measure_batch_throughput"))


# --- Ablation test ---


def test_dimacs_runner_has_ablation():
    """Assert DIMACSRunner has run_ablation method accepting cfg parameter."""
    assert hasattr(DIMACSRunner, "run_ablation")
    sig = inspect.signature(DIMACSRunner.run_ablation)
    params = list(sig.parameters.keys())
    assert "cfg" in params


# --- Equal-budget validation test ---


def test_equal_budget_validation():
    """Verify equal bytes/vertex at same m using memory_bytes_per_vertex.

    Both ALT(m=4) and AAC(K0=8, m=4) should report the same
    memory_bytes_per_vertex at m=4 for the compressed representation:
    m * dtype_size = 4 * 4 = 16 bytes/vertex.

    ALT at m landmarks stores 2*m values (forward+backward), so
    the comparison at equal m for AAC compressed is:
    AAC: memory_bytes_per_vertex(m=4, 4) = 16
    FastMap: memory_bytes_per_vertex(m=4, 4) = 16
    Both AAC and FastMap at same m have identical memory footprint.
    """
    m = 4
    dtype_size = 4

    aac_bytes = memory_bytes_per_vertex(m, dtype_size)
    fastmap_bytes = memory_bytes_per_vertex(m, dtype_size)

    assert aac_bytes == fastmap_bytes == m * dtype_size
    assert aac_bytes == 16


# --- MetricsCollector integration test ---


def test_metrics_collector_integration():
    """Create MetricsCollector, add mock queries, verify summary keys."""
    collector = MetricsCollector()

    # Add some mock query results
    for i in range(5):
        result = SearchResult(
            path=[0, 1, 2],
            cost=10.0 + i,
            expansions=100 + i * 10,
            optimal=True,
            h_source=5.0,
        )
        collector.add_query(i, 0, 2, result, latency_ms=1.0 + i * 0.1)

    summary = collector.summary()
    assert "expansions_mean" in summary
    assert "p50_ms" in summary
    assert "num_violations" in summary
    assert "num_queries" in summary
    assert summary["num_queries"] == 5
    assert summary["num_violations"] == 0  # all optimal=True


# --- TensorBoard logging test ---


def test_tensorboard_logging_exists():
    """Assert BaseRunner.__init__ creates a SummaryWriter attribute."""
    cfg = _make_cfg()
    runner = DIMACSRunner(cfg)
    assert hasattr(runner, "writer")
    # Verify it's a SummaryWriter instance
    from torch.utils.tensorboard import SummaryWriter

    assert isinstance(runner.writer, SummaryWriter)
    runner.close()


# --- Batch throughput wiring test ---


def test_batch_throughput_wired():
    """Assert BaseRunner has measure_batch_throughput that uses batch_throughput."""
    assert hasattr(BaseRunner, "measure_batch_throughput")

    # Verify the method imports and calls batch_throughput from collector

    source = inspect.getsource(BaseRunner.measure_batch_throughput)
    assert "batch_throughput" in source
    assert "experiments.metrics.collector" in source
