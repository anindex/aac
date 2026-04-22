"""Tests for memory estimation and guard utilities."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from aac.utils.memory import (
    estimate_teacher_label_memory,
    get_available_memory_bytes,
    memory_guard,
)

# ---------------------------------------------------------------------------
# estimate_teacher_label_memory
# ---------------------------------------------------------------------------


def test_estimate_memory_undirected_float64():
    """K=16, V=1000, undirected, float64 -> 16 * 1000 * 8 * 1 = 128000 bytes."""
    result = estimate_teacher_label_memory(
        num_anchors=16, num_nodes=1000, is_directed=False, dtype=torch.float64
    )
    assert result == 128_000


def test_estimate_memory_directed_float64():
    """K=16, V=1000, directed, float64 -> 16 * 1000 * 8 * 2 = 256000 bytes."""
    result = estimate_teacher_label_memory(
        num_anchors=16, num_nodes=1000, is_directed=True, dtype=torch.float64
    )
    assert result == 256_000


def test_estimate_memory_float32():
    """K=16, V=1000, undirected, float32 -> 16 * 1000 * 4 * 1 = 64000 bytes."""
    result = estimate_teacher_label_memory(
        num_anchors=16, num_nodes=1000, is_directed=False, dtype=torch.float32
    )
    assert result == 64_000


# ---------------------------------------------------------------------------
# get_available_memory_bytes
# ---------------------------------------------------------------------------


def test_get_available_memory_returns_positive():
    """On Linux, get_available_memory_bytes() must return a positive integer."""
    result = get_available_memory_bytes()
    assert result > 0, f"Expected positive available memory, got {result}"


# ---------------------------------------------------------------------------
# memory_guard
# ---------------------------------------------------------------------------


def test_memory_guard_no_reduction_when_fits():
    """When estimated memory fits in available RAM, chunk_size is returned unchanged."""
    with patch(
        "aac.utils.memory.get_available_memory_bytes", return_value=1_000_000_000
    ):
        result = memory_guard(
            num_anchors=16,
            num_nodes=1000,
            is_directed=False,
            chunk_size=8,
            dtype=torch.float64,
        )
        # estimated = 16 * 1000 * 8 * 1 = 128000 << 1 GB
        assert result == 8


def test_memory_guard_reduces_chunk_when_exceeds():
    """When estimated memory exceeds available, chunk_size is reduced to fit."""
    with patch(
        "aac.utils.memory.get_available_memory_bytes", return_value=100_000
    ):
        result = memory_guard(
            num_anchors=64,
            num_nodes=100_000,
            is_directed=False,
            chunk_size=64,
            dtype=torch.float64,
        )
        # estimated = 64 * 100000 * 8 * 1 = 51.2 MB >> 100 KB
        # safe_budget = 100000 * 0.8 = 80000
        # max_chunk = 80000 // (100000 * 8 * 1) = 0 -> clamped to 1
        assert result >= 1
        assert result < 64


def test_memory_guard_warns_when_reducing():
    """When guard reduces chunk_size, a ResourceWarning is issued."""
    with patch(
        "aac.utils.memory.get_available_memory_bytes", return_value=100_000
    ):
        with pytest.warns(ResourceWarning, match="exceeds available"):
            memory_guard(
                num_anchors=64,
                num_nodes=100_000,
                is_directed=False,
                chunk_size=64,
                dtype=torch.float64,
            )


def test_memory_guard_returns_none_when_no_chunk_and_fits():
    """When chunk_size=None and estimated memory fits, returns None (no chunking needed)."""
    with patch(
        "aac.utils.memory.get_available_memory_bytes", return_value=1_000_000_000
    ):
        result = memory_guard(
            num_anchors=16,
            num_nodes=1000,
            is_directed=False,
            chunk_size=None,
            dtype=torch.float64,
        )
        assert result is None


def test_memory_guard_activates_chunking_when_no_chunk_but_exceeds():
    """When chunk_size=None but estimated exceeds available, returns positive int chunk_size."""
    with patch(
        "aac.utils.memory.get_available_memory_bytes", return_value=100_000
    ):
        with pytest.warns(ResourceWarning):
            result = memory_guard(
                num_anchors=64,
                num_nodes=100_000,
                is_directed=False,
                chunk_size=None,
                dtype=torch.float64,
            )
        assert isinstance(result, int)
        assert result >= 1
