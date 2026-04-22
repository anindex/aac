"""GPU-aware timing utility with warmup and percentile reporting."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TimingResult:
    """Timing measurement with percentile statistics.

    Attributes:
        median_ms: Median latency in milliseconds.
        p50_ms: 50th percentile (same as median).
        p95_ms: 95th percentile latency.
        min_ms: Minimum observed latency.
        max_ms: Maximum observed latency.
        num_runs: Number of timed runs.
    """

    median_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    num_runs: int


def time_query(
    fn: Callable, warmup_runs: int = 10, num_runs: int = 100
) -> TimingResult:
    """Time a callable with warmup, GPU sync, and percentile stats.

    Runs the function warmup_runs times (results discarded), then
    num_runs times with precise timing. If CUDA is available,
    torch.cuda.synchronize() is called before each start/stop.

    Args:
        fn: Zero-argument callable to time.
        warmup_runs: Number of warmup invocations.
        num_runs: Number of timed invocations.

    Returns:
        TimingResult with percentile statistics.
    """
    cuda_available = torch.cuda.is_available()

    # Warmup phase
    for _ in range(warmup_runs):
        fn()

    # Timed phase
    times_ms: list[float] = []
    for _ in range(num_runs):
        if cuda_available:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if cuda_available:
            torch.cuda.synchronize()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    arr = np.array(times_ms)
    arr.sort()

    return TimingResult(
        median_ms=float(np.median(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        num_runs=num_runs,
    )
