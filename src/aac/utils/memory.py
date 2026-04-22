"""Memory estimation and guard utilities for large-scale SSSP computation.

Provides functions to estimate teacher label memory requirements,
detect available system RAM, and automatically adjust chunk sizes
to prevent out-of-memory conditions on large graphs.
"""

from __future__ import annotations

import logging
import warnings

import torch

logger = logging.getLogger(__name__)


def get_available_memory_bytes() -> int:
    """Return available system memory in bytes by parsing /proc/meminfo.

    Reads the ``MemAvailable`` line from ``/proc/meminfo`` on Linux.

    Returns:
        Available memory in bytes, or -1 if detection fails.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # Format: "MemAvailable:   12345678 kB"
                    parts = line.split()
                    kb = int(parts[1])
                    return kb * 1024
        # MemAvailable line not found
        logger.warning("MemAvailable not found in /proc/meminfo")
        return -1
    except (OSError, ValueError, IndexError) as exc:
        logger.warning("Could not read /proc/meminfo: %s", exc)
        return -1


def estimate_teacher_label_memory(
    num_anchors: int,
    num_nodes: int,
    is_directed: bool,
    dtype: torch.dtype = torch.float64,
) -> int:
    """Estimate peak memory for teacher label tensors in bytes.

    Formula: ``num_anchors * num_nodes * bytes_per_element * directions``

    For directed graphs, both forward (d_out) and reverse (d_in) distance
    matrices are stored, doubling the memory requirement.

    Args:
        num_anchors: Number of anchor/landmark vertices (K).
        num_nodes: Number of graph vertices (V).
        is_directed: Whether the graph is directed (doubles memory for d_in).
        dtype: Tensor dtype. Default torch.float64 (8 bytes per element).

    Returns:
        Estimated memory in bytes.
    """
    bytes_per_element = 8 if dtype == torch.float64 else 4
    directions = 2 if is_directed else 1
    return num_anchors * num_nodes * bytes_per_element * directions


def memory_guard(
    num_anchors: int,
    num_nodes: int,
    is_directed: bool,
    chunk_size: int | None,
    dtype: torch.dtype = torch.float64,
    safety_factor: float = 0.8,
) -> int | None:
    """Check if teacher label computation fits in available RAM.

    If estimated memory exceeds available RAM (scaled by safety_factor),
    automatically reduces ``chunk_size`` to fit and issues a
    :class:`ResourceWarning`.

    Args:
        num_anchors: Number of anchor vertices (K).
        num_nodes: Number of graph vertices (V).
        is_directed: Whether the graph is directed.
        chunk_size: Current chunk size (None means no chunking).
        dtype: Tensor dtype for labels.
        safety_factor: Fraction of available RAM to use (default 0.8).

    Returns:
        Adjusted chunk_size. Returns ``None`` if no chunking is needed
        and the full computation fits in memory. Returns a positive int
        if chunking is required.
    """
    estimated = estimate_teacher_label_memory(num_anchors, num_nodes, is_directed, dtype)
    available = get_available_memory_bytes()

    if available < 0:
        logger.warning(
            "Could not detect available memory; skipping guard"
        )
        return chunk_size

    safe_budget = int(available * safety_factor)

    if estimated <= safe_budget:
        return chunk_size

    # Compute maximum chunk_size that fits in the safe budget
    bytes_per_element = 8 if dtype == torch.float64 else 4
    directions = 2 if is_directed else 1
    per_anchor_bytes = num_nodes * bytes_per_element * directions
    max_chunk = max(1, safe_budget // per_anchor_bytes)

    est_gb = estimated / (1024**3)
    avail_gb = available / (1024**3)
    logger.warning(
        "Estimated memory (%.1f GB) exceeds available (%.1f GB). "
        "Auto-reducing chunk_size to %d.",
        est_gb,
        avail_gb,
        max_chunk,
    )
    warnings.warn(
        f"Estimated memory ({est_gb:.1f} GB) exceeds available "
        f"({avail_gb:.1f} GB). Auto-reducing chunk_size to {max_chunk}.",
        ResourceWarning,
        stacklevel=3,
    )

    if chunk_size is not None:
        return min(chunk_size, max_chunk)
    return max_chunk
