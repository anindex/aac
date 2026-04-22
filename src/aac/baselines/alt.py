"""ALT baseline: landmark-based triangle inequality heuristic.

ALT is the identity-compression special case of AAC (A=I).
Same anchor selection + SSSP preprocessing, different heuristic evaluation.
Uses the triangle inequality with landmarks for admissible lower bounds.

References:
    Goldberg & Harrelson (2005). Computing the Shortest Path: A* Search
    Meets Graph Theory. SODA.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.types import Graph, TeacherLabels


def alt_preprocess(
    graph: Graph,
    num_landmarks: int,
    seed_vertex: int | None = None,
    rng: torch.Generator | None = None,
    valid_vertices: torch.Tensor | None = None,
) -> TeacherLabels:
    """Preprocess graph for ALT heuristic using farthest-point sampling.

    ALT is the identity-compression special case of AAC (A=I). Same anchor
    selection + SSSP, different heuristic evaluation.

    Args:
        graph: Input graph in CSR format.
        num_landmarks: Number of landmarks to select (K).
        seed_vertex: Starting vertex index for FPS. Random if None.
        rng: Optional torch generator for reproducible seed selection.
        valid_vertices: Optional (N,) int64 tensor restricting anchor selection
            to a subset of vertices (e.g., largest connected component).

    Returns:
        TeacherLabels with d_out (K, V) and d_in (K, V) distance matrices.
    """
    anchors = farthest_point_sampling(
        graph, num_landmarks, seed_vertex=seed_vertex, rng=rng,
        valid_vertices=valid_vertices,
    )
    return compute_teacher_labels(graph, anchors, use_gpu=False)


def make_alt_heuristic(
    teacher_labels: TeacherLabels,
) -> Callable[[int, int], float]:
    """Create A*-compatible heuristic from ALT preprocessing.

    For directed graphs:
        h(u,t) = max(0, max_k(d_out[k,t] - d_out[k,u]), max_k(d_in[k,u] - d_in[k,t]))
    For undirected graphs:
        h(u,t) = max(0, max_k |d_out[k,u] - d_out[k,t]|)

    Sentinel masking: landmarks where either vertex is unreachable
    (distance >= 0.99 * SENTINEL) are excluded from the max to prevent
    inadmissible overestimates. Returns 0 (Dijkstra fallback) if all
    landmarks are masked.

    Args:
        teacher_labels: TeacherLabels from alt_preprocess.

    Returns:
        Callable h(node: int, target: int) -> float.
    """
    import numpy as np
    from aac.utils.numerics import SENTINEL

    # Pre-convert to numpy for fair timing comparison with AAC
    # (avoids torch tensor indexing + .item() overhead per call).
    d_out_np = teacher_labels.d_out.detach().cpu().numpy()  # (K, V)
    d_in_np = teacher_labels.d_in.detach().cpu().numpy()    # (K, V)
    is_directed = teacher_labels.is_directed
    sentinel_thresh = 0.99 * SENTINEL

    if is_directed:
        def h(node: int, target: int) -> float:
            # Forward bound: d(k,t) - d(k,u) <= d(u,t)
            d_out_n = d_out_np[:, node]
            d_out_t = d_out_np[:, target]
            fwd_valid = (d_out_n < sentinel_thresh) & (d_out_t < sentinel_thresh)

            # Backward bound: d(u,k) - d(t,k) <= d(u,t)
            d_in_n = d_in_np[:, node]
            d_in_t = d_in_np[:, target]
            bwd_valid = (d_in_n < sentinel_thresh) & (d_in_t < sentinel_thresh)

            result = 0.0
            if fwd_valid.any():
                result = max(result, float(np.max((d_out_t - d_out_n)[fwd_valid])))
            if bwd_valid.any():
                result = max(result, float(np.max((d_in_n - d_in_t)[bwd_valid])))
            return result
    else:
        def h(node: int, target: int) -> float:
            d_n = d_out_np[:, node]
            d_t = d_out_np[:, target]
            valid = (d_n < sentinel_thresh) & (d_t < sentinel_thresh)
            if not valid.any():
                return 0.0
            return max(0.0, float(np.max(np.abs((d_n - d_t)[valid]))))

    return h


def alt_memory_bytes(
    num_landmarks: int,
    dtype_size: int = 4,
    is_directed: bool = True,
) -> int:
    """Compute per-vertex deployed memory for ALT.

    On directed graphs, ALT stores both forward (d_out) and reverse (d_in)
    distances per landmark per vertex for 2*K floats/vertex. On undirected
    graphs, d_in = d_out and only a single K-column distance table is stored
    (see aac.embeddings.sssp.compute_teacher_labels), so storage is K floats
    per vertex.

    Args:
        num_landmarks: Number of landmarks (K).
        dtype_size: Bytes per element. Default 4 (float32) per METR-05.
        is_directed: Whether the underlying graph is directed. Default True
            preserves the historical convention used throughout the DIMACS
            and OSMnx road-network tables. Pass False for SBM/BA/OGB-arXiv
            and other undirected graphs.

    Returns:
        Per-vertex memory in bytes: (2*K if directed else K) * dtype_size.
    """
    factor = 2 if is_directed else 1
    return factor * num_landmarks * dtype_size
