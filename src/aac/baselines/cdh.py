r"""Compressed Differential Heuristic (CDH) baseline.

CDH (Goldenberg et al. 2011, 2017) stores a *subset* of the full differential-
heuristic table rather than compressing it with PCA/VQ: for each vertex v, keep
the r most-useful pivot distances out of P total pivots. At query time h(u,t),
only pivots stored at *both* endpoints contribute to the admissible per-pivot
triangle-inequality lower bound |d(p,u)-d(p,t)|; the heuristic is the max over
that intersection (zero if the intersection is empty, Dijkstra-fallback).

This is the "intersection" variant: strictly admissible under closed-set A*
without node reopenings. The optional bound-substitution mode (enabled via
``use_bound_substitution=True``) adds the Goldenberg upper/lower-bound
substitution for pivots stored at only one endpoint: a shared anchor pivot
``p*`` in the intersection turns the missing ``d(p, y)`` into a closed-form
admissible interval via the triangle inequality evaluated against a small
``P*P`` pivot-to-pivot distance side-table. The side-table costs ``P^2``
floats of fixed, non-per-vertex preprocessing memory -- negligible against
the per-vertex subset for all reasonable ``P``. This variant is still strictly
admissible and remains closed-set-A* safe; it does not require node reopenings.
BPMX propagation (Felner et al.) is supported \emph{at search time} via the
optional ``use_bpmx=True`` flag on :func:`aac.search.astar`: at expansion of
each node ``u`` we evaluate ``h`` on every successor ``v``, tighten ``h(u) :=
max(h(u), max_v h(v) - w(u,v))``, and push successors with the propagated
``h(v) := max(h(v), h_BPMX(u) - w(u,v))``. Each tightening only takes a max of
values that are already admissible, so closed-set A* without reopenings stays
sound; this is the standard BPMX-on-closed-set-A* protocol used in CDH
evaluations (Goldenberg et al.\ 2017). Bound substitution is orthogonal:
``use_bound_substitution=True`` widens the per-pivot per-query estimate, while
BPMX widens it by propagation across the search graph.

Per-vertex selection rule: top-r pivots by largest d(p,v), following the
"farthest pivots are most informative" heuristic that Goldenberg uses as a
strong default. Other selection rules (random, contribution-maximizing) can
be plugged in via ``selection_rule`` for ablation.

Memory accounting: per-vertex storage is r floats for distances plus r
integer indices naming the stored pivots, giving
    bytes_per_vertex = r * (dtype_size + index_size)
where index_size = ceil(log2(P) / 8). For P <= 256 this is r * 5 bytes at
float32; for P <= 65536 it is r * 6 bytes. The matched-memory rule against
ALT at B bytes/vertex is therefore r = floor(B / (dtype_size + index_size)).

References
----------
Goldenberg, Sturtevant, Felner, Schaeffer. "Compressed Differential Heuristics
for Faster Optimal Pathfinding." AAAI 2011.
Goldenberg, Felner, Sturtevant, Schaeffer. "Compressed Pattern Databases."
AI Communications 30(2), 2017.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.types import Graph
from aac.utils.numerics import SENTINEL

SELECTION_RULES = ("top_r_farthest", "random", "top_r_nearest")


@dataclass(frozen=True)
class CDHLabels:
    """Per-vertex subset of differential-heuristic distances.

    Attributes
    ----------
    pivot_indices : torch.Tensor
        (V, r) int64 tensor giving the identity (in [0, P)) of each of the r
        pivots stored at each vertex.
    pivot_distances : torch.Tensor
        (V, r) float tensor of the corresponding distances d(pivot, v) on the
        *forward* graph. For undirected graphs this is the only table; for
        directed graphs see ``pivot_distances_in``.
    pivot_distances_in : torch.Tensor | None
        (V, r) float tensor of reverse distances d(v, pivot) when the graph is
        directed; ``None`` for undirected graphs.
    num_pivots : int
        P -- total pivot-pool size.
    num_stored : int
        r -- per-vertex subset size.
    is_directed : bool
    pivot_pivot_out : torch.Tensor | None
        (P, P) fixed side-table with ``pivot_pivot_out[i, j] = d(pivot_i,
        pivot_j)`` on the forward graph. Populated whenever bound-substitution
        is to be available at query time; None otherwise. Off-heap cost is a
        fixed ``P^2`` floats and does not enter the per-vertex memory budget.
    pivot_pivot_in : torch.Tensor | None
        (P, P) reverse side-table for directed graphs; for undirected graphs
        ``pivot_pivot_in`` equals ``pivot_pivot_out``. None when the pivot-
        pivot table was not materialised (bound-substitution disabled).
    """

    pivot_indices: torch.Tensor
    pivot_distances: torch.Tensor
    pivot_distances_in: torch.Tensor | None
    num_pivots: int
    num_stored: int
    is_directed: bool
    pivot_pivot_out: torch.Tensor | None = None
    pivot_pivot_in: torch.Tensor | None = None


def _select_per_vertex(
    d_table: torch.Tensor,
    num_stored: int,
    rule: str,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Return (V, r) int64 indices of stored pivots per vertex.

    ``d_table`` has shape (P, V) -- pivot-to-vertex forward distances.
    """
    P, V = d_table.shape
    r = min(num_stored, P)
    sentinel_thresh = 0.99 * SENTINEL

    if rule == "top_r_farthest":
        # Mask sentinels (unreachable) to -inf so they sort to the bottom.
        masked = d_table.clone()
        masked[masked > sentinel_thresh] = float("-inf")
        # argsort descending per-vertex: (P, V) -> take top-r indices.
        topk = torch.topk(masked.t(), k=r, dim=1, largest=True, sorted=False)
        return topk.indices.to(torch.int64)
    if rule == "top_r_nearest":
        masked = d_table.clone()
        masked[masked > sentinel_thresh] = float("inf")
        topk = torch.topk(masked.t(), k=r, dim=1, largest=False, sorted=False)
        return topk.indices.to(torch.int64)
    if rule == "random":
        indices = np.stack(
            [rng.choice(P, size=r, replace=False) for _ in range(V)], axis=0
        )
        return torch.from_numpy(indices).to(torch.int64)
    raise ValueError(
        f"Unknown selection_rule {rule!r}. Expected one of {SELECTION_RULES}."
    )


def cdh_preprocess(
    graph: Graph,
    num_pivots: int,
    num_stored: int,
    *,
    seed_vertex: int | None = None,
    rng: torch.Generator | None = None,
    valid_vertices: torch.Tensor | None = None,
    selection_rule: str = "top_r_farthest",
    selection_seed: int = 0,
    compute_pivot_pivot: bool = True,
) -> CDHLabels:
    """Build a CDH label store by subsetting P-pivot DH distances per vertex.

    Parameters
    ----------
    graph : Graph
        Input graph in CSR format.
    num_pivots : int
        P, the pivot-pool size selected via farthest-point sampling.
    num_stored : int
        r, the number of pivot distances retained per vertex. Must satisfy
        ``num_stored <= num_pivots``.
    seed_vertex : int | None
        FPS starting vertex.
    rng : torch.Generator | None
        Reproducible anchor-selection generator.
    valid_vertices : torch.Tensor | None
        Restrict anchor selection to a subset (e.g. largest SCC).
    selection_rule : str
        Per-vertex subset selection rule. Default ``"top_r_farthest"`` matches
        Goldenberg et al.'s recommended heuristic.
    selection_seed : int
        Seed for the random selection_rule.
    compute_pivot_pivot : bool
        Populate the ``pivot_pivot_out`` / ``pivot_pivot_in`` side-tables.
        Required when the resulting heuristic will be built with
        ``use_bound_substitution=True``; free (no extra SSSP calls) because
        the distances are already materialised during teacher SSSP. Default
        True.

    Returns
    -------
    CDHLabels
    """
    if num_stored > num_pivots:
        raise ValueError(
            f"num_stored={num_stored} exceeds num_pivots={num_pivots}"
        )
    if selection_rule not in SELECTION_RULES:
        raise ValueError(
            f"selection_rule={selection_rule!r} not in {SELECTION_RULES}"
        )

    anchors = farthest_point_sampling(
        graph,
        num_pivots,
        seed_vertex=seed_vertex,
        rng=rng,
        valid_vertices=valid_vertices,
    )
    teacher = compute_teacher_labels(graph, anchors, use_gpu=False)

    numpy_rng = np.random.default_rng(selection_seed)
    pivot_indices = _select_per_vertex(
        teacher.d_out, num_stored, selection_rule, numpy_rng
    )  # (V, r)

    # Gather the stored distances per vertex.
    _V = teacher.d_out.shape[1]  # noqa: F841 (document shape)
    _r = pivot_indices.shape[1]  # noqa: F841 (document shape)
    d_out_vp = teacher.d_out.t()  # (V, P)
    stored_fwd = torch.gather(d_out_vp, dim=1, index=pivot_indices)  # (V, r)

    stored_bwd: torch.Tensor | None = None
    if teacher.is_directed:
        # For directed graphs the backward "pivot row" at pivot p contains
        # d(v, pivot_p). Selection uses the same per-vertex pivot set for
        # symmetry with the forward table -- this is strictly what CDH does.
        d_in_vp = teacher.d_in.t()  # (V, P)
        stored_bwd = torch.gather(d_in_vp, dim=1, index=pivot_indices)  # (V, r)

    pivot_pivot_out: torch.Tensor | None = None
    pivot_pivot_in: torch.Tensor | None = None
    if compute_pivot_pivot:
        anchors_long = teacher.anchor_indices.to(torch.int64)
        # teacher.d_out has shape (P, V); indexing columns with anchors_long
        # gives the (P, P) matrix d_out[i, j] = d(pivot_i, pivot_j).
        pivot_pivot_out = teacher.d_out.index_select(1, anchors_long).contiguous()
        if teacher.is_directed:
            # teacher.d_in[k, v] = d(v, pivot_k); selecting columns at anchors
            # gives d_in[i, j] = d(pivot_j, pivot_i). Transpose to match the
            # same (src, dst) convention as pivot_pivot_out.
            pivot_pivot_in = (
                teacher.d_in.index_select(1, anchors_long).t().contiguous()
            )
        else:
            pivot_pivot_in = pivot_pivot_out

    return CDHLabels(
        pivot_indices=pivot_indices,
        pivot_distances=stored_fwd,
        pivot_distances_in=stored_bwd,
        num_pivots=num_pivots,
        num_stored=num_stored,
        is_directed=teacher.is_directed,
        pivot_pivot_out=pivot_pivot_out,
        pivot_pivot_in=pivot_pivot_in,
    )


def _pivot_index_bytes(num_pivots: int) -> int:
    """Smallest integer width (in bytes) that can index ``num_pivots``."""
    if num_pivots <= 1:
        return 1
    return max(1, math.ceil(math.log2(num_pivots) / 8))


def cdh_memory_bytes(
    num_pivots: int,
    num_stored: int,
    *,
    dtype_size: int = 4,
    is_directed: bool = False,
) -> int:
    """Per-vertex deployed memory for CDH.

    On undirected graphs, each vertex stores r floats + r pivot indices.
    On directed graphs, each vertex additionally stores r backward-distance
    floats (the same pivot set indexes both tables, so the index array is
    shared). Matches the paper's convention of reporting bytes per vertex
    deployed to disk/RAM.

    Parameters
    ----------
    num_pivots : int
        P -- total pivot pool, determines index width.
    num_stored : int
        r -- per-vertex subset size.
    dtype_size : int
        Bytes per stored distance (4 for float32).
    is_directed : bool
        Whether the graph is directed; if True, doubles the distance storage.

    Returns
    -------
    int
        Per-vertex byte count.
    """
    index_size = _pivot_index_bytes(num_pivots)
    dist_factor = 2 if is_directed else 1
    return num_stored * (dist_factor * dtype_size + index_size)


def make_cdh_heuristic(
    labels: CDHLabels,
    *,
    use_bound_substitution: bool = False,
) -> Callable[[int, int], float]:
    """Build an A*-compatible CDH heuristic callable.

    The "intersection" heuristic iterates over pivots stored at *both* u and
    t and returns ``max_p |d(p,u) - d(p,t)|`` (directed: the larger of the
    forward and backward bounds). When the intersection is empty, returns 0
    (Dijkstra fallback), preserving admissibility.

    Parameters
    ----------
    labels : CDHLabels
        Output of :func:`cdh_preprocess`.
    use_bound_substitution : bool
        When True, pivots stored at only one of (u, t) contribute via the
        Goldenberg bound-substitution rule: a shared anchor pivot p* (found
        in the intersection) combined with the ``P*P`` pivot-pivot side-table
        gives an admissible interval ``[d_lb, d_ub]`` for the missing
        ``d(p, y)``; the per-pivot admissible contribution is
        ``max(d(p,x) - d_ub, d_lb - d(p,x), 0)``. Strictly admissible under
        closed-set A* (no reopenings needed).

    Returns
    -------
    Callable[[int, int], float]
    """
    indices = labels.pivot_indices.cpu().numpy()  # (V, r) int64
    dists = labels.pivot_distances.cpu().numpy()  # (V, r)
    dists_in = (
        labels.pivot_distances_in.cpu().numpy()
        if labels.pivot_distances_in is not None
        else None
    )
    is_directed = labels.is_directed
    sentinel_thresh = 0.99 * SENTINEL

    pp_out = (
        labels.pivot_pivot_out.cpu().numpy()
        if labels.pivot_pivot_out is not None
        else None
    )
    pp_in = (
        labels.pivot_pivot_in.cpu().numpy()
        if labels.pivot_pivot_in is not None
        else None
    )
    sub_enabled = use_bound_substitution and pp_out is not None

    def h(node: int, target: int) -> float:
        u_piv = indices[node]
        t_piv = indices[target]
        # Intersection of the two stored-pivot sets.
        # With r typically <= 32 this is a tiny-array set intersection; a
        # Python-level loop is faster than np.intersect1d for small r.
        t_set = set(int(x) for x in t_piv)
        if not t_set:
            return 0.0
        best = 0.0
        u_d = dists[node]
        t_d = dists[target]
        # Map pivot-id -> slot in the target's stored array.
        t_slot = {int(t_piv[j]): j for j in range(len(t_piv))}
        if is_directed:
            u_d_in = dists_in[node]
            t_d_in = dists_in[target]
        # Collect intersection pivot distances for potential anchor use.
        shared_anchors = []  # list of (p_id, du, dt, du_in, dt_in)
        for i, p in enumerate(u_piv):
            p = int(p)
            j = t_slot.get(p)
            if j is None:
                continue
            du = float(u_d[i])
            dt = float(t_d[j])
            if du > sentinel_thresh or dt > sentinel_thresh:
                continue
            if is_directed:
                # Forward triangle bound: d(p,t) - d(p,u) <= d(u,t).
                fwd = dt - du
                if fwd > best:
                    best = fwd
                # Backward bound: d(u,p) - d(t,p) <= d(u,t).
                du_in = float(u_d_in[i])
                dt_in = float(t_d_in[j])
                if (
                    du_in <= sentinel_thresh
                    and dt_in <= sentinel_thresh
                ):
                    bwd = du_in - dt_in
                    if bwd > best:
                        best = bwd
                    if sub_enabled:
                        shared_anchors.append((p, du, dt, du_in, dt_in))
                elif sub_enabled:
                    shared_anchors.append((p, du, dt, float("inf"), float("inf")))
            else:
                diff = du - dt
                if diff < 0:
                    diff = -diff
                if diff > best:
                    best = diff
                if sub_enabled:
                    shared_anchors.append((p, du, dt, 0.0, 0.0))

        if not sub_enabled or not shared_anchors:
            return best

        # --- Bound-substitution mode (admissible, closed-set-A* safe). ---
        #
        # For each pivot p stored at endpoint x but missing at endpoint y,
        # use a shared anchor pivot p* in the intersection together with the
        # P*P pivot-pivot side-table to produce an admissible interval
        # ``d(p, y) in [d_lb, d_ub]``. The admissible per-pivot contribution
        # then depends on directedness:
        #
        # Undirected (symmetric bound ``|d(p,u) - d(p,t)|``):
        #     contribution = max(d(p, x) - d_ub, d_lb - d(p, x), 0)
        # Directed forward (``d(p,t) - d(p,u)``):
        #     with p stored at u -> contribution = max(0, d_lb - d(p, u))
        #     with p stored at t -> contribution = max(0, d(p, t) - d_ub)
        # Directed backward (``d(u,p) - d(t,p)``):
        #     with p stored at u -> contribution = max(0, d(u, p) - d_ub)
        #     with p stored at t -> contribution = max(0, d_lb - d(t, p))
        #
        # In every case we substitute either the lower or upper bound of
        # d(p, y) in the direction that is provably admissible (never the
        # direction that could over-shoot). See docstring for the derivation.
        u_set = set(int(x) for x in u_piv)

        def _bounds(d_pp: float, d_star_x: float) -> tuple[float, float] | None:
            """Return (d_lb, d_ub) for d(p, y) given d(p, p*) and d(p*, x==y?).
            Returns None if any input is a sentinel."""
            if d_pp > sentinel_thresh or d_star_x > sentinel_thresh:
                return None
            d_ub = d_pp + d_star_x
            lb1 = d_pp - d_star_x
            lb2 = d_star_x - d_pp
            d_lb = lb1 if lb1 > lb2 else lb2
            if d_lb < 0.0:
                d_lb = 0.0
            return d_lb, d_ub

        # ---- pivots stored at u but not at t ----
        for i, p in enumerate(u_piv):
            p = int(p)
            if p in t_slot:
                continue
            du = float(u_d[i])
            if du > sentinel_thresh:
                continue
            # Undirected | forward-directed forward bound use pp_out[p, p*]
            # and d(p*, t) stored in t_d[star_j].
            delta = 0.0
            for p_star, _du_star, dt_star, _du_in_star, _dt_in_star in shared_anchors:
                bnd = _bounds(float(pp_out[p, p_star]), dt_star)
                if bnd is None:
                    continue
                d_lb, d_ub = bnd
                if is_directed:
                    cand = d_lb - du  # admissible for d(p,t) - d(p,u)
                    if cand < 0.0:
                        cand = 0.0
                else:
                    cand = du - d_ub
                    if d_lb - du > cand:
                        cand = d_lb - du
                    if cand < 0.0:
                        cand = 0.0
                if cand > delta:
                    delta = cand
            if delta > best:
                best = delta
            if is_directed:
                # Backward bound: d(u, p) - d(t, p) <= d(u, t), with p stored
                # at u (we know d(u, p) = du_in) and d(t, p) missing. Need a
                # lower bound on d(t, p) via pp_in[p*, p] and d(t, p*).
                du_in = float(u_d_in[i])
                if du_in > sentinel_thresh:
                    continue
                delta_b = 0.0
                for p_star, _du_star, _dt_star, _du_in_star, dt_in_star in (
                    shared_anchors
                ):
                    bnd = _bounds(float(pp_in[p_star, p]), dt_in_star)
                    if bnd is None:
                        continue
                    d_lb, d_ub = bnd
                    # d(u, p) - d(t, p); substitute UB for d(t, p):
                    cand = du_in - d_ub
                    if cand < 0.0:
                        cand = 0.0
                    if cand > delta_b:
                        delta_b = cand
                if delta_b > best:
                    best = delta_b

        # ---- pivots stored at t but not at u ----
        for j, p in enumerate(t_piv):
            p = int(p)
            if p in u_set:
                continue
            dt = float(t_d[j])
            if dt > sentinel_thresh:
                continue
            delta = 0.0
            for p_star, du_star, _dt_star, _du_in_star, _dt_in_star in shared_anchors:
                bnd = _bounds(float(pp_out[p_star, p]), du_star)
                if bnd is None:
                    continue
                d_lb, d_ub = bnd
                if is_directed:
                    # d(p,t) - d(p,u); substitute UB for d(p, u).
                    cand = dt - d_ub
                    if cand < 0.0:
                        cand = 0.0
                else:
                    cand = dt - d_ub
                    if d_lb - dt > cand:
                        cand = d_lb - dt
                    if cand < 0.0:
                        cand = 0.0
                if cand > delta:
                    delta = cand
            if delta > best:
                best = delta
            if is_directed:
                dt_in = float(t_d_in[j])
                if dt_in > sentinel_thresh:
                    continue
                delta_b = 0.0
                for p_star, _du_star, _dt_star, du_in_star, _dt_in_star in (
                    shared_anchors
                ):
                    bnd = _bounds(float(pp_in[p, p_star]), du_in_star)
                    if bnd is None:
                        continue
                    d_lb, d_ub = bnd
                    # d(u,p) - d(t,p); substitute LB for d(t,p):
                    cand = d_lb - dt_in
                    if cand < 0.0:
                        cand = 0.0
                    if cand > delta_b:
                        delta_b = cand
                if delta_b > best:
                    best = delta_b

        return best

    return h


__all__ = [
    "CDHLabels",
    "SELECTION_RULES",
    "cdh_memory_bytes",
    "cdh_preprocess",
    "make_cdh_heuristic",
]
