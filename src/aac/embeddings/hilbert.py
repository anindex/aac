"""Hilbert simplex embedding for undirected graphs.

Maps vertices to the positive orthant of R^{2K} via log-coordinates:
    phi(v) = 1/2 * (d(v,l_1),...,d(v,l_K), -d(v,l_1),...,-d(v,l_K))

The Hilbert projective distance in log-coordinates equals the variation norm:
    rho_H(Psi(u), Psi(v)) = max(phi(u) - phi(v)) - min(phi(u) - phi(v))

With K = V (all vertices as anchors), rho_H = d_G (graph distance).
With K < V, this provides the standard ALT lower bound.

IMPORTANT: We stay in log-coordinates throughout. Never compute exp(phi)
as DIMACS distances reach 100K+ and exp(50000) overflows fp64.
"""

from __future__ import annotations

import torch

from aac.graphs.types import Embedding, TeacherLabels


def build_hilbert_embedding(teacher_labels: TeacherLabels) -> Embedding:
    """Build Hilbert simplex embedding from teacher labels.

    Construction for UNDIRECTED graphs only.
    For undirected graphs, d_out == d_in so we use d_out only.

    Args:
        teacher_labels: TeacherLabels with d_out (K, V) from undirected graph.

    Returns:
        Embedding with phi (V, 2K), kind="hilbert", is_directed=False.

    Raises:
        AssertionError: If teacher_labels is from a directed graph.
    """
    assert not teacher_labels.is_directed, (
        "Hilbert embedding requires undirected graph (use build_tropical_embedding for directed)"
    )

    D = teacher_labels.d_out  # (K, V) -- for undirected, d_out == d_in
    D_t = D.t()  # (V, K)

    # phi(v) = 1/2 * [d(v,l_1),...,d(v,l_K), -d(v,l_1),...,-d(v,l_K)]
    phi = 0.5 * torch.cat([D_t, -D_t], dim=1)  # (V, 2K)

    return Embedding(
        phi=phi,
        kind="hilbert",
        is_directed=False,
        num_anchors=D.shape[0],
    )
