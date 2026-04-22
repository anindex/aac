"""Tropical/Funk embedding for directed graphs.

Maps vertices to R^{2K} via forward and reverse distance labels:
    Phi(v) = (d(v, l_1),...,d(v, l_K), -d(l_1, v),...,-d(l_K, v))

where d(v, l_j) = distance from v to anchor j (teacher_labels.d_in[j, v]),
and d(l_j, v) = distance from anchor j to v (teacher_labels.d_out[j, v]).

Note the naming convention: compute_teacher_labels defines
  d_out[k, v] = d(anchor_k, v) -- forward SSSP FROM anchor
  d_in[k, v]  = d(v, anchor_k) -- reverse SSSP TO anchor

The Funk/tropical distance:
    delta(Phi(u), Phi(t)) = max_i(Phi_i(u) - Phi_i(t)) = d_G(u, t)

With K = V (all vertices as anchors), this is exact.
With K < V, it provides the standard ALT lower bound for directed graphs.
"""

from __future__ import annotations

import torch

from aac.graphs.types import Embedding, TeacherLabels


def build_tropical_embedding(teacher_labels: TeacherLabels) -> Embedding:
    """Build tropical/Funk embedding from teacher labels.

    Construction for DIRECTED graphs. Can also be used on
    undirected graphs but the Hilbert construction is more natural.

    Uses the tropical embedding construction:
        Phi(v) = (d(v, l_1),...,d(v, l_K), -d(l_1, v),...,-d(l_K, v))

    In our convention:
        d(v, l_j) = teacher_labels.d_in[j, v]   (reverse: v -> anchor)
        d(l_j, v) = teacher_labels.d_out[j, v]   (forward: anchor -> v)

    Args:
        teacher_labels: TeacherLabels with d_out (K, V) and d_in (K, V).

    Returns:
        Embedding with phi (V, 2K), kind="tropical", is_directed=True.
    """
    # d(v, anchor_j) = teacher_labels.d_in[j, v] -- distance from v to anchor
    d_v_to_anchor = teacher_labels.d_in.t()  # (V, K)
    # d(anchor_j, v) = teacher_labels.d_out[j, v] -- distance from anchor to v
    d_anchor_to_v = teacher_labels.d_out.t()  # (V, K)

    # Phi(v) = [d(v, l_1),...,d(v, l_K), -d(l_1, v),...,-d(l_K, v)]
    phi = torch.cat([d_v_to_anchor, -d_anchor_to_v], dim=1)  # (V, 2K)

    return Embedding(
        phi=phi,
        kind="tropical",
        is_directed=teacher_labels.is_directed,
        num_anchors=d_v_to_anchor.shape[1],
    )
