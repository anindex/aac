"""PyTorch 2.11+ compilation utilities for AAC hot paths.

Call ``compile_hot_paths()`` once before training to JIT-compile the
performance-critical kernels (Bellman-Ford relaxation, smooth scatter-min).
This is optional -- all code works without compilation, but compilation
provides significant speedups on GPU for large graphs.

Requires PyTorch >= 2.11.  On older versions, ``compile_hot_paths()``
is a no-op.
"""

from __future__ import annotations

import torch


def compile_hot_paths() -> None:
    """Compile performance-critical AAC functions with torch.compile.

    Compiles:
    - ``_bf_relax_step``: Bellman-Ford inner relaxation (scatter_reduce)
    - ``_smooth_scatter_min``: Smooth BF logsumexp aggregation

    Safe to call multiple times (idempotent). No-op if torch.compile
    is unavailable.
    """
    import aac.contextual.smooth_bf as smooth_bf_mod
    import aac.embeddings.sssp as sssp_mod

    if not hasattr(torch, "compile"):
        return

    # Only compile once
    if not getattr(sssp_mod._bf_relax_step, "_tgs_compiled", False):
        sssp_mod._bf_relax_step = torch.compile(
            sssp_mod._bf_relax_step, dynamic=True
        )
        sssp_mod._bf_relax_step._tgs_compiled = True  # type: ignore[attr-defined]

    if not getattr(smooth_bf_mod._smooth_scatter_min, "_tgs_compiled", False):
        smooth_bf_mod._smooth_scatter_min = torch.compile(
            smooth_bf_mod._smooth_scatter_min, dynamic=True
        )
        smooth_bf_mod._smooth_scatter_min._tgs_compiled = True  # type: ignore[attr-defined]
