"""Baseline heuristic methods for comparison with AAC.

Provides:
- ALT: Landmark-based triangle inequality heuristic (A=I special case of AAC)
- FastMap: Iterative farthest-pair Euclidean embedding with L1 heuristic
- PHIL: Reported numbers from Pandy et al. LoG 2022 (no public code)

# DataSP baseline for contextual comparison
"""

from aac.baselines.alt import alt_memory_bytes, alt_preprocess, make_alt_heuristic
from aac.baselines.cdh import (
    CDHLabels,
    cdh_memory_bytes,
    cdh_preprocess,
    make_cdh_heuristic,
)
from aac.baselines.fastmap import (
    fastmap_memory_bytes,
    fastmap_preprocess,
    make_fastmap_heuristic,
)

PHIL_REPORTED = {
    "modena": {
        "expansions_reduction_vs_dijkstra": None,
        "note": "Extract from Pandy et al. LoG 2022 Table",
    },
    "new_york": {
        "expansions_reduction_vs_dijkstra": None,
        "note": "Extract from Pandy et al. LoG 2022 Table",
    },
    "caveat": "No public code available; numbers reported from paper only",
}

__all__ = [
    "alt_preprocess",
    "make_alt_heuristic",
    "alt_memory_bytes",
    "cdh_preprocess",
    "make_cdh_heuristic",
    "cdh_memory_bytes",
    "CDHLabels",
    "fastmap_preprocess",
    "make_fastmap_heuristic",
    "fastmap_memory_bytes",
    "PHIL_REPORTED",
]
