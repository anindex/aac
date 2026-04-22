"""Numerical utilities and common types."""

from aac.utils.memory import (
    estimate_teacher_label_memory,
    get_available_memory_bytes,
    memory_guard,
)
from aac.utils.numerics import (
    LOG_SENTINEL,
    SENTINEL,
    is_sentinel,
    safe_exp,
    safe_log,
    shifted_softmin,
)

__all__ = [
    "SENTINEL",
    "LOG_SENTINEL",
    "shifted_softmin",
    "safe_log",
    "safe_exp",
    "is_sentinel",
    "get_available_memory_bytes",
    "estimate_teacher_label_memory",
    "memory_guard",
]
