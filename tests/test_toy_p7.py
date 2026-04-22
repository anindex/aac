"""Pin the four key numbers of the P_7 toy example.

Verifies the closed-form computation in
``scripts/run_toy_p7_gap_vs_covering.py`` via independent rational arithmetic.
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from run_toy_p7_gap_vs_covering import (  # noqa: E402
    covering_radius,
    expected_gap,
)


def test_covering_radius_s_cov():
    assert covering_radius((2, 4)) == 2


def test_covering_radius_s_gap():
    assert covering_radius((0, 6)) == 3


def test_expected_gap_s_cov_is_one_fifth():
    e_gap, n_exact = expected_gap((2, 4))
    assert e_gap == Fraction(1, 5)
    assert n_exact == 18


def test_expected_gap_s_gap_is_zero():
    e_gap, n_exact = expected_gap((0, 6))
    assert e_gap == Fraction(0)
    assert n_exact == 20


def test_divergence_is_strict_in_both_directions():
    r_cov_s_cov = covering_radius((2, 4))
    r_cov_s_gap = covering_radius((0, 6))
    e_gap_s_cov, _ = expected_gap((2, 4))
    e_gap_s_gap, _ = expected_gap((0, 6))
    assert r_cov_s_cov < r_cov_s_gap, "covering radius favors S_cov"
    assert e_gap_s_gap < e_gap_s_cov, "expected gap favors S_gap"
