#!/usr/bin/env python
"""Closed-form toy: gap-to-teacher vs covering radius on the path graph P_7.

Enumerates all 2-element landmark subsets of V = {0, ..., 6} on the unit-weight
path graph P_7. For each subset S = {l1, l2} computes:

  * covering radius      r_2(S) = max_{v in V} min_{l in S} |v - l|
  * expected ALT gap     E_q[d(s,t) - h_ALT^S(s,t)]   over the uniform query
                         distribution Q on { (s,t) : s,t in {1..5}, s != t }
                         (20 ordered pairs)

with h_ALT^S(s,t) = max_{l in S} |d(s,l) - d(l,t)| and d(i,j) = |i - j|.

The two highlighted subsets are the symmetric, canonical representatives of
their respective optima:

  S_cov = {2, 4}  (symmetric k-center / minimax optimum)   r_2 = 2, E[gap] = 1/5
  S_gap = {0, 6}  (symmetric peripheral / gap optimum)     r_2 = 3, E[gap] = 0

Only the two longest queries (1,5) and (5,1) straddle both landmarks of S_cov,
each contributing a gap of 2; all 18 other queries are exact. Hence E[gap] =
4 / 20 = 1/5.

Multiple other subsets tie on each metric: every subset containing vertex 0
or vertex 6 has at least one landmark strictly outside any interior query
interval, hence achieves E[gap] = 0; and several subsets ({0,4}, {1,4},
{2,4}, {2,5}, {2,6}) achieve r_2 = 2. The canonical contrast above selects
the *symmetric* representative of each class.

The script writes ``results/toy_p7/all_subsets.csv`` and
``results/toy_p7/highlight.csv`` and prints a 2x2 fact table.

Used by ``scripts/plot_toy_p7_divergence.py`` to render Figure (toy-p7) in
paper/main.tex Section 3.5.
"""

from __future__ import annotations

import csv
import itertools
import os
import sys
from fractions import Fraction
from typing import Iterable, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "toy_p7")
ALL_SUBSETS_CSV = os.path.join(OUTPUT_DIR, "all_subsets.csv")
HIGHLIGHT_CSV = os.path.join(OUTPUT_DIR, "highlight.csv")

# Toy specification.
N_VERTICES = 7                       # P_7: vertices 0..6
LANDMARK_BUDGET = 2                  # m = 2
QUERY_SUPPORT = tuple(range(1, 6))   # interior vertices {1, 2, 3, 4, 5}


def covering_radius(S: Tuple[int, ...]) -> int:
    """max_{v in V} min_{l in S} |v - l|  on P_7 with unit edges."""
    return max(min(abs(v - l) for l in S) for v in range(N_VERTICES))


def alt_heuristic(S: Tuple[int, ...], s: int, t: int) -> int:
    """h_ALT^S(s, t) = max_{l in S} |d(s, l) - d(l, t)|  on P_7."""
    return max(abs(abs(s - l) - abs(l - t)) for l in S)


def query_pairs() -> Iterable[Tuple[int, int]]:
    """Uniform support over ordered pairs in QUERY_SUPPORT^2 \\ diag."""
    for s in QUERY_SUPPORT:
        for t in QUERY_SUPPORT:
            if s != t:
                yield s, t


def expected_gap(S: Tuple[int, ...]) -> Tuple[Fraction, int]:
    """Returns (E[gap] as Fraction, count of queries with gap == 0)."""
    pairs = list(query_pairs())
    total = Fraction(0)
    n_exact = 0
    for s, t in pairs:
        gap = abs(s - t) - alt_heuristic(S, s, t)
        if gap == 0:
            n_exact += 1
        total += Fraction(gap)
    return total / len(pairs), n_exact


def enumerate_all_subsets() -> list[dict]:
    rows = []
    for l1, l2 in itertools.combinations(range(N_VERTICES), LANDMARK_BUDGET):
        S = (l1, l2)
        r_cov = covering_radius(S)
        e_gap, n_exact = expected_gap(S)
        rows.append({
            "l1": l1,
            "l2": l2,
            "r_cov": r_cov,
            "exp_gap": float(e_gap),
            "exp_gap_num": e_gap.numerator,
            "exp_gap_den": e_gap.denominator,
            "n_exact_queries": n_exact,
        })
    return rows


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    rows = enumerate_all_subsets()
    write_csv(
        ALL_SUBSETS_CSV,
        rows,
        ["l1", "l2", "r_cov", "exp_gap",
         "exp_gap_num", "exp_gap_den", "n_exact_queries"],
    )

    # Identify highlighted subsets.
    # We use the symmetric, canonical representative of each optimum class
    # (multiple subsets tie on each metric; see module docstring).
    S_COV = (2, 4)  # symmetric k-center optimum
    S_GAP = (0, 6)  # symmetric peripheral / gap optimum
    by_subset = {(r["l1"], r["l2"]): r for r in rows}
    s_cov_row = by_subset[S_COV]
    s_gap_row = by_subset[S_GAP]
    s_cov, s_gap = S_COV, S_GAP

    # Sanity: the canonical representatives exhibit strict divergence in
    # both directions (r_cov favors S_cov, E[gap] favors S_gap).
    assert s_cov_row["r_cov"] < s_gap_row["r_cov"], (
        f"r_cov({s_cov})={s_cov_row['r_cov']} not strictly less than "
        f"r_cov({s_gap})={s_gap_row['r_cov']}."
    )
    assert s_gap_row["exp_gap"] < s_cov_row["exp_gap"], (
        f"exp_gap({s_gap})={s_gap_row['exp_gap']} not strictly less than "
        f"exp_gap({s_cov})={s_cov_row['exp_gap']}."
    )

    highlight_rows = [
        {"name": "S_cov", **s_cov_row},
        {"name": "S_gap", **s_gap_row},
    ]
    write_csv(
        HIGHLIGHT_CSV,
        highlight_rows,
        ["name", "l1", "l2", "r_cov", "exp_gap",
         "exp_gap_num", "exp_gap_den", "n_exact_queries"],
    )

    # 2x2 fact table to stdout.
    print("Toy P_7 closed-form: gap-to-teacher vs covering radius")
    print("=" * 64)
    print(f"  Graph: path P_{N_VERTICES} (vertices 0..{N_VERTICES - 1}, "
          f"unit edges); m = {LANDMARK_BUDGET}")
    print(f"  Query distribution: uniform on {QUERY_SUPPORT}^2 \\ diag "
          f"({len(list(query_pairs()))} ordered pairs)")
    print()
    print(f"  {'Subset':<14} {'r_cov':>6} {'E[gap]':>10} {'#exact / 20':>14}")
    print(f"  {'-'*14} {'-'*6} {'-'*10} {'-'*14}")
    for label, row in (("S_cov = {%d,%d}" % s_cov, s_cov_row),
                       ("S_gap = {%d,%d}" % s_gap, s_gap_row)):
        print(f"  {label:<14} {row['r_cov']:>6} "
              f"{row['exp_gap']:>10.4f} {row['n_exact_queries']:>14}")
    print()
    print(f"  Wrote {os.path.relpath(ALL_SUBSETS_CSV, PROJECT_ROOT)} "
          f"({len(rows)} subsets)")
    print(f"  Wrote {os.path.relpath(HIGHLIGHT_CSV, PROJECT_ROOT)} "
          f"(2 highlighted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
