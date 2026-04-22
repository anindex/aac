#!/usr/bin/env python
"""TOST equivalence test for the matched-memory non-road comparisons.

Reads ``results/hybrid_nonroad/matched_budget_hybrid.csv`` (per-seed paired
AAC vs. ALT reduction-pct rows for SBM, BA, and OGB-arXiv) and computes the
two one-sided tests (TOST) at equivalence margin delta=1 pp, alpha=0.05,
following Lakens (2017). Writes a tidy summary CSV that backs the paper's
TOST claims (e.g., "TOST accepts equivalence within delta at B=128 only" on
SBM/BA in Section 5.9; "no cell achieves TOST equivalence at delta=1 pp" on
OGB-arXiv in Section 5.9.3).

For each (graph_type, total_budget_B) cell the script computes:
    n_seeds          - number of paired observations.
    mean_diff        - mean of (AAC - ALT) reduction-pct across seeds.
    sd_diff          - sample standard deviation of the paired differences.
    se_diff          - standard error.
    delta            - equivalence margin (1 pp).
    t_lower / p_lower - one-sided test that mean_diff > -delta.
    t_upper / p_upper - one-sided test that mean_diff < +delta.
    p_tost           - max(p_lower, p_upper).
    tost_accepts     - True iff p_tost < alpha (=0.05).

Output: ``results/synthetic/tost_equivalence.csv``
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT = PROJECT_ROOT / "results" / "hybrid_nonroad" / "matched_budget_hybrid.csv"
OUTPUT = PROJECT_ROOT / "results" / "synthetic" / "tost_equivalence.csv"

DELTA = 1.0  # equivalence margin in percentage points
ALPHA = 0.05


def _student_t_sf(t: float, df: int) -> float:
    """Survival function of Student's t (two arguments). Tries scipy then
    falls back to a numerical approximation of the regularised incomplete
    beta function (Abramowitz-Stegun 26.5.27)."""
    try:
        from scipy.stats import t as _t  # type: ignore
        return float(_t.sf(t, df))
    except Exception:
        # Fallback: t-distribution survival via the incomplete beta function.
        # P(T > t) = 0.5 * I_{df/(df + t^2)}(df/2, 0.5) for t >= 0.
        x = df / (df + t * t)
        return 0.5 * _betainc_reg(df / 2.0, 0.5, x) if t >= 0 else 1.0 - 0.5 * _betainc_reg(df / 2.0, 0.5, df / (df + t * t))


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a,b) via continued fraction
    (Numerical Recipes-style). Pure-Python fallback used only if scipy is
    unavailable; not optimised."""
    if x < 0 or x > 1:
        raise ValueError("x must be in [0, 1]")
    if x == 0 or x == 1:
        return x
    log_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def main() -> int:
    cells: dict[tuple[str, int], list[float]] = defaultdict(list)
    with INPUT.open() as f:
        for row in csv.DictReader(f):
            graph = row["graph_type"]
            budget = int(row["total_budget_B"])
            aac_val = float(row["pure_aac_reduction_pct"])
            alt = float(row["pure_alt_reduction_pct"])
            cells[(graph, budget)].append(aac_val - alt)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "graph_type", "total_budget_B", "n_seeds",
        "mean_diff_pp", "sd_diff_pp", "se_diff_pp",
        "delta_pp", "alpha",
        "t_lower", "p_lower", "t_upper", "p_upper",
        "p_tost", "tost_accepts",
    ]
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (graph, budget) in sorted(cells.keys()):
            diffs = cells[(graph, budget)]
            n = len(diffs)
            if n < 2:
                continue
            mean = sum(diffs) / n
            sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
            se = sd / math.sqrt(n) if sd > 0 else float("nan")
            df = n - 1
            # Two one-sided tests: H0_lower: mean <= -delta vs H1: mean > -delta
            #                      H0_upper: mean >= +delta vs H1: mean < +delta
            if se > 0 and not math.isnan(se):
                t_lower = (mean - (-DELTA)) / se
                t_upper = (mean - (+DELTA)) / se
                p_lower = _student_t_sf(t_lower, df)         # P(T > t_lower)
                p_upper = _student_t_sf(-t_upper, df)        # P(T < t_upper) = P(T > -t_upper)
            else:
                t_lower = t_upper = float("nan")
                p_lower = p_upper = float("nan")
            p_tost = max(p_lower, p_upper) if not math.isnan(p_lower) else float("nan")
            accepts = (not math.isnan(p_tost)) and p_tost < ALPHA
            writer.writerow({
                "graph_type": graph,
                "total_budget_B": budget,
                "n_seeds": n,
                "mean_diff_pp": f"{mean:.4f}",
                "sd_diff_pp": f"{sd:.4f}",
                "se_diff_pp": f"{se:.4f}",
                "delta_pp": f"{DELTA:.2f}",
                "alpha": f"{ALPHA:.2f}",
                "t_lower": f"{t_lower:.4f}",
                "p_lower": f"{p_lower:.6f}",
                "t_upper": f"{t_upper:.4f}",
                "p_upper": f"{p_upper:.6f}",
                "p_tost": f"{p_tost:.6f}",
                "tost_accepts": str(accepts),
            })
    print(f"Wrote {OUTPUT}")
    print()
    with OUTPUT.open() as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
