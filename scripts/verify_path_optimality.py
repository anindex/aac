#!/usr/bin/env python
"""Audit result CSVs that log per-run path optimality.

Several experiment CSVs include an ``all_optimal`` boolean column recording
whether A*-without-reopenings, under the tested admissible heuristic, returned
the same path cost as Dijkstra on every query of that run. This script
scans all such CSVs under ``results/`` and asserts that every admissible-method
row has ``all_optimal == True`` (string ``True``/``true`` or boolean).

The main DIMACS/OSMnx multi-seed Wilcoxon, hybrid, and Pareto CSVs do not
carry this column (only admissibility-violation counts and expansion numbers
are logged); path-cost checks on those benchmarks were performed at run time
but not persisted. The paper's path-optimality claim in Section 5.5 is
therefore scoped to the five audited benchmarks below; for the remaining
benchmarks we rely on the architectural admissibility chain
(Theorem~\ref{thm:admissibility}) together with the zero admissibility-
violation counts reported in their CSVs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

AUDITED = [
    "results/ablation_selection/exp1_selection_strategy.csv",
    "results/ablation_selection/exp2_admissibility_robustness.csv",
    "results/coverage_aware/coverage_aware_results.csv",
    "results/query_distributions/query_mode_results.csv",
    "results/query_distributions/query_mode_summary.csv",
]

# Per-query DIMACS path-optimality CSVs (column: "optimal" per query row).
# Each file contains 100 queries on one DIMACS graph; rows are per-query.
PER_QUERY_DIMACS = [
    "results/dimacs/aac_NY.csv",
    "results/dimacs/aac_FLA.csv",
    "results/dimacs/aac_BAY.csv",
    "results/dimacs/aac_COL.csv",
]


def _parse_bool(x: str) -> bool | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    if s == "":
        return None
    return None


def audit(path: Path) -> tuple[int, int, list[int]]:
    """Return (n_rows, n_optimal, suboptimal_row_indices)."""
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    n_rows = len(rows)
    suboptimal: list[int] = []
    n_opt = 0
    for i, row in enumerate(rows, start=2):  # row 2 is the first data row
        val = _parse_bool(row.get("all_optimal", ""))
        if val is True:
            n_opt += 1
        elif val is False:
            suboptimal.append(i)
        # None -> column missing or blank; skip (does not count against total)
    return n_rows, n_opt, suboptimal


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    any_fail = False
    summary_rows = []
    for rel in AUDITED:
        p = repo / rel
        if not p.exists():
            print(f"[SKIP] {rel}: not found")
            continue
        n, n_opt, bad = audit(p)
        status = "OK" if not bad else "FAIL"
        if bad:
            any_fail = True
        if bad:
            bad_preview = bad[:5]
            more = "..." if len(bad) > 5 else ""
            detail = f"; suboptimal row(s): {bad_preview}{more}"
        else:
            detail = ""
        print(f"[{status}] {rel}: {n_opt}/{n} rows all_optimal=True{detail}")
        summary_rows.append((rel, n, n_opt, len(bad)))

    # Per-query DIMACS audit: column is "optimal" rather than "all_optimal".
    for rel in PER_QUERY_DIMACS:
        p = repo / rel
        if not p.exists():
            print(f"[SKIP] {rel}: not found")
            continue
        with p.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n = len(rows)
        n_opt = sum(1 for r in rows if _parse_bool(r.get("optimal", "")) is True)
        bad = [i for i, r in enumerate(rows, start=2)
               if _parse_bool(r.get("optimal", "")) is False]
        status = "OK" if not bad else "FAIL"
        if bad:
            any_fail = True
        print(f"[{status}] {rel}: {n_opt}/{n} per-query rows optimal=True")
        summary_rows.append((rel, n, n_opt, len(bad)))

    out = repo / "results" / "dimacs" / "path_optimality_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["csv_path", "n_rows", "n_all_optimal_true", "n_suboptimal"])
        for row in summary_rows:
            w.writerow(row)
    print(f"\nSummary written to {out.relative_to(repo)}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
