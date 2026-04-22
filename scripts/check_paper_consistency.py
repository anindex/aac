#!/usr/bin/env python3
"""Numerical drift detector between paper/table_*.tex and source CSVs.

Each ``paper/table_*.tex`` carries a machine-readable provenance header:

    %%% PAPER-TABLE-PROVENANCE
    %%% generator:  hand-maintained        (or scripts/foo.py for auto-generated)
    %%% sources:    results/dimacs/foo.csv
    %%%             results/dimacs/bar.csv
    %%% paper-ref:  Table tab:foo; Sec X.Y
    %%% verify:     python scripts/check_paper_consistency.py paper/table_foo.tex
    %%% END-PROVENANCE

This script parses every header, tokenizes the numeric cells of the table body
(i.e. content inside ``\\begin{tabular}...\\end{tabular}``, skipping captions /
labels / spec markup), and verifies each cell value can be located in at least
one source CSV (string or rounded match, with scientific-notation conversion).

Output:
- PASS:   every numeric cell appears in a source CSV.
- WARN:   one or more cells not found in any source CSV. Common causes:
          aggregated values (Fisher / Stouffer / Wilcoxon combination p-values),
          derived metrics (gap, ratio, percentage point delta), or values
          formatted with rounding finer than CSV precision. Listed for manual
          inspection; does NOT block the build.
- FAIL:   provenance header missing or malformed, OR source CSV missing on disk.

Exit codes:
    0 = all PASS or PASS+WARN (CSV ground-truth fully present).
    2 = one or more FAIL conditions (missing header, missing CSV).

Usage:
    python scripts/check_paper_consistency.py                  # all 20 tables
    python scripts/check_paper_consistency.py paper/table_X.tex  # one table
    python scripts/check_paper_consistency.py --strict         # WARN -> exit 1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Provenance header parsing
# ---------------------------------------------------------------------------

HEADER_START = "%%% PAPER-TABLE-PROVENANCE"
HEADER_END = "%%% END-PROVENANCE"
FIELD_RE = re.compile(r"^%%%\s*(generator|sources|paper-ref|verify):\s*(.*)$")
CONT_RE = re.compile(r"^%%%\s+(\S.*)$")


class Provenance(NamedTuple):
    generator: str
    sources: list[Path]
    paper_ref: str


def parse_provenance(tex_path: Path) -> Provenance | None:
    """Parse the %%% PAPER-TABLE-PROVENANCE block. Returns None if missing."""
    text = tex_path.read_text()
    if HEADER_START not in text or HEADER_END not in text:
        return None
    block = text.split(HEADER_START, 1)[1].split(HEADER_END, 1)[0]
    fields: dict[str, list[str]] = {}
    current_key: str | None = None
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line.startswith("%%%"):
            continue
        match = FIELD_RE.match(line)
        if match:
            current_key = match.group(1)
            fields.setdefault(current_key, []).append(match.group(2).strip())
            continue
        cont = CONT_RE.match(line)
        if cont and current_key is not None:
            fields[current_key].append(cont.group(1).strip())
    if "generator" not in fields or "sources" not in fields:
        return None
    sources = [PROJECT_ROOT / s for s in fields["sources"] if s]
    return Provenance(
        generator=" ".join(fields["generator"]),
        sources=sources,
        paper_ref=" ".join(fields.get("paper-ref", [""])),
    )


# ---------------------------------------------------------------------------
# Table body tokenisation
# ---------------------------------------------------------------------------

TABULAR_RE = re.compile(
    r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}",
    re.DOTALL,
)
# Numeric token: optional sign, integer or decimal, optional scientific suffix.
NUM_RE = re.compile(
    r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])"
)
# LaTeX scientific notation: 1.3 \times 10^{-3}  or  $1.3\!\times\!10^{-3}$
SCI_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:\\!)?\\times(?:\\!)?\s*10\^\{?(-?\d+)\}?"
)
# Tokens to strip wholesale before tokenising.
STRIP_PATTERNS = [
    re.compile(r"\\label\{[^}]*\}"),
    re.compile(r"\\ref\{[^}]*\}"),
    re.compile(r"\\cite[ptp]?\{[^}]*\}"),
    re.compile(r"\\input\{[^}]*\}"),
    re.compile(r"\\rowcolor\{[^}]*\}"),
    re.compile(r"\\textcolor\{[^}]*\}\{[^}]*\}"),
    re.compile(r"\\cellcolor\{[^}]*\}"),
    re.compile(r"\\multirow\{[^}]*\}\{[^}]*\}"),
    re.compile(r"\\multicolumn\{[^}]*\}\{[^}]*\}"),
    re.compile(r"\\resizebox\{[^}]*\}\{[^}]*\}"),
    # Dimensions (e.g. 0.5pt, 12em) are skipped via \w boundary in NUM_RE.
]
# Skip parameter assignments like K_0{=}64, $m{=}16$, B{=}32, $\alpha_c{=}1$
PARAM_ASSIGN_RE = re.compile(r"[A-Za-z_\\][A-Za-z_0-9\\{}\\^]*\s*[{=]?=[}=]?\s*-?\d+(?:\.\d+)?")
# Skip braces around digits used purely for thousands separators: 169{,}343 -> 169343
# The braces must be escaped (literal `{` and `}`); unescaped `{,}` would be
# interpreted as the regex quantifier `{m,n}` and silently match zero characters.
THOUSANDS_BRACE_RE = re.compile(r"(\d)\{,\}(\d)")
THOUSANDS_COMMA_RE = re.compile(r"(\d),(\d)")


def normalise_thousands(s: str) -> str:
    while True:
        new = THOUSANDS_BRACE_RE.sub(r"\1\2", s)
        new = THOUSANDS_COMMA_RE.sub(r"\1\2", new)
        if new == s:
            return s
        s = new


def extract_data_cells(tex_path: Path) -> list[tuple[int, str]]:
    """Return [(line_no, raw_cell_content)] for every data cell in the table.

    Data cells = content inside \\begin{tabular}...\\end{tabular}, split first
    by \\\\ then by &; column-spec arg of \\begin{tabular}{...} is skipped.
    """
    text = tex_path.read_text()
    body_blocks = TABULAR_RE.findall(text)
    cells: list[tuple[int, str]] = []
    line_offset = 0
    for block in body_blocks:
        # Skip the \begin{tabular}{...} header itself; tabular column spec
        # is the first {...} after \begin{tabular}.
        body = re.sub(r"\\begin\{tabular\*?\}\{[^}]*\}", "", block, count=1)
        body = re.sub(r"\\end\{tabular\*?\}", "", body, count=1)
        # Drop \toprule / \midrule / \bottomrule / \cmidrule / \hline.
        body = re.sub(
            r"\\(?:top|mid|bottom|cmidrule)rule\b(?:\([^)]*\))?(?:\{[^}]*\})*",
            "",
            body,
        )
        body = re.sub(r"\\hline", "", body)
        # Drop strip patterns.
        for pat in STRIP_PATTERNS:
            body = pat.sub("", body)
        # Split rows by \\, cells by &.
        for row in body.split("\\\\"):
            for cell in row.split("&"):
                cell = cell.strip()
                if cell:
                    cells.append((line_offset, cell))
        line_offset += 1
    return cells


def tokens_for_csv(cell: str) -> list[str]:
    """Return numeric tokens to look up in CSVs.

    Converts LaTeX scientific notation (1.3 \\times 10^{-3}) to canonical
    (1.3e-3); strips thousands braces; skips parameter assignments.
    """
    cell = normalise_thousands(cell)
    out: list[str] = []
    # First: scientific notation.
    for sci_match in SCI_RE.finditer(cell):
        mantissa, exp = sci_match.group(1), sci_match.group(2)
        out.append(f"{mantissa}e{exp}")
        out.append(f"{mantissa}e+{exp}" if not exp.startswith("-") else f"{mantissa}e{exp}")
        try:
            out.append(f"{float(mantissa) * (10 ** int(exp)):.6g}")
        except (ValueError, OverflowError):
            pass
    # Strip the scientific-notation matches so we don't re-tokenise the digits.
    cell_stripped = SCI_RE.sub("", cell)
    # Strip parameter assignments (K_0=64, m=16, etc.) so the RHS isn't taken
    # as a data value. We strip the whole "name=value" substring.
    cell_stripped = PARAM_ASSIGN_RE.sub("", cell_stripped)
    out.extend(NUM_RE.findall(cell_stripped))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    return [t for t in out if not (t in seen or seen.add(t))]


# ---------------------------------------------------------------------------
# CSV side: load every numeric cell as a normalised string set
# ---------------------------------------------------------------------------


def load_csv_strings(csv_path: Path) -> tuple[set[str], list[float]]:
    """Return (string-set, float-list) of every cell in csv_path.

    The string-set is for fast string match (handles non-numeric IDs too).
    The float-list is for tolerance-based numeric match.
    """
    strings: set[str] = set()
    floats: list[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                cell = cell.strip()
                if not cell:
                    continue
                strings.add(cell)
                # Also store rounded variants commonly used in tables.
                try:
                    val = float(cell)
                    floats.append(val)
                except ValueError:
                    continue
    return strings, floats


def load_json_strings(json_path: Path) -> tuple[set[str], list[float]]:
    """Recursively flatten every numeric leaf in a JSON document."""
    strings: set[str] = set()
    floats: list[float] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, (int, float)):
            strings.add(repr(node))
            floats.append(float(node))
        elif isinstance(node, str):
            strings.add(node)
            try:
                floats.append(float(node))
            except ValueError:
                pass

    try:
        walk(json.loads(json_path.read_text()))
    except (json.JSONDecodeError, OSError):
        pass
    return strings, floats


def value_matches(token: str, strings: set[str], floats: list[float]) -> bool:
    """Return True if token appears in CSV (string) or matches a CSV float."""
    if token in strings:
        return True
    try:
        val = float(token)
    except ValueError:
        return False
    # Exact float in CSV?
    for f_val in floats:
        if val == f_val:
            return True
    # Rounding tolerance: scale-relative epsilon plus per-decimal-place band.
    # e.g. table cell "89.13" should match CSV "89.131234" or "89.125".
    if "." in token:
        decimals = len(token.split(".", 1)[1].rstrip("0"))
    else:
        decimals = 0
    tol = max(5 * 10 ** (-decimals - 1), abs(val) * 1e-4)
    for f_val in floats:
        if abs(f_val - val) <= tol:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-table check
# ---------------------------------------------------------------------------


class TableResult(NamedTuple):
    table: Path
    status: str  # PASS / WARN / FAIL
    found: int
    warned: int
    missing_sources: list[Path]
    warn_samples: list[tuple[str, str]]  # (cell, token)


def check_table(tex_path: Path, max_warn_samples: int = 5) -> TableResult:
    prov = parse_provenance(tex_path)
    if prov is None:
        return TableResult(tex_path, "FAIL", 0, 0, [], [])
    missing = [p for p in prov.sources if not p.exists()]
    if missing:
        return TableResult(tex_path, "FAIL", 0, 0, missing, [])
    # Aggregate the union of every source's content.
    all_strings: set[str] = set()
    all_floats: list[float] = []
    for src in prov.sources:
        if src.suffix == ".json":
            s, f = load_json_strings(src)
        else:
            s, f = load_csv_strings(src)
        all_strings.update(s)
        all_floats.extend(f)
    cells = extract_data_cells(tex_path)
    found = 0
    warned = 0
    warn_samples: list[tuple[str, str]] = []
    for _line, cell in cells:
        for token in tokens_for_csv(cell):
            if value_matches(token, all_strings, all_floats):
                found += 1
            else:
                warned += 1
                if len(warn_samples) < max_warn_samples:
                    warn_samples.append((cell.strip()[:80], token))
    if warned == 0:
        return TableResult(tex_path, "PASS", found, 0, [], [])
    return TableResult(tex_path, "WARN", found, warned, [], warn_samples)


# ---------------------------------------------------------------------------
# Unescaped-% guard for matplotlib label/title strings
# ---------------------------------------------------------------------------
#
# ``setup_style()`` enables ``text.usetex=True`` globally. With LaTeX rendering
# active, a literal ``%`` in a label or title string is interpreted as the
# start of a comment and silently truncates the rest of the string. This guard
# greps the plotting code for that footgun and fails CI if any bare ``%``
# appears inside a label / title / annotation string. The fix is always to
# escape it as ``\%``.

_LABEL_CALL_RE = re.compile(
    r"\b(?:set_xlabel|set_ylabel|set_title|suptitle|annotate|"
    r"(?:ax|fig|axes\[[^\]]*\])\.text)\s*\("
)
_STRING_LITERAL_RE = re.compile(
    r"""(?P<prefix>[fFrRbB]{0,2})(?P<quote>'''|\"\"\"|'|\")(?P<body>.*?)(?P=quote)""",
    re.DOTALL,
)
_UNESCAPED_PCT_RE = re.compile(r"(?<!\\)%")
_PLOT_SCAN_DIRS = (
    Path("scripts"),
    Path("src") / "aac" / "viz",
    Path("src") / "experiments" / "reporting",
)


def _scan_unescaped_percent(py_path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_number, raw_line, offending_string) hits.

    Catches both single-line and multi-line label-call patterns.  A
    multi-line call like::

        ax.text(
            0.5, 0.5,
            f"value = {x:.2f}%",   # offending unescaped '%'
        )

    is detected by tracking the parenthesis depth across lines: once a
    matching ``\\b(set_xlabel|...)\\s*\\(`` opens, every string literal
    encountered before the closing parenthesis is scanned for unescaped
    ``%``.
    """
    hits: list[tuple[int, str, str]] = []
    try:
        text = py_path.read_text(encoding="utf-8")
    except OSError:
        return hits
    lines = text.splitlines()
    in_call = False
    paren_depth = 0
    for lineno, line in enumerate(lines, start=1):
        if not in_call and _LABEL_CALL_RE.search(line):
            in_call = True
            paren_depth = 0
        if in_call:
            paren_depth += line.count("(") - line.count(")")
            for m in _STRING_LITERAL_RE.finditer(line):
                body = m.group("body")
                if _UNESCAPED_PCT_RE.search(body):
                    hits.append((lineno, line.rstrip(), body))
            if paren_depth <= 0:
                in_call = False
    return hits


def check_unescaped_percent_in_plot_scripts() -> list[tuple[Path, int, str, str]]:
    """Scan plotting code for unescaped ``%`` in matplotlib label strings.

    The guard itself contains illustrative ``%`` examples in its docstring,
    so this script is excluded from the scan.
    """
    self_path = Path(__file__).resolve()
    violations: list[tuple[Path, int, str, str]] = []
    for rel_dir in _PLOT_SCAN_DIRS:
        directory = PROJECT_ROOT / rel_dir
        if not directory.is_dir():
            continue
        for py_path in sorted(directory.glob("*.py")):
            if py_path.resolve() == self_path:
                continue
            for lineno, raw_line, body in _scan_unescaped_percent(py_path):
                violations.append((py_path, lineno, raw_line, body))
    return violations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify paper/table_*.tex cells against source CSVs."
    )
    parser.add_argument(
        "tables",
        nargs="*",
        help="Specific table_*.tex files to check (default: all).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat WARN as failure (exit 1).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-warn samples; print summary only.",
    )
    args = parser.parse_args()

    if args.tables:
        tables = [Path(t).resolve() for t in args.tables]
    else:
        tables = sorted((PROJECT_ROOT / "paper").glob("table_*.tex"))

    results: list[TableResult] = [check_table(t) for t in tables]

    fail_count = sum(1 for r in results if r.status == "FAIL")
    warn_count = sum(1 for r in results if r.status == "WARN")
    pass_count = sum(1 for r in results if r.status == "PASS")
    total_found = sum(r.found for r in results)
    total_warned = sum(r.warned for r in results)

    print()
    print("=" * 72)
    print("Paper / CSV consistency check")
    print("=" * 72)
    for r in results:
        rel = r.table.relative_to(PROJECT_ROOT)
        if r.status == "FAIL":
            if r.missing_sources:
                src_list = ", ".join(
                    str(p.relative_to(PROJECT_ROOT)) for p in r.missing_sources
                )
                print(f"  FAIL  {rel}: missing source CSV(s): {src_list}")
            else:
                print(f"  FAIL  {rel}: missing or malformed PROVENANCE header")
        elif r.status == "WARN":
            print(
                f"  WARN  {rel}: {r.found} cells matched, "
                f"{r.warned} not found in source CSVs"
            )
            if not args.quiet:
                for cell, token in r.warn_samples:
                    print(f"          token {token!r:>12}  in cell: {cell}")
                if r.warned > len(r.warn_samples):
                    print(
                        f"          ... and {r.warned - len(r.warn_samples)} more"
                    )
        else:
            print(f"  PASS  {rel}: {r.found} cells matched")
    print("-" * 72)
    print(
        f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL "
        f"({total_found} cells matched, {total_warned} warned)"
    )
    print("=" * 72)
    print()

    pct_violations = check_unescaped_percent_in_plot_scripts()
    print()
    print("=" * 72)
    print("Plot-script guard: unescaped '%' in matplotlib label strings")
    print("=" * 72)
    if pct_violations:
        for py_path, lineno, raw_line, body in pct_violations:
            rel = py_path.relative_to(PROJECT_ROOT)
            print(f"  FAIL  {rel}:{lineno}: {raw_line.strip()}")
            print(f"          offending string: {body!r} (escape '%' as '\\%')")
        print("-" * 72)
        print(
            f"Summary: {len(pct_violations)} unescaped '%' in plot scripts "
            "(text.usetex=True silently truncates labels at '%')."
        )
    else:
        print("  PASS: no unescaped '%' found.")
    print("=" * 72)
    print()

    if fail_count:
        print("FAIL: missing provenance headers or source CSVs (see above).")
        return 2
    if pct_violations:
        print("FAIL: unescaped '%' detected in plot label strings (see above).")
        return 2
    if args.strict and warn_count:
        print("WARN treated as failure under --strict.")
        return 1
    if warn_count:
        print(
            "Note: WARN entries are typical for derived / aggregated cells "
            "(e.g. Fisher / Stouffer combination p-values, gap deltas).\n"
            "Inspect the listed samples manually; the underlying CSVs are present."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
