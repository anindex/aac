"""Aggregate per-query CSV results into summary DataFrames."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def get_git_hash() -> str:
    """Return the short git commit hash of the current HEAD.

    Falls back to ``"unknown"`` if git is unavailable or the working
    directory is not a git repository.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def write_csv_metadata(f) -> None:
    """Write a metadata comment line to an open file handle.

    The line has the form::

        # git_hash=<hash>, timestamp=<ISO 8601 UTC>

    Because it starts with ``#``, pandas can skip it via
    ``pd.read_csv(..., comment='#')``, and the stdlib :mod:`csv` module
    ignores it automatically when reading rows.

    Args:
        f: Writable file object (text mode).
    """
    git_hash = get_git_hash()
    timestamp = datetime.now(timezone.utc).isoformat()
    f.write(f"# git_hash={git_hash}, timestamp={timestamp}\n")


def aggregate_results(results_dir: str, track: str) -> pd.DataFrame:
    """Aggregate per-query CSV files into a summary DataFrame.

    Scans ``results_dir/track/*.csv`` for per-query result files produced by
    :class:`experiments.metrics.collector.MetricsCollector`. Each file is
    summarised into one row with aggregate statistics.

    File naming convention expected:
        ``{method}_{graph}[_m{m}][_K{K0}].csv``
        or any name containing the method and graph tokens.

    Args:
        results_dir: Root results directory (e.g. ``"results"``).
        track: Track sub-directory name (e.g. ``"dimacs"``, ``"osmnx"``).

    Returns:
        DataFrame with columns: graph, method, m, K0, expansions_mean,
        expansions_median, p50_ms, p95_ms, memory_bytes_per_vertex,
        preprocess_total_sec, num_violations.
    """
    track_dir = Path(results_dir) / track
    csv_files = sorted(track_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {track_dir}. Run experiments first."
        )

    rows: list[dict] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, comment="#")
        if df.empty:
            continue

        info = _parse_filename(csv_path.stem)
        expansions = df["expansions"].values
        latencies = df["latency_ms"].values

        row = {
            "graph": info.get("graph", "unknown"),
            "method": info.get("method", "unknown"),
            "m": info.get("m", 0),
            "K0": info.get("K0", 0),
            "expansions_mean": float(np.mean(expansions)),
            "expansions_median": float(np.median(expansions)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            # ALT stores both forward and reverse distances per landmark (2*m),
            # while AAC and FastMap store m values per vertex.
            "memory_bytes_per_vertex": (
                info.get("m", 0) * 2 * 4 if info.get("method") == "alt"
                else info.get("m", 0) * 4
            ),  # float32
            "preprocess_total_sec": 0.0,  # filled from metadata if available
            "num_violations": int((~df["optimal"]).sum()) if "optimal" in df.columns else 0,
        }

        # Try reading companion metadata JSON/YAML if present
        meta_path = csv_path.with_suffix(".json")
        if meta_path.exists():
            import json

            with open(meta_path) as f:
                meta = json.load(f)
            row["memory_bytes_per_vertex"] = meta.get(
                "memory_bytes_per_vertex", row["memory_bytes_per_vertex"]
            )
            row["preprocess_total_sec"] = meta.get(
                "preprocess_total_sec", row["preprocess_total_sec"]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def merge_results(summary_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple summary DataFrames and sort.

    Args:
        summary_dfs: List of DataFrames (e.g. from different tracks).

    Returns:
        Merged DataFrame sorted by graph, then memory_bytes_per_vertex.
    """
    merged = pd.concat(summary_dfs, ignore_index=True)
    merged = merged.sort_values(
        ["graph", "memory_bytes_per_vertex"], ignore_index=True
    )
    return merged


def _parse_filename(stem: str) -> dict:
    """Extract method, graph, m, K0 from a result file stem.

    Examples:
        ``aac_NY_m16_K64``  -> {method: aac, graph: NY, m: 16, K0: 64}
        ``alt_BAY_m8``             -> {method: alt, graph: BAY, m: 8, K0: 0}
        ``dijkstra_COL``           -> {method: dijkstra, graph: COL, m: 0, K0: 0}
    """
    info: dict = {"method": "unknown", "graph": "unknown", "m": 0, "K0": 0}

    # Extract m parameter
    m_match = re.search(r"_m(\d+)", stem)
    if m_match:
        info["m"] = int(m_match.group(1))

    # Extract K0 parameter
    k_match = re.search(r"_K(\d+)", stem)
    if k_match:
        info["K0"] = int(k_match.group(1))

    # Remove parameter suffixes to find method and graph
    clean = re.sub(r"_m\d+", "", stem)
    clean = re.sub(r"_K\d+", "", clean)

    # Known methods (order matters: aac before others to avoid partial match)
    known_methods = ["aac", "dijkstra", "fastmap", "alt"]
    for method in known_methods:
        if clean.startswith(method):
            info["method"] = method
            remainder = clean[len(method) :].strip("_")
            if remainder:
                info["graph"] = remainder
            break

    return info
