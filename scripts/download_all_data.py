#!/usr/bin/env python
"""Download all datasets needed to reproduce the paper experiments.

This script downloads DIMACS road graphs, OSMnx city/country extracts, and
generates synthetic Warcraft terrain maps. Run this once before executing
``scripts/reproduce_paper.py``.

Usage:
    python scripts/download_all_data.py          # all datasets
    python scripts/download_all_data.py --dimacs  # DIMACS only
    python scripts/download_all_data.py --osmnx   # OSMnx only
    python scripts/download_all_data.py --warcraft # Warcraft only

Approximate download sizes:
    DIMACS:   ~200 MB (4 road graphs: NY, BAY, COL, FLA)
    OSMnx:    ~150 MB (5 city/country extracts)
    Warcraft: ~50 MB  (generated locally, no download)

All data is stored under ``data/`` relative to the repository root.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_dimacs() -> None:
    """Download DIMACS 9th Challenge USA road graphs."""
    print("=" * 60)
    print("  Downloading DIMACS road graphs")
    print("=" * 60)
    script = _PROJECT_ROOT / "scripts" / "download_dimacs.py"
    subprocess.run([sys.executable, str(script)], check=True)
    print()


def download_osmnx() -> None:
    """Download OSMnx city and country graph extracts."""
    print("=" * 60)
    print("  Downloading OSMnx city/country graphs")
    print("=" * 60)
    script = _PROJECT_ROOT / "scripts" / "download_osmnx.py"
    subprocess.run([sys.executable, str(script)], check=True)
    print()


def generate_warcraft() -> None:
    """Generate synthetic Warcraft terrain maps."""
    print("=" * 60)
    print("  Generating Warcraft terrain maps")
    print("=" * 60)
    script = _PROJECT_ROOT / "scripts" / "generate_warcraft_data.py"
    subprocess.run([sys.executable, str(script)], check=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all datasets for paper reproduction"
    )
    parser.add_argument("--dimacs", action="store_true",
                        help="Download DIMACS road graphs only")
    parser.add_argument("--osmnx", action="store_true",
                        help="Download OSMnx city/country graphs only")
    parser.add_argument("--warcraft", action="store_true",
                        help="Generate Warcraft terrain maps only")
    args = parser.parse_args()

    # If no flags specified, download everything
    download_all = not (args.dimacs or args.osmnx or args.warcraft)

    print("AAC: Downloading datasets for paper reproduction")
    print(f"Repository root: {_PROJECT_ROOT}")
    print()

    if download_all or args.dimacs:
        download_dimacs()

    if download_all or args.osmnx:
        download_osmnx()

    if download_all or args.warcraft:
        generate_warcraft()

    print("=" * 60)
    print("  All datasets ready!")
    print(f"  Data directory: {_PROJECT_ROOT / 'data'}")
    print("=" * 60)
    print()
    print("Next step: python scripts/reproduce_paper.py")


if __name__ == "__main__":
    main()
