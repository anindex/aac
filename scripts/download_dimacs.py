"""Download DIMACS 9th Challenge road network files."""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

BASE_URL = "http://www.diag.uniroma1.it/challenge9/data/USA-road-d"
GRAPHS = {
    "NY": {"nodes": 264346, "edges": 733846},
    "BAY": {"nodes": 321270, "edges": 800172},
    "COL": {"nodes": 435666, "edges": 1057066},
    "FLA": {"nodes": 1070376, "edges": 2712798},
}


def download_graph(name: str, data_dir: str = "data/dimacs") -> None:
    """Download and decompress a DIMACS road network graph.

    Downloads .gr.gz (edge weights) and .co.gz (coordinates) files,
    decompresses them, and removes the compressed originals.

    Args:
        name: Graph name (e.g., "NY", "BAY", "COL", "FLA").
        data_dir: Directory to store downloaded files.
    """
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in [".gr.gz", ".co.gz"]:
        filename = f"USA-road-d.{name}{ext}"
        url = f"{BASE_URL}/{filename}"
        gz_path = out / filename
        final_path = out / filename.replace(".gz", "")
        if final_path.exists():
            print(f"  {final_path} already exists, skipping")
            continue
        if not gz_path.exists():
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, gz_path)
        print(f"  Decompressing {gz_path} ...")
        with gzip.open(gz_path, "rb") as f_in, open(final_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()


def main() -> None:
    """Download all configured DIMACS graphs."""
    for name in GRAPHS:
        print(f"Downloading {name} ...")
        download_graph(name)
    print("Done.")


if __name__ == "__main__":
    main()
