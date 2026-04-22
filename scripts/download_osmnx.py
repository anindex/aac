"""Download and cache OSMnx city graphs for Track 2."""

from __future__ import annotations

import pickle
from pathlib import Path

CITIES = {
    "modena": {"place": "Modena, Italy", "network_type": "drive"},
    "new_york": {"place": "Manhattan, New York, USA", "network_type": "drive"},
}


def download_city(
    name: str, place: str, network_type: str, data_dir: str = "data/osmnx"
) -> None:
    """Download a city road network via OSMnx and cache as pickle.

    Args:
        name: Short name for the city (used as filename).
        place: Geocoding query string for OSMnx.
        network_type: OSMnx network type (e.g., "drive").
        data_dir: Directory to store cached graphs.
    """
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / f"{name}.pkl"
    if cache_path.exists():
        print(f"  {cache_path} already exists, skipping")
        return
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("pip install osmnx to download city graphs")
    print(f"  Downloading {place} via OSMnx ...")
    G = ox.graph_from_place(place, network_type=network_type)
    with open(cache_path, "wb") as f:
        pickle.dump(G, f)
    print(f"  Saved to {cache_path}")


def main() -> None:
    """Download all configured city graphs."""
    for name, cfg in CITIES.items():
        print(f"Downloading {name} ...")
        download_city(name, **cfg)
    print("Done.")


if __name__ == "__main__":
    main()
