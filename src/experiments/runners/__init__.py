"""Experiment runner dispatch for all tracks.

Track 1: DIMACS (exact weighted point-to-point)
Track 2: OSMnx (road graphs, PHIL comparison)
Track 3: Warcraft (Contextual vs DataSP), Cabspotting (Contextual vs DataSP)
"""

from __future__ import annotations

from experiments.runners.base import BaseRunner
from experiments.runners.cabspotting_runner import CabspottingRunner
from experiments.runners.dimacs_runner import DIMACSRunner
from experiments.runners.osmnx_runner import OSMnxRunner
from experiments.runners.warcraft_runner import WarcraftRunner


def get_runner(track_name: str) -> type[BaseRunner]:
    """Return the runner class for the given track name.

    Args:
        track_name: One of "dimacs", "osmnx", "warcraft", or "cabspotting".

    Returns:
        Runner class (not instance) corresponding to the track.

    Raises:
        ValueError: If track_name is not recognized.
    """
    runners: dict[str, type[BaseRunner]] = {
        "dimacs": DIMACSRunner,
        "osmnx": OSMnxRunner,
        "warcraft": WarcraftRunner,
        "cabspotting": CabspottingRunner,
    }
    if track_name not in runners:
        raise ValueError(
            f"Unknown track: {track_name}. Available: {list(runners.keys())}"
        )
    return runners[track_name]


__all__ = [
    "BaseRunner",
    "CabspottingRunner",
    "DIMACSRunner",
    "OSMnxRunner",
    "WarcraftRunner",
    "get_runner",
]
