"""Tests for data download scripts."""

from __future__ import annotations


class TestDimacsScript:
    """Tests for DIMACS download script."""

    def test_download_function_exists(self) -> None:
        """download_graph is importable and callable."""
        from scripts.download_dimacs import download_graph

        assert callable(download_graph)

    def test_graph_names(self) -> None:
        """GRAPHS dict contains expected road network names."""
        from scripts.download_dimacs import GRAPHS

        expected = {"NY", "BAY", "COL", "FLA"}
        assert set(GRAPHS.keys()) == expected

    def test_base_url(self) -> None:
        """BASE_URL points to DIMACS challenge data."""
        from scripts.download_dimacs import BASE_URL

        assert "diag.uniroma1.it" in BASE_URL
        assert "USA-road" in BASE_URL

    def test_graph_metadata(self) -> None:
        """Each graph entry has nodes and edges counts."""
        from scripts.download_dimacs import GRAPHS

        for name, meta in GRAPHS.items():
            assert "nodes" in meta, f"{name} missing 'nodes'"
            assert "edges" in meta, f"{name} missing 'edges'"
            assert meta["nodes"] > 0
            assert meta["edges"] > 0


class TestOsmnxScript:
    """Tests for OSMnx download script."""

    def test_download_function_exists(self) -> None:
        """download_city is importable and callable."""
        from scripts.download_osmnx import download_city

        assert callable(download_city)

    def test_cities_dict(self) -> None:
        """CITIES dict contains expected city entries."""
        from scripts.download_osmnx import CITIES

        assert "modena" in CITIES
        assert "new_york" in CITIES

    def test_city_metadata(self) -> None:
        """Each city entry has place and network_type."""
        from scripts.download_osmnx import CITIES

        for name, cfg in CITIES.items():
            assert "place" in cfg, f"{name} missing 'place'"
            assert "network_type" in cfg, f"{name} missing 'network_type'"
