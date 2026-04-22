"""Regression tests for sentinel masking in Lipschitz heuristic (BUG-EXH01)."""

import numpy as np

from aac.compression.lipschitz import LipschitzCompressor, make_lipschitz_heuristic
from aac.utils.numerics import SENTINEL


class TestLipschitzSentinelMasking:
    """Verify that sentinel values (1e18) in distance arrays do not corrupt
    the Lipschitz heuristic.  Before the fix, sentinel distances were fed
    directly into the neural network, producing garbage output that could
    violate admissibility."""

    def _make_distances_with_sentinel(self, K, V, sentinel_vertices, is_directed):
        """Create (K, V) distance arrays where sentinel_vertices have SENTINEL distances."""
        rng = np.random.RandomState(42)
        d_out = rng.uniform(10, 1000, size=(K, V))
        for v in sentinel_vertices:
            d_out[:, v] = SENTINEL  # All landmarks unreachable from v
        if is_directed:
            d_in = rng.uniform(10, 1000, size=(K, V))
            for v in sentinel_vertices:
                d_in[:, v] = SENTINEL
            return d_out, d_in
        return d_out, None

    def test_directed_sentinel_masking(self):
        """Directed graph with some unreachable vertices returns 0, not garbage."""
        K, V, m = 4, 10, 3
        sentinel_verts = [7, 8]  # These vertices are unreachable
        d_out, d_in = self._make_distances_with_sentinel(K, V, sentinel_verts, is_directed=True)

        comp = LipschitzCompressor(K=K, m=m, is_directed=True)
        h = make_lipschitz_heuristic(comp.net_fwd, comp.net_bwd, d_out, d_in, is_directed=True)

        # Queries involving sentinel vertices should return 0.0
        for sv in sentinel_verts:
            assert h(sv, 0) == 0.0, f"h({sv}, 0) should be 0 for unreachable vertex"
            assert h(0, sv) == 0.0, f"h(0, {sv}) should be 0 for unreachable target"

        # Queries between normal vertices should return a finite value
        val = h(0, 1)
        assert np.isfinite(val), f"h(0, 1) should be finite, got {val}"
        assert val < 1e10, f"h(0, 1) should be reasonable, got {val}"

    def test_undirected_sentinel_masking(self):
        """Undirected graph with some unreachable vertices returns 0."""
        K, V, m = 4, 10, 3
        sentinel_verts = [5, 6]
        d_out, _ = self._make_distances_with_sentinel(K, V, sentinel_verts, is_directed=False)

        comp = LipschitzCompressor(K=K, m=m, is_directed=False)
        # Undirected compressor exposes .net, not .net_fwd/.net_bwd
        h = make_lipschitz_heuristic(comp.net, None, d_out, None, is_directed=False)

        for sv in sentinel_verts:
            assert h(sv, 0) == 0.0
            assert h(0, sv) == 0.0

        val = h(0, 1)
        assert np.isfinite(val) and val < 1e10

    def test_all_sentinel_returns_zero(self):
        """If ALL vertices have sentinel distances, h must return 0."""
        K, V, m = 3, 5, 2
        d_out = np.full((K, V), SENTINEL)
        d_in = np.full((K, V), SENTINEL)

        comp = LipschitzCompressor(K=K, m=m, is_directed=True)
        h = make_lipschitz_heuristic(comp.net_fwd, comp.net_bwd, d_out, d_in, is_directed=True)

        for u in range(V):
            for v in range(V):
                if u != v:
                    assert h(u, v) == 0.0, f"h({u},{v}) should be 0 when all sentinel"

    def test_no_sentinel_behaves_normally(self):
        """With no sentinels, heuristic should produce non-trivial values."""
        K, V, m = 4, 8, 3
        rng = np.random.RandomState(123)
        d_out = rng.uniform(10, 1000, size=(K, V))
        d_in = rng.uniform(10, 1000, size=(K, V))

        comp = LipschitzCompressor(K=K, m=m, is_directed=True)
        h = make_lipschitz_heuristic(comp.net_fwd, comp.net_bwd, d_out, d_in, is_directed=True)

        # At least some pairs should have non-zero heuristic
        values = [h(u, v) for u in range(4) for v in range(4, 8)]
        assert any(v > 0 for v in values), "With no sentinels, some h values should be > 0"
        assert all(np.isfinite(v) for v in values), "All values should be finite"

    def test_single_sentinel_landmark(self):
        """Only one landmark has sentinel for a specific vertex."""
        K, V, m = 4, 6, 2
        rng = np.random.RandomState(42)
        d_out = rng.uniform(10, 500, size=(K, V))
        d_in = rng.uniform(10, 500, size=(K, V))
        # Only landmark 2 is unreachable from vertex 3
        d_out[2, 3] = SENTINEL
        d_in[2, 3] = SENTINEL

        comp = LipschitzCompressor(K=K, m=m, is_directed=True)
        h = make_lipschitz_heuristic(comp.net_fwd, comp.net_bwd, d_out, d_in, is_directed=True)

        # Vertex 3 has a sentinel input, so queries involving it should return 0
        assert h(3, 0) == 0.0
        assert h(0, 3) == 0.0
        # But vertex 0 vs 1 should be fine
        val = h(0, 1)
        assert np.isfinite(val) and val < 1e10
