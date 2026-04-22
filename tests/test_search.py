"""Tests for search algorithms: A*, Dijkstra, bidirectional A*, and batched queries."""

import math

import pytest
import scipy.sparse.csgraph
import torch

from aac.graphs.convert import edges_to_graph, graph_to_scipy
from aac.graphs.types import Graph


# ---------------------------------------------------------------------------
# Helper: build a grid graph with coordinates for heuristic testing
# ---------------------------------------------------------------------------


def _build_grid_graph(rows: int, cols: int) -> Graph:
    """Build a grid graph with Euclidean edge weights and coordinates.

    Nodes are numbered row-major: node = row * cols + col.
    Edges connect 4-neighbors (up, down, left, right) with weight = 1.0.
    Coordinates are (col, row) so Euclidean distance is admissible.
    """
    sources = []
    targets = []
    weights = []
    coords = []

    for r in range(rows):
        for c in range(cols):
            coords.append([float(c), float(r)])
            node = r * cols + c
            # Right neighbor
            if c + 1 < cols:
                neighbor = r * cols + (c + 1)
                sources.append(node)
                targets.append(neighbor)
                weights.append(1.0)
            # Down neighbor
            if r + 1 < rows:
                neighbor = (r + 1) * cols + c
                sources.append(node)
                targets.append(neighbor)
                weights.append(1.0)

    num_nodes = rows * cols
    s = torch.tensor(sources, dtype=torch.int64)
    t = torch.tensor(targets, dtype=torch.int64)
    w = torch.tensor(weights, dtype=torch.float64)
    coordinates = torch.tensor(coords, dtype=torch.float64)

    return edges_to_graph(s, t, w, num_nodes, is_directed=False, coordinates=coordinates)


# ---------------------------------------------------------------------------
# SearchResult type tests
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_search_result_frozen(self):
        """SearchResult is frozen dataclass with path/cost/expansions/optimal/h_source."""
        from aac.search.types import SearchResult

        result = SearchResult(
            path=[0, 1, 2],
            cost=3.0,
            expansions=5,
            optimal=True,
            h_source=1.5,
        )
        assert result.path == [0, 1, 2]
        assert result.cost == 3.0
        assert result.expansions == 5
        assert result.optimal is True
        assert result.h_source == 1.5

        # Frozen: cannot assign
        with pytest.raises(AttributeError):
            result.cost = 999.0  # type: ignore[misc]

    def test_search_result_default_h_source(self):
        """h_source defaults to 0.0."""
        from aac.search.types import SearchResult

        result = SearchResult(path=[], cost=float("inf"), expansions=0, optimal=True)
        assert result.h_source == 0.0

    def test_search_result_expansion_fields_default_none(self):
        """expanded_nodes and g_values default to None (backward compatible)."""
        from aac.search.types import SearchResult

        result = SearchResult(path=[0], cost=0.0, expansions=0, optimal=True)
        assert result.expanded_nodes is None
        assert result.g_values is None

    def test_search_result_expansion_fields_populated(self):
        """expanded_nodes and g_values can be populated explicitly."""
        from aac.search.types import SearchResult

        result = SearchResult(
            path=[0, 1],
            cost=2.0,
            expansions=2,
            optimal=True,
            expanded_nodes=[0, 1],
            g_values={0: 0.0, 1: 2.0},
        )
        assert result.expanded_nodes == [0, 1]
        assert result.g_values == {0: 0.0, 1: 2.0}


# ---------------------------------------------------------------------------
# A* expansion tracking tests (VIZ-02)
# ---------------------------------------------------------------------------


class TestAstarExpansionTracking:
    def test_tracking_disabled_by_default(self, small_undirected_graph):
        """A* without track_expansions returns None for expansion fields."""
        from aac.search.astar import astar

        zero_h = lambda node, target: 0.0
        result = astar(small_undirected_graph, 0, 4, heuristic=zero_h)
        assert result.expanded_nodes is None
        assert result.g_values is None

    def test_tracking_populates_expanded_nodes(self, small_undirected_graph):
        """A* with track_expansions=True returns ordered expanded node list."""
        from aac.search.astar import astar

        zero_h = lambda node, target: 0.0
        result = astar(
            small_undirected_graph, 0, 4, heuristic=zero_h, track_expansions=True
        )
        assert result.expanded_nodes is not None
        assert len(result.expanded_nodes) == result.expansions
        # Source must be first expanded
        assert result.expanded_nodes[0] == 0
        # Target must be last expanded (A* terminates on target expansion)
        assert result.expanded_nodes[-1] == 4

    def test_tracking_populates_g_values(self, small_undirected_graph):
        """A* with track_expansions=True returns g_values dict with source g=0."""
        from aac.search.astar import astar

        zero_h = lambda node, target: 0.0
        result = astar(
            small_undirected_graph, 0, 4, heuristic=zero_h, track_expansions=True
        )
        assert result.g_values is not None
        assert result.g_values[0] == 0.0
        # Target g-value should equal path cost
        assert abs(result.g_values[4] - result.cost) < 1e-10

    def test_tracking_source_equals_target(self, small_undirected_graph):
        """A* with track_expansions on source==target returns empty list and source g."""
        from aac.search.astar import astar

        zero_h = lambda node, target: 0.0
        result = astar(
            small_undirected_graph, 2, 2, heuristic=zero_h, track_expansions=True
        )
        assert result.expanded_nodes == []
        assert result.g_values == {2: 0.0}

    def test_tracking_does_not_change_cost(self):
        """Enabling track_expansions does not change the search result."""
        from aac.search.astar import astar

        grid = _build_grid_graph(5, 5)
        zero_h = lambda node, target: 0.0

        r_no = astar(grid, 0, 24, heuristic=zero_h)
        r_yes = astar(grid, 0, 24, heuristic=zero_h, track_expansions=True)

        assert abs(r_no.cost - r_yes.cost) < 1e-10
        assert r_no.expansions == r_yes.expansions
        assert r_no.path == r_yes.path

    def test_tracking_unreachable(self, small_directed_graph):
        """A* with track_expansions on unreachable target returns expansion data."""
        from aac.search.astar import astar

        zero_h = lambda node, target: 0.0
        result = astar(
            small_directed_graph, 4, 0, heuristic=zero_h, track_expansions=True
        )
        assert result.path == []
        assert result.cost == float("inf")
        assert result.expanded_nodes is not None
        # Should have expanded at least the source
        assert len(result.expanded_nodes) > 0
        assert result.expanded_nodes[0] == 4
        assert result.g_values is not None

    def test_gvalues_keys_match_expanded_nodes(self):
        """g_values keys must exactly match expanded_nodes (no open-list-only entries).

        Uses a 5-node directed graph where A* (zero heuristic) finds the target
        before expanding all discovered nodes:
          0 -> 1 (w=1), 0 -> 2 (w=1), 1 -> 3 (w=1), 2 -> 4 (w=100)
        Source=0, target=3.  Node 4 gets discovered (g=101) via node 2 but is
        never expanded because target 3 is reached first.  The bug in
        _extract_g_values would include node 4 in g_values despite it not
        being in expanded_nodes.
        """
        from aac.search.astar import astar

        graph = edges_to_graph(
            sources=torch.tensor([0, 0, 1, 2], dtype=torch.int64),
            targets=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            weights=torch.tensor([1.0, 1.0, 1.0, 100.0], dtype=torch.float64),
            num_nodes=5,
            is_directed=True,
        )

        zero_h = lambda node, target: 0.0
        result = astar(graph, 0, 3, heuristic=zero_h, track_expansions=True)
        assert result.expanded_nodes is not None
        assert result.g_values is not None
        assert set(result.g_values.keys()) == set(result.expanded_nodes), (
            f"g_values keys {set(result.g_values.keys())} != "
            f"expanded_nodes {set(result.expanded_nodes)}"
        )


# ---------------------------------------------------------------------------
# A* tests
# ---------------------------------------------------------------------------


class TestAstar:
    def test_astar_optimal_undirected(self, small_undirected_graph):
        """A* on small_undirected_graph with zero heuristic returns optimal path."""
        from aac.search.astar import astar
        from aac.search.dijkstra import dijkstra

        zero_h = lambda node, target: 0.0
        result_astar = astar(small_undirected_graph, 0, 4, heuristic=zero_h)
        result_dijk = dijkstra(small_undirected_graph, 0, 4)

        assert abs(result_astar.cost - result_dijk.cost) < 1e-10
        assert result_astar.optimal is True
        # Known: d(0,4) = 7 via 0->1->2->3->4
        assert abs(result_astar.cost - 7.0) < 1e-10

    def test_astar_optimal_directed(self, small_directed_graph):
        """A* on small_directed_graph with zero heuristic returns correct shortest path."""
        from aac.search.astar import astar

        zero_h = lambda node, target: 0.0
        result = astar(small_directed_graph, 0, 4, heuristic=zero_h)

        assert result.optimal is True
        # Known: d(0,4) = 6 via 0->1->2->3->4
        assert abs(result.cost - 6.0) < 1e-10
        assert result.path[0] == 0
        assert result.path[-1] == 4

    def test_astar_fewer_expansions(self):
        """A* with admissible heuristic expands <= nodes compared to Dijkstra."""
        from aac.search.astar import astar
        from aac.search.dijkstra import dijkstra

        grid = _build_grid_graph(5, 5)  # 25-node grid
        coords = grid.coordinates
        assert coords is not None

        # Euclidean heuristic (admissible on unit-weight grid)
        def euclidean_h(node: int, target: int) -> float:
            dx = coords[node, 0].item() - coords[target, 0].item()
            dy = coords[node, 1].item() - coords[target, 1].item()
            return math.sqrt(dx * dx + dy * dy)

        # Query from top-left to bottom-right
        source, target = 0, 24
        result_astar = astar(grid, source, target, heuristic=euclidean_h)
        result_dijk = dijkstra(grid, source, target)

        # A* should expand fewer or equal nodes
        assert result_astar.expansions <= result_dijk.expansions, (
            f"A* expanded {result_astar.expansions} > Dijkstra {result_dijk.expansions}"
        )
        # Both should find optimal cost
        assert abs(result_astar.cost - result_dijk.cost) < 1e-10

    def test_astar_unreachable(self, small_directed_graph):
        """A* returns empty path and inf cost for unreachable target."""
        from aac.search.astar import astar

        zero_h = lambda node, target: 0.0
        # In the directed graph, node 4 has no outgoing edges to reach node 0
        result = astar(small_directed_graph, 4, 0, heuristic=zero_h)

        assert result.path == []
        assert result.cost == float("inf")
        assert result.optimal is False

    def test_astar_source_equals_target(self, small_undirected_graph):
        """A* returns single-node path with cost 0.0 when source == target."""
        from aac.search.astar import astar

        zero_h = lambda node, target: 0.0
        result = astar(small_undirected_graph, 2, 2, heuristic=zero_h)

        assert result.path == [2]
        assert result.cost == 0.0
        assert result.expansions == 0
        assert result.optimal is True

    def test_astar_path_reconstruction(self, small_undirected_graph):
        """Returned path is valid: consecutive vertices are neighbors with correct weights."""
        from aac.search.astar import astar

        zero_h = lambda node, target: 0.0
        result = astar(small_undirected_graph, 0, 4, heuristic=zero_h)

        assert len(result.path) >= 2
        graph = small_undirected_graph
        crow = graph.crow_indices
        col = graph.col_indices
        vals = graph.values

        total_weight = 0.0
        for i in range(len(result.path) - 1):
            u = result.path[i]
            v = result.path[i + 1]
            # Find edge (u, v) in CSR
            start = crow[u].item()
            end = crow[u + 1].item()
            neighbors = col[start:end].tolist()
            assert v in neighbors, f"Edge ({u}, {v}) not found in graph"
            idx = start + neighbors.index(v)
            total_weight += vals[idx].item()

        assert abs(total_weight - result.cost) < 1e-10, (
            f"Path weight {total_weight} != reported cost {result.cost}"
        )


# ---------------------------------------------------------------------------
# Dijkstra tests
# ---------------------------------------------------------------------------


class TestDijkstra:
    def test_dijkstra_correctness_undirected(self, small_undirected_graph):
        """Dijkstra on small_undirected_graph matches scipy reference."""
        from aac.search.dijkstra import dijkstra

        scipy_mat = graph_to_scipy(small_undirected_graph)
        ref_dists = scipy.sparse.csgraph.dijkstra(scipy_mat, directed=False)

        for s in range(5):
            for t in range(5):
                result = dijkstra(small_undirected_graph, s, t)
                expected = ref_dists[s, t]
                if expected == float("inf"):
                    assert result.cost == float("inf")
                else:
                    assert abs(result.cost - expected) < 1e-10, (
                        f"d({s},{t}): got {result.cost}, expected {expected}"
                    )

    def test_dijkstra_correctness_directed(self, small_directed_graph):
        """Dijkstra on small_directed_graph matches scipy reference."""
        from aac.search.dijkstra import dijkstra

        scipy_mat = graph_to_scipy(small_directed_graph)
        ref_dists = scipy.sparse.csgraph.dijkstra(scipy_mat, directed=True)

        for s in range(5):
            for t in range(5):
                result = dijkstra(small_directed_graph, s, t)
                expected = ref_dists[s, t]
                if expected == float("inf"):
                    assert result.cost == float("inf")
                else:
                    assert abs(result.cost - expected) < 1e-10, (
                        f"d({s},{t}): got {result.cost}, expected {expected}"
                    )

    def test_dijkstra_all_pairs(self, small_undirected_graph):
        """For every (s,t) pair in small graph, dijkstra cost matches scipy reference."""
        from aac.search.dijkstra import dijkstra

        scipy_mat = graph_to_scipy(small_undirected_graph)
        ref_dists = scipy.sparse.csgraph.dijkstra(scipy_mat, directed=False)

        for s in range(small_undirected_graph.num_nodes):
            for t in range(small_undirected_graph.num_nodes):
                result = dijkstra(small_undirected_graph, s, t)
                expected = ref_dists[s, t]
                if expected == float("inf"):
                    assert result.cost == float("inf")
                else:
                    assert abs(result.cost - expected) < 1e-10

    def test_dijkstra_track_expansions(self, small_undirected_graph):
        """dijkstra forwards track_expansions to astar correctly."""
        from aac.search.dijkstra import dijkstra

        result = dijkstra(small_undirected_graph, 0, 4, track_expansions=True)
        assert result.expanded_nodes is not None
        assert result.g_values is not None
        assert result.expanded_nodes[0] == 0
        assert result.g_values[0] == 0.0
        assert len(result.expanded_nodes) == result.expansions


# ---------------------------------------------------------------------------
# Bidirectional A* tests
# ---------------------------------------------------------------------------


class TestBidirectionalAstar:
    def test_bidirectional_correctness_undirected(self, small_undirected_graph):
        """bidirectional_astar on small_undirected_graph matches unidirectional A* cost."""
        from aac.search.astar import astar
        from aac.search.bidirectional import bidirectional_astar

        zero_h = lambda node, target: 0.0

        for s in range(5):
            for t in range(5):
                if s == t:
                    continue
                result_uni = astar(small_undirected_graph, s, t, heuristic=zero_h)
                result_bi = bidirectional_astar(
                    small_undirected_graph, s, t,
                    h_forward=zero_h,
                    h_backward=zero_h,
                )
                assert abs(result_bi.cost - result_uni.cost) < 1e-10, (
                    f"Bidirectional cost {result_bi.cost} != A* cost {result_uni.cost} "
                    f"for ({s}, {t})"
                )

    def test_bidirectional_path_valid(self, small_undirected_graph):
        """Returned path is valid: consecutive vertices are neighbors."""
        from aac.search.bidirectional import bidirectional_astar

        zero_h = lambda node, target: 0.0
        result = bidirectional_astar(
            small_undirected_graph, 0, 4,
            h_forward=zero_h,
            h_backward=zero_h,
        )

        assert len(result.path) >= 2
        assert result.path[0] == 0
        assert result.path[-1] == 4

        graph = small_undirected_graph
        crow = graph.crow_indices
        col = graph.col_indices
        vals = graph.values

        total_weight = 0.0
        for i in range(len(result.path) - 1):
            u = result.path[i]
            v = result.path[i + 1]
            start = crow[u].item()
            end = crow[u + 1].item()
            neighbors = col[start:end].tolist()
            assert v in neighbors, f"Edge ({u}, {v}) not found in graph"
            idx = start + neighbors.index(v)
            total_weight += vals[idx].item()

        assert abs(total_weight - result.cost) < 1e-10, (
            f"Path weight {total_weight} != reported cost {result.cost}"
        )

    def test_bidirectional_mu_stopping(self, small_undirected_graph):
        """Bidirectional stops correctly and returns optimal cost."""
        from aac.search.astar import astar
        from aac.search.bidirectional import bidirectional_astar

        zero_h = lambda node, target: 0.0
        result_bi = bidirectional_astar(
            small_undirected_graph, 0, 4,
            h_forward=zero_h,
            h_backward=zero_h,
        )
        result_uni = astar(small_undirected_graph, 0, 4, heuristic=zero_h)

        # Must return optimal cost (matching unidirectional)
        assert abs(result_bi.cost - result_uni.cost) < 1e-10
        assert result_bi.optimal is True
        # Known: d(0,4) = 7
        assert abs(result_bi.cost - 7.0) < 1e-10

    def test_bidirectional_unreachable(self, small_directed_graph):
        """Returns empty path and inf cost for unreachable target on directed graph."""
        from aac.search.bidirectional import bidirectional_astar

        zero_h = lambda node, target: 0.0
        # Node 4 cannot reach node 0 in this directed graph
        result = bidirectional_astar(
            small_directed_graph, 4, 0,
            h_forward=zero_h,
            h_backward=zero_h,
        )
        assert result.path == []
        assert result.cost == float("inf")
        assert result.optimal is False

    def test_bidirectional_fewer_total_expansions(self):
        """Bidirectional A* with heuristic expands fewer nodes than without on grid.

        On a large grid, bidirectional A* with Euclidean heuristic should expand
        fewer total nodes than bidirectional A* with zero heuristic (bidirectional
        Dijkstra), demonstrating the benefit of heuristic guidance in both directions.
        """
        from aac.search.bidirectional import bidirectional_astar

        grid = _build_grid_graph(10, 10)  # 100-node grid
        coords = grid.coordinates
        assert coords is not None

        def euclidean_h(node: int, target: int) -> float:
            dx = coords[node, 0].item() - coords[target, 0].item()
            dy = coords[node, 1].item() - coords[target, 1].item()
            return math.sqrt(dx * dx + dy * dy)

        zero_h = lambda node, target: 0.0
        source, target = 0, 99  # top-left to bottom-right

        result_with_h = bidirectional_astar(
            grid, source, target,
            h_forward=euclidean_h,
            h_backward=euclidean_h,
        )
        result_no_h = bidirectional_astar(
            grid, source, target,
            h_forward=zero_h,
            h_backward=zero_h,
        )

        # Both should find optimal cost
        assert abs(result_with_h.cost - result_no_h.cost) < 1e-10
        # Heuristic-guided bidirectional should expand fewer nodes
        assert result_with_h.expansions <= result_no_h.expansions, (
            f"With heuristic: {result_with_h.expansions} > "
            f"without: {result_no_h.expansions}"
        )

    def test_bidirectional_source_equals_target(self, small_undirected_graph):
        """Bidirectional returns single-node path with cost 0 when source == target."""
        from aac.search.bidirectional import bidirectional_astar

        zero_h = lambda node, target: 0.0
        result = bidirectional_astar(
            small_undirected_graph, 2, 2,
            h_forward=zero_h,
            h_backward=zero_h,
        )
        assert result.path == [2]
        assert result.cost == 0.0
        assert result.expansions == 0


# ---------------------------------------------------------------------------
# Batched query tests
# ---------------------------------------------------------------------------


class TestBatchSearch:
    def test_batch_search_correctness(self, small_undirected_graph):
        """batch_search with 10 queries returns 10 SearchResults matching individual astar."""
        from aac.search.astar import astar
        from aac.search.batch import batch_search

        zero_h = lambda node, target: 0.0
        queries = [(s, t) for s in range(5) for t in range(5) if s != t][:10]

        results = batch_search(small_undirected_graph, queries, heuristic=zero_h)

        assert len(results) == 10
        for i, (s, t) in enumerate(queries):
            individual = astar(small_undirected_graph, s, t, heuristic=zero_h)
            assert abs(results[i].cost - individual.cost) < 1e-10, (
                f"Query ({s},{t}): batch cost {results[i].cost} != "
                f"individual cost {individual.cost}"
            )

    def test_batch_search_empty(self, small_undirected_graph):
        """batch_search with 0 queries returns empty list."""
        from aac.search.batch import batch_search

        zero_h = lambda node, target: 0.0
        results = batch_search(small_undirected_graph, [], heuristic=zero_h)
        assert results == []

    def test_batch_search_mixed(self, small_directed_graph):
        """batch_search with mix of reachable and unreachable queries handles all correctly."""
        from aac.search.astar import astar
        from aac.search.batch import batch_search

        zero_h = lambda node, target: 0.0
        # Mix reachable (0->4, 0->1) and unreachable (4->0, 3->0)
        queries = [(0, 4), (0, 1), (4, 0), (3, 0)]

        results = batch_search(small_directed_graph, queries, heuristic=zero_h)

        assert len(results) == 4
        for i, (s, t) in enumerate(queries):
            individual = astar(small_directed_graph, s, t, heuristic=zero_h)
            assert abs(results[i].cost - individual.cost) < 1e-10 or (
                results[i].cost == float("inf") and individual.cost == float("inf")
            )
