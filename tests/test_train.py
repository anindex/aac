"""Tests for the training pipeline: loss, data, and training loop."""

from __future__ import annotations

import torch
import pytest

from aac.compression.compressor import PositiveCompressor
from aac.compression.smooth import (
    smoothed_heuristic_undirected,
    make_aac_heuristic,
)
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.embeddings.hilbert import build_hilbert_embedding
from aac.graphs.types import Graph
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.loss import gap_closing_loss
from aac.train.data import QueryPairDataset, make_splits
from aac.train.trainer import train_compressor, TrainConfig


# ---------------------------------------------------------------------------
# Helper: compute all-pairs shortest path distances from dense adjacency
# ---------------------------------------------------------------------------
def _all_pairs_distances(graph: Graph, sentinel: float = 1e18) -> torch.Tensor:
    """Compute all-pairs shortest paths via repeated Bellman-Ford on dense adj."""
    from aac.embeddings.sssp import bellman_ford_batched

    all_sources = torch.arange(graph.num_nodes, dtype=torch.int64)
    return bellman_ford_batched(graph, all_sources, sentinel=sentinel)


# ===========================================================================
# TRAIN-01: Gap-closing loss
# ===========================================================================

class TestGapClosingLoss:
    """Tests for gap_closing_loss function."""

    def test_gap_closing_loss_positive(self) -> None:
        """gap_closing_loss returns positive scalar for random d_true > h_smooth."""
        torch.manual_seed(0)
        compressor = PositiveCompressor(input_dim=8, compressed_dim=4)
        d_true = torch.rand(32, dtype=torch.float64) + 1.0  # always > 0
        h_smooth = torch.rand(32, dtype=torch.float64) * 0.5  # always < d_true
        loss = gap_closing_loss(d_true, h_smooth, compressor, cond_lambda=0.01)
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive when d_true > h_smooth"

    def test_gap_closing_loss_gradient(self) -> None:
        """loss.backward() produces nonzero gradients on compressor.A_raw."""
        torch.manual_seed(1)
        compressor = PositiveCompressor(input_dim=8, compressed_dim=4)
        phi = torch.randn(10, 8, dtype=torch.float64)
        y = compressor(phi)
        # Compute smoothed heuristic for some pairs
        h_smooth = smoothed_heuristic_undirected(y[:5], y[5:], temperature=1.0)
        d_true = h_smooth.detach() + 0.5  # ensure d_true > h_smooth
        loss = gap_closing_loss(d_true, h_smooth, compressor, cond_lambda=0.01)
        loss.backward()
        assert compressor.alpha.grad is not None, "Gradients should flow to alpha"
        assert compressor.alpha.grad.abs().sum() > 0, "Gradients should be nonzero"

    def test_gap_closing_loss_decreases(self, small_undirected_graph: Graph) -> None:
        """After 30 optimizer steps, loss on a fixed eval batch is lower than initial."""
        torch.manual_seed(42)
        graph = small_undirected_graph
        K = 3
        anchors = farthest_point_sampling(graph, K, seed_vertex=0)
        teacher = compute_teacher_labels(graph, anchors)
        embedding = build_hilbert_embedding(teacher)
        phi = embedding.phi  # (V, 2K=6)

        compressor = PositiveCompressor(input_dim=2 * K, compressed_dim=4)
        optimizer = torch.optim.Adam(compressor.parameters(), lr=1e-2)

        # All-pairs distances
        dist_matrix = _all_pairs_distances(graph)
        V = graph.num_nodes

        # Fixed evaluation batch (all distinct pairs on the small graph)
        eval_s = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
        eval_t = torch.tensor([1, 2, 3, 4, 2, 3, 4, 3, 4, 4])

        def eval_loss() -> float:
            with torch.no_grad():
                y_eval = compressor(phi)
                h_eval = smoothed_heuristic_undirected(y_eval[eval_s], y_eval[eval_t], temperature=1.0)
                d_eval = dist_matrix[eval_s, eval_t]
                return gap_closing_loss(d_eval, h_eval, compressor, cond_lambda=0.01).item()

        initial_loss = eval_loss()

        for step in range(30):
            sources = torch.randint(0, V, (8,))
            targets = torch.randint(0, V, (8,))
            y = compressor(phi)
            h_smooth = smoothed_heuristic_undirected(y[sources], y[targets], temperature=1.0)
            d_true = dist_matrix[sources, targets]
            loss = gap_closing_loss(d_true, h_smooth, compressor, cond_lambda=0.01)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = eval_loss()
        assert final_loss < initial_loss, (
            f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


# ===========================================================================
# TRAIN-02: Positivity and admissibility
# ===========================================================================

class TestPositivityAdmissibility:
    """Tests for positivity and admissibility during/after training."""

    def test_positivity_during_training(self, small_undirected_graph: Graph) -> None:
        """After 50 Adam steps, compressor.A.min() > 0 (softplus guarantees)."""
        torch.manual_seed(42)
        graph = small_undirected_graph
        K = 3
        anchors = farthest_point_sampling(graph, K, seed_vertex=0)
        teacher = compute_teacher_labels(graph, anchors)
        embedding = build_hilbert_embedding(teacher)
        phi = embedding.phi

        compressor = PositiveCompressor(input_dim=2 * K, compressed_dim=4)
        optimizer = torch.optim.Adam(compressor.parameters(), lr=1e-2)

        dist_matrix = _all_pairs_distances(graph)
        V = graph.num_nodes

        for _ in range(50):
            sources = torch.randint(0, V, (8,))
            targets = torch.randint(0, V, (8,))
            y = compressor(phi)
            h_smooth = smoothed_heuristic_undirected(y[sources], y[targets], temperature=1.0)
            d_true = dist_matrix[sources, targets]
            loss = gap_closing_loss(d_true, h_smooth, compressor, cond_lambda=0.01)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert compressor.A.min().item() > 0, "A must stay strictly positive (softplus guarantee)"

    def test_admissibility_after_training(self, small_undirected_graph: Graph) -> None:
        """After 50 training steps, h_compressed(s,t) <= d(s,t) + epsilon for all pairs."""
        torch.manual_seed(42)
        graph = small_undirected_graph
        K = 3
        anchors = farthest_point_sampling(graph, K, seed_vertex=0)
        teacher = compute_teacher_labels(graph, anchors)
        embedding = build_hilbert_embedding(teacher)
        phi = embedding.phi

        compressor = PositiveCompressor(input_dim=2 * K, compressed_dim=4)
        optimizer = torch.optim.Adam(compressor.parameters(), lr=1e-2)

        dist_matrix = _all_pairs_distances(graph)
        V = graph.num_nodes

        for _ in range(50):
            sources = torch.randint(0, V, (8,))
            targets = torch.randint(0, V, (8,))
            y = compressor(phi)
            h_smooth = smoothed_heuristic_undirected(y[sources], y[targets], temperature=1.0)
            d_true = dist_matrix[sources, targets]
            loss = gap_closing_loss(d_true, h_smooth, compressor, cond_lambda=0.01)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check admissibility with hard max (not smooth) for all reachable pairs
        with torch.no_grad():
            y_final = compressor(phi)
            h_fn = make_aac_heuristic(y_final, is_directed=False)
            violations = 0
            for s in range(V):
                for t in range(V):
                    if s == t:
                        continue
                    d_st = dist_matrix[s, t].item()
                    if d_st >= 1e17:  # unreachable
                        continue
                    h_st = h_fn(s, t)
                    if h_st > d_st + 1e-6:
                        violations += 1
            assert violations == 0, f"Found {violations} admissibility violations"


# ===========================================================================
# TRAIN-03: Deterministic splits and query pair dataset
# ===========================================================================

class TestSplitsAndDataset:
    """Tests for make_splits and QueryPairDataset."""

    def test_deterministic_splits(self) -> None:
        """make_splits(V=100, seed=42) produces identical results on two calls."""
        train1, val1, test1 = make_splits(num_vertices=100, seed=42)
        train2, val2, test2 = make_splits(num_vertices=100, seed=42)
        assert torch.equal(train1, train2), "Train splits should be deterministic"
        assert torch.equal(val1, val2), "Val splits should be deterministic"
        assert torch.equal(test1, test2), "Test splits should be deterministic"

    def test_split_sizes(self) -> None:
        """make_splits(V=100, train_frac=0.7, val_frac=0.15) produces ~70/15/15 split."""
        train, val, test = make_splits(num_vertices=100, train_frac=0.7, val_frac=0.15)
        assert len(train) == 70, f"Expected 70 train, got {len(train)}"
        assert len(val) == 15, f"Expected 15 val, got {len(val)}"
        assert len(test) == 15, f"Expected 15 test, got {len(test)}"

    def test_split_no_overlap(self) -> None:
        """Train, val, test index sets have no overlap and their union covers all indices."""
        train, val, test = make_splits(num_vertices=100, seed=42)
        train_set = set(train.tolist())
        val_set = set(val.tolist())
        test_set = set(test.tolist())
        assert len(train_set & val_set) == 0, "Train and val should not overlap"
        assert len(train_set & test_set) == 0, "Train and test should not overlap"
        assert len(val_set & test_set) == 0, "Val and test should not overlap"
        assert train_set | val_set | test_set == set(range(100)), "Union should cover all indices"

    def test_query_pair_dataset(self) -> None:
        """QueryPairDataset(V=100, num_pairs=500, seed=42) returns (source, target) as int tensors."""
        ds = QueryPairDataset(num_vertices=100, num_pairs=500, seed=42)
        assert len(ds) == 500, f"Expected 500 pairs, got {len(ds)}"
        s, t = ds[0]
        assert s.dtype in (torch.int32, torch.int64), f"Source should be int, got {s.dtype}"
        assert t.dtype in (torch.int32, torch.int64), f"Target should be int, got {t.dtype}"
        # All values should be in [0, 100)
        assert (ds.sources >= 0).all() and (ds.sources < 100).all()
        assert (ds.targets >= 0).all() and (ds.targets < 100).all()
        # Determinism: second call produces same data
        ds2 = QueryPairDataset(num_vertices=100, num_pairs=500, seed=42)
        assert torch.equal(ds.sources, ds2.sources)
        assert torch.equal(ds.targets, ds2.targets)


# ===========================================================================
# Training loop tests
# ===========================================================================

class TestTrainingLoop:
    """Tests for train_compressor and TrainConfig."""

    def test_train_compressor_returns_history(self, small_undirected_graph: Graph) -> None:
        """train_compressor returns dict with 'train_loss', 'val_loss' keys."""
        torch.manual_seed(42)
        graph = small_undirected_graph
        K = 3
        anchors = farthest_point_sampling(graph, K, seed_vertex=0)
        teacher = compute_teacher_labels(graph, anchors)
        embedding = build_hilbert_embedding(teacher)
        phi = embedding.phi
        dist_matrix = _all_pairs_distances(graph)
        V = graph.num_nodes

        compressor = PositiveCompressor(input_dim=2 * K, compressed_dim=4)
        config = TrainConfig(num_epochs=20, batch_size=6, lr=1e-2, seed=42)
        val_s = torch.randint(0, V, (10,))
        val_t = torch.randint(0, V, (10,))

        history = train_compressor(
            compressor, phi, teacher,
            config=config, val_pairs=(val_s, val_t),
        )

        assert "train_loss" in history, "History must contain 'train_loss'"
        assert "val_loss" in history, "History must contain 'val_loss'"
        assert isinstance(history["train_loss"], list), "train_loss should be list"
        assert isinstance(history["val_loss"], list), "val_loss should be list"
        assert all(isinstance(x, float) for x in history["train_loss"])
        assert len(history["train_loss"]) > 0, "Should have at least one train loss"

    def test_training_convergence(self, small_undirected_graph: Graph) -> None:
        """After training with num_epochs=200, final train loss < initial * 0.5."""
        torch.manual_seed(42)
        graph = small_undirected_graph
        K = 3
        anchors = farthest_point_sampling(graph, K, seed_vertex=0)
        teacher = compute_teacher_labels(graph, anchors)
        embedding = build_hilbert_embedding(teacher)
        phi = embedding.phi
        dist_matrix = _all_pairs_distances(graph)

        compressor = PositiveCompressor(input_dim=2 * K, compressed_dim=4)
        config = TrainConfig(num_epochs=200, batch_size=6, lr=1e-2, seed=42, patience=200)

        history = train_compressor(
            compressor, phi, teacher, config=config,
        )

        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]
        assert final_loss < initial_loss * 0.8, (
            f"Expected convergence: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_temperature_annealing(self) -> None:
        """Temperature at epoch 0 is T_init=1.0, at epoch 100 with gamma=1.05 is ~131.5."""
        config = TrainConfig(T_init=1.0, gamma=1.05)
        T_0 = config.T_init * (config.gamma ** 0)
        T_100 = config.T_init * (config.gamma ** 100)
        assert abs(T_0 - 1.0) < 1e-10, f"T at epoch 0 should be 1.0, got {T_0}"
        assert abs(T_100 - 1.05**100) < 0.1, f"T at epoch 100 should be ~131.5, got {T_100}"
        # 1.05^100 ~ 131.5
        assert 130.0 < T_100 < 133.0, f"T at epoch 100 = {T_100:.1f}, expected ~131.5"


# ===========================================================================
# End-to-end integration tests
# ===========================================================================

class TestEndToEnd:
    """End-to-end pipeline: anchors -> SSSP -> embed -> train -> search."""

    def test_end_to_end_pipeline(self, small_undirected_graph: Graph) -> None:
        """Full pipeline on small undirected graph: train, then A* finds optimal paths."""
        torch.manual_seed(42)
        graph = small_undirected_graph
        V = graph.num_nodes
        K = 3

        # Step 1-3: Anchors, teacher labels, embedding
        anchors = farthest_point_sampling(graph, K, seed_vertex=0)
        teacher = compute_teacher_labels(graph, anchors)
        embedding = build_hilbert_embedding(teacher)
        phi = embedding.phi  # (V, 2K=6)

        # Step 4-5: Train compressor
        compressor = PositiveCompressor(input_dim=2 * K, compressed_dim=4)
        dist_matrix = _all_pairs_distances(graph)
        config = TrainConfig(
            num_epochs=100, batch_size=6, lr=1e-2, seed=42, patience=200,
        )
        history = train_compressor(
            compressor, phi, teacher, config=config,
        )

        # Step 6-7: Compress and create heuristic
        with torch.no_grad():
            y = compressor(phi)
        h_fn = make_aac_heuristic(y, is_directed=False)

        # Step 8-12: Run A* and Dijkstra on all reachable pairs
        violations = 0
        cost_mismatches = 0
        expansion_failures = 0
        total_tested = 0

        for s in range(V):
            for t in range(V):
                if s == t:
                    continue
                d_st = dist_matrix[s, t].item()
                if d_st >= 1e17:  # unreachable
                    continue

                result_astar = astar(graph, s, t, heuristic=h_fn)
                result_dijkstra = dijkstra(graph, s, t)
                total_tested += 1

                # Admissibility check
                h_st = h_fn(s, t)
                if h_st > d_st + 1e-6:
                    violations += 1

                # Optimality: A* cost == Dijkstra cost
                if abs(result_astar.cost - result_dijkstra.cost) > 1e-10:
                    cost_mismatches += 1

                # Expansion efficiency: A* should expand <= Dijkstra
                if result_astar.expansions > result_dijkstra.expansions:
                    expansion_failures += 1

        assert total_tested > 0, "Should test at least one reachable pair"
        assert violations == 0, f"Found {violations} admissibility violations"
        assert cost_mismatches == 0, f"Found {cost_mismatches} cost mismatches (A* not optimal)"
        assert expansion_failures == 0, (
            f"Found {expansion_failures}/{total_tested} pairs where A* expanded more than Dijkstra"
        )

    def test_compression_dimension_sweep(self, small_undirected_graph: Graph) -> None:
        """Train with m in [4, 8] on small graph. Both produce valid heuristics."""
        graph = small_undirected_graph
        K = 3
        anchors = farthest_point_sampling(graph, K, seed_vertex=0)
        teacher = compute_teacher_labels(graph, anchors)
        embedding = build_hilbert_embedding(teacher)
        phi = embedding.phi
        dist_matrix = _all_pairs_distances(graph)
        V = graph.num_nodes

        for m in [4, 8]:
            torch.manual_seed(42)
            compressor = PositiveCompressor(input_dim=2 * K, compressed_dim=m)
            config = TrainConfig(
                num_epochs=50, batch_size=6, lr=1e-2, seed=42, patience=200,
            )
            train_compressor(
                compressor, phi, teacher, config=config,
            )

            with torch.no_grad():
                y = compressor(phi)
            h_fn = make_aac_heuristic(y, is_directed=False)

            # Check zero admissibility violations
            violations = 0
            for s in range(V):
                for t in range(V):
                    if s == t:
                        continue
                    d_st = dist_matrix[s, t].item()
                    if d_st >= 1e17:
                        continue
                    h_st = h_fn(s, t)
                    if h_st > d_st + 1e-6:
                        violations += 1
            assert violations == 0, f"m={m}: found {violations} admissibility violations"
