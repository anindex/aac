"""Tests for Contextual: smooth Bellman-Ford, neural edge cost encoders, pipeline, and losses."""

from __future__ import annotations

import math

import torch
from torch.autograd import gradcheck

from aac.contextual.smooth_bf import graph_with_weights, smooth_bellman_ford_batched
from aac.graphs.convert import edges_to_graph
from aac.graphs.types import Graph


def _make_chain_graph(n: int = 4, weight: float = 1.0) -> Graph:
    """Build a directed chain: 0 -> 1 -> 2 -> ... -> n-1, all weights = weight."""
    sources = torch.arange(n - 1, dtype=torch.int64)
    targets = torch.arange(1, n, dtype=torch.int64)
    weights = torch.full((n - 1,), weight, dtype=torch.float64)
    return edges_to_graph(sources, targets, weights, num_nodes=n, is_directed=True)


def _make_5node_graph() -> Graph:
    """Build a 5-node directed graph with multiple paths.

    Edges: 0->1 (1.0), 0->2 (3.0), 1->2 (1.0), 1->3 (4.0), 2->3 (1.0), 3->4 (1.0)
    Shortest from 0: [0, 1, 2, 3, 4]
    """
    sources = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int64)
    targets = torch.tensor([1, 2, 2, 3, 3, 4], dtype=torch.int64)
    weights = torch.tensor([1.0, 3.0, 1.0, 4.0, 1.0, 1.0], dtype=torch.float64)
    return edges_to_graph(sources, targets, weights, num_nodes=5, is_directed=True)


def _make_grid_graph(rows: int, cols: int) -> Graph:
    """Build a directed 4-connected grid graph (rows x cols)."""
    sources = []
    targets = []
    weights = []
    num_nodes = rows * cols

    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            # Right neighbor
            if c + 1 < cols:
                sources.append(node)
                targets.append(node + 1)
                weights.append(1.0)
            # Down neighbor
            if r + 1 < rows:
                sources.append(node)
                targets.append(node + cols)
                weights.append(1.0)
            # Left neighbor
            if c - 1 >= 0:
                sources.append(node)
                targets.append(node - 1)
                weights.append(1.0)
            # Up neighbor
            if r - 1 >= 0:
                sources.append(node)
                targets.append(node - cols)
                weights.append(1.0)

    src_t = torch.tensor(sources, dtype=torch.int64)
    tgt_t = torch.tensor(targets, dtype=torch.int64)
    wgt_t = torch.tensor(weights, dtype=torch.float64)
    return edges_to_graph(src_t, tgt_t, wgt_t, num_nodes=num_nodes, is_directed=True)


class TestSmoothBF:
    """Tests for smooth_bellman_ford_batched."""

    def test_high_beta_matches_hard_bf(self) -> None:
        """Test 1: With high beta, smooth BF should approximate hard BF on a chain graph."""
        graph = _make_chain_graph(n=4, weight=1.0)
        source_indices = torch.tensor([0], dtype=torch.int64)
        expected = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float64)

        dist = smooth_bellman_ford_batched(graph, source_indices, beta=50.0)
        assert dist.shape == (1, 4)
        assert torch.allclose(dist, expected, atol=0.1), f"Got {dist}, expected close to {expected}"

    def test_low_beta_undershoots(self) -> None:
        """Test 2: With low beta, smooth-min should undershoot (return values <= hard min)."""
        graph = _make_5node_graph()
        source_indices = torch.tensor([0], dtype=torch.int64)

        dist_smooth = smooth_bellman_ford_batched(graph, source_indices, beta=1.0)

        from aac.embeddings.sssp import bellman_ford_batched

        dist_hard = bellman_ford_batched(graph, source_indices)

        # For nodes reachable by multiple paths, smooth-min undershoots hard min
        # (smooth_min <= hard_min because logsumexp(a, b) >= max(a, b))
        assert (dist_smooth <= dist_hard + 1e-10).all(), (
            f"Smooth should be <= hard but got smooth={dist_smooth}, hard={dist_hard}"
        )

    def test_gradcheck(self) -> None:
        """Test 3: Gradients flow through smooth BF back to edge weights."""
        # Build a small chain graph with differentiable weights
        n = 4
        sources = torch.arange(n - 1, dtype=torch.int64)
        targets = torch.arange(1, n, dtype=torch.int64)
        weights = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float64, requires_grad=True)

        graph = edges_to_graph(sources, targets, weights, num_nodes=n, is_directed=True)
        source_indices = torch.tensor([0], dtype=torch.int64)

        def fn(w: torch.Tensor) -> torch.Tensor:
            g = graph_with_weights(graph, w)
            return smooth_bellman_ford_batched(g, source_indices, beta=5.0)

        assert gradcheck(fn, (weights,), eps=1e-5, atol=1e-3), "Gradient check failed"

    def test_convergence_to_hard_bf(self) -> None:
        """Test 4: As beta increases, smooth BF output approaches hard BF output."""
        graph = _make_5node_graph()
        source_indices = torch.tensor([0], dtype=torch.int64)

        from aac.embeddings.sssp import bellman_ford_batched

        dist_hard = bellman_ford_batched(graph, source_indices)

        betas = [1.0, 5.0, 10.0, 50.0, 100.0]
        prev_error = float("inf")

        for beta in betas:
            dist_smooth = smooth_bellman_ford_batched(graph, source_indices, beta=beta)
            # Only compare reachable nodes
            mask = dist_hard < 1e17
            error = (dist_smooth[mask] - dist_hard[mask]).abs().max().item()
            assert error <= prev_error + 1e-10, (
                f"Error should decrease with beta: beta={beta}, error={error}, prev={prev_error}"
            )
            prev_error = error

        # At beta=100, should be very close to hard BF
        assert prev_error < 0.1, f"At beta=100, error should be <0.1 but got {prev_error}"

    def test_multiple_sources(self) -> None:
        """Test 5: Multiple sources produce correct shape and values."""
        graph = _make_5node_graph()
        source_indices = torch.tensor([0, 3], dtype=torch.int64)

        dist = smooth_bellman_ford_batched(graph, source_indices, beta=50.0)

        assert dist.shape == (2, 5), f"Expected (2, 5) but got {dist.shape}"

        # Source 0: distances should be finite for reachable nodes
        assert dist[0, 0].item() < 0.1, "Distance from 0 to 0 should be ~0"
        # Source 3: distance to node 4 should be ~1.0
        assert abs(dist[1, 4].item() - 1.0) < 0.1, f"Distance from 3 to 4 should be ~1.0, got {dist[1, 4]}"

    def test_12x12_grid(self) -> None:
        """Test 6: 12x12 grid graph completes without error and produces finite distances."""
        graph = _make_grid_graph(12, 12)
        source_indices = torch.tensor([0], dtype=torch.int64)

        dist = smooth_bellman_ford_batched(graph, source_indices, beta=10.0)

        assert dist.shape == (1, 144), f"Expected (1, 144) but got {dist.shape}"
        assert torch.isfinite(dist).all(), "All distances should be finite"
        assert dist[0, 0].item() < 0.1, "Distance from 0 to 0 should be ~0"
        # Corner-to-corner on 12x12 grid (Manhattan distance = 22)
        assert dist[0, 143].item() < 30.0, "Distance to far corner should be reasonable"


class TestEncoders:
    """Tests for WarcraftCNN, CabspottingMLP, and cell_costs_to_edge_weights."""

    def test_warcraft_cnn_output_shape_and_positivity(self) -> None:
        """Test 1: WarcraftCNN produces strictly positive (B, H*W) output."""
        from aac.contextual.encoders import WarcraftCNN

        model = WarcraftCNN(grid_size=12, img_size=96)
        x = torch.randn(2, 3, 96, 96)
        out = model(x)

        assert out.shape == (2, 12 * 12), f"Expected (2, 144) but got {out.shape}"
        assert (out > 0).all(), f"All outputs should be strictly positive, min={out.min()}"

    def test_warcraft_cnn_backward(self) -> None:
        """Test 2: Gradients flow back from WarcraftCNN output to input image."""
        from aac.contextual.encoders import WarcraftCNN

        model = WarcraftCNN(grid_size=12, img_size=96)
        x = torch.randn(2, 3, 96, 96, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Gradient should flow back to input"
        assert x.grad.shape == (2, 3, 96, 96)

    def test_cabspotting_mlp_output_shape_and_positivity(self) -> None:
        """Test 3: CabspottingMLP produces strictly positive (B, E) output."""
        from aac.contextual.encoders import CabspottingMLP

        model = CabspottingMLP(input_dim=6, hidden_dim=64)
        features = torch.randn(4, 100, 6)
        out = model(features)

        assert out.shape == (4, 100), f"Expected (4, 100) but got {out.shape}"
        assert (out > 0).all(), f"All outputs should be strictly positive, min={out.min()}"

    def test_cabspotting_mlp_backward(self) -> None:
        """Test 4: Gradients flow back from CabspottingMLP output to input features."""
        from aac.contextual.encoders import CabspottingMLP

        model = CabspottingMLP(input_dim=6, hidden_dim=64)
        features = torch.randn(4, 100, 6, requires_grad=True)
        out = model(features)
        loss = out.sum()
        loss.backward()

        assert features.grad is not None, "Gradient should flow back to input"
        assert features.grad.shape == (4, 100, 6)

    def test_cell_costs_to_edge_weights_shape_and_convention(self) -> None:
        """Test 5: cell_costs_to_edge_weights matches warcraft averaging convention."""
        from aac.contextual.encoders import build_grid_edge_index, cell_costs_to_edge_weights

        grid_size = 3
        # Uniform costs -> edge weight = 0.5*(1+1)*dist_factor = dist_factor
        cell_costs = torch.ones(1, grid_size, grid_size, dtype=torch.float64)
        sources, targets, dist_factors = build_grid_edge_index(grid_size)
        edge_weights = cell_costs_to_edge_weights(cell_costs, grid_size)

        num_edges = sources.shape[0]
        assert edge_weights.shape == (1, num_edges), f"Expected (1, {num_edges}) but got {edge_weights.shape}"

        # With uniform cost=1.0, edge weights should equal distance factors
        expected = dist_factors.unsqueeze(0).to(edge_weights.dtype)
        assert torch.allclose(edge_weights, expected, atol=1e-10), (
            "With uniform costs, weights should equal dist_factors"
        )

    def test_cell_costs_to_edge_weights_differentiable(self) -> None:
        """Test 6: cell_costs_to_edge_weights is differentiable."""
        from aac.contextual.encoders import cell_costs_to_edge_weights

        grid_size = 3
        cell_costs = torch.ones(2, grid_size, grid_size, dtype=torch.float64, requires_grad=True)
        edge_weights = cell_costs_to_edge_weights(cell_costs, grid_size)
        loss = edge_weights.sum()
        loss.backward()

        assert cell_costs.grad is not None, "Gradient should flow to cell_costs"
        assert cell_costs.grad.shape == (2, grid_size, grid_size)


class TestLoss:
    """Tests for path_kl_loss, cost_regret_loss, and contextual_loss."""

    def test_path_kl_loss_positive_scalar_with_gradients(self) -> None:
        """Test 1: path_kl_loss with uniform predicted and one-hot target returns positive scalar with gradients."""
        from aac.contextual.loss import path_kl_loss

        E = 10
        B = 4
        # Uniform predicted log-probs: log(1/E) for all edges
        predicted_logprobs = torch.full(
            (B, E), math.log(1.0 / E), dtype=torch.float64, requires_grad=True
        )
        # One-hot ground truth: only edge 3 is in the shortest path
        target = torch.zeros(B, E, dtype=torch.float64)
        target[:, 3] = 1.0

        loss = path_kl_loss(predicted_logprobs, target)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

        # Check gradients flow
        loss.backward()
        assert predicted_logprobs.grad is not None, "Gradients should flow to predicted_logprobs"

    def test_cost_regret_loss(self) -> None:
        """Test 2: cost_regret_loss with predicted_costs=1.1, optimal_costs=1.0 returns ~0.1."""
        from aac.contextual.loss import cost_regret_loss

        predicted = torch.tensor([1.1, 1.2, 1.05], dtype=torch.float64)
        optimal = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        loss = cost_regret_loss(predicted, optimal)
        # Expected: mean((1.1-1.0)/1.0, (1.2-1.0)/1.0, (1.05-1.0)/1.0) = mean(0.1, 0.2, 0.05) = 0.1167
        expected = ((0.1 + 0.2 + 0.05) / 3)
        assert abs(loss.item() - expected) < 1e-6, f"Expected ~{expected}, got {loss.item()}"


class TestPipeline:
    """Tests for contextual_forward end-to-end pipeline."""

    def test_forward_shapes_3x3_grid(self) -> None:
        """Test 3: contextual_forward on 3x3 grid produces correct output shapes."""
        from aac.compression.compressor import LinearCompressor
        from aac.contextual.encoders import WarcraftCNN, build_grid_edge_index
        from aac.contextual.pipeline import ContextualOutput, contextual_forward

        grid_size = 3
        B = 2
        K = 2
        m = 4

        encoder = WarcraftCNN(grid_size=grid_size, img_size=24)
        compressor = LinearCompressor(K=K, m=m, is_directed=False)
        images = torch.randn(B, 3, 24, 24)

        # Build a template graph for the grid topology
        src, tgt, dist_factors = build_grid_edge_index(grid_size)
        dummy_weights = torch.ones(src.shape[0], dtype=torch.float64)
        template_graph = edges_to_graph(
            src, tgt, dummy_weights,
            num_nodes=grid_size * grid_size,
            is_directed=True,
        )
        anchor_indices = torch.tensor([0, grid_size * grid_size - 1], dtype=torch.int64)

        output = contextual_forward(
            encoder=encoder,
            compressor=compressor,
            context=images,
            graph_template=template_graph,
            anchor_indices=anchor_indices,
            beta=5.0,
            compressed_dim=m,
            is_directed=False,
        )

        assert isinstance(output, ContextualOutput)
        V = grid_size * grid_size  # 9
        num_edges = src.shape[0]
        assert output.compressed_labels.shape == (V, m), (
            f"Expected ({V}, {m}), got {output.compressed_labels.shape}"
        )
        assert output.edge_costs.shape[0] == B, (
            f"Expected batch dim {B}, got {output.edge_costs.shape[0]}"
        )
        assert output.edge_costs.shape[1] == num_edges, (
            f"Expected {num_edges} edges, got {output.edge_costs.shape[1]}"
        )

    def test_end_to_end_gradient_flow(self) -> None:
        """Test 4: Gradients from compressed_labels flow back to CNN parameters."""
        from aac.compression.compressor import LinearCompressor
        from aac.contextual.encoders import WarcraftCNN, build_grid_edge_index
        from aac.contextual.pipeline import contextual_forward

        grid_size = 3
        B = 1
        K = 2
        m = 4

        encoder = WarcraftCNN(grid_size=grid_size, img_size=24)
        compressor = LinearCompressor(K=K, m=m, is_directed=False)
        images = torch.randn(B, 3, 24, 24)

        src, tgt, dist_factors = build_grid_edge_index(grid_size)
        dummy_weights = torch.ones(src.shape[0], dtype=torch.float64)
        template_graph = edges_to_graph(
            src, tgt, dummy_weights,
            num_nodes=grid_size * grid_size,
            is_directed=True,
        )
        anchor_indices = torch.tensor([0, grid_size * grid_size - 1], dtype=torch.int64)

        output = contextual_forward(
            encoder=encoder,
            compressor=compressor,
            context=images,
            graph_template=template_graph,
            anchor_indices=anchor_indices,
            beta=5.0,
            compressed_dim=m,
            is_directed=False,
        )

        # Compute a scalar loss from compressed_labels
        loss = output.compressed_labels.sum()
        loss.backward()

        # Check that encoder CNN parameters have gradients
        has_grad = False
        for p in encoder.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Gradients should flow from compressed_labels back to encoder parameters"

    def test_forward_stores_raw_distances(self) -> None:
        """Test 5: contextual_forward with is_directed=False stores raw distances (V, K) in phi."""
        from aac.compression.compressor import LinearCompressor
        from aac.contextual.encoders import WarcraftCNN, build_grid_edge_index
        from aac.contextual.pipeline import contextual_forward

        grid_size = 3
        B = 1
        K = 2
        m = 4

        encoder = WarcraftCNN(grid_size=grid_size, img_size=24)
        compressor = LinearCompressor(K=K, m=m, is_directed=False)
        images = torch.randn(B, 3, 24, 24)

        # Build a template graph (build_grid_edge_index emits both directions)
        src, tgt, dist_factors = build_grid_edge_index(grid_size)
        dummy_weights = torch.ones(src.shape[0], dtype=torch.float64)
        template_graph = edges_to_graph(
            src, tgt, dummy_weights,
            num_nodes=grid_size * grid_size,
            is_directed=True,
        )
        anchor_indices = torch.tensor([0, grid_size * grid_size - 1], dtype=torch.int64)

        output = contextual_forward(
            encoder=encoder,
            compressor=compressor,
            context=images,
            graph_template=template_graph,
            anchor_indices=anchor_indices,
            beta=5.0,
            compressed_dim=m,
            is_directed=False,
        )

        V = grid_size * grid_size
        assert output.phi.shape == (V, K), (
            f"Raw distances should be (V, K)=({V}, {K}), got {output.phi.shape}"
        )
        assert output.compressed_labels.shape == (V, m)


class TestContextualTraining:
    """Tests for ContextualTrainer, ContextualConfig, ContextualMetrics."""

    def _make_training_fixtures(self):
        """Create small 3x3 grid training fixtures with synthetic ground truth."""
        import scipy.sparse
        import scipy.sparse.csgraph

        from aac.compression.compressor import LinearCompressor
        from aac.contextual.encoders import WarcraftCNN, build_grid_edge_index

        grid_size = 3
        K = 2
        m = 4
        V = grid_size * grid_size

        encoder = WarcraftCNN(grid_size=grid_size, img_size=24)
        compressor = LinearCompressor(K=K, m=m, is_directed=False)

        # Build template graph (directed)
        src, tgt, dist_factors = build_grid_edge_index(grid_size)
        dummy_weights = torch.ones(src.shape[0], dtype=torch.float64)
        template_graph = edges_to_graph(
            src, tgt, dummy_weights,
            num_nodes=V,
            is_directed=True,
        )
        anchor_indices = torch.tensor([0, V - 1], dtype=torch.int64)

        # Synthetic ground truth distances from SciPy Dijkstra on uniform cost grid
        from aac.graphs.convert import graph_to_scipy

        scipy_csr = graph_to_scipy(template_graph)
        gt_distances_np = scipy.sparse.csgraph.dijkstra(scipy_csr)
        gt_distances = torch.tensor(gt_distances_np, dtype=torch.float64)

        # Create synthetic training data: random images with ground truth distances
        torch.manual_seed(42)
        train_data = []
        for _ in range(8):
            img = torch.randn(3, 24, 24)
            train_data.append((img, gt_distances))

        return {
            "encoder": encoder,
            "compressor": compressor,
            "template_graph": template_graph,
            "anchor_indices": anchor_indices,
            "train_data": train_data,
            "gt_distances": gt_distances,
            "grid_size": grid_size,
            "K": K,
            "m": m,
        }

    def test_config_has_expected_fields(self) -> None:
        """Test 1: ContextualConfig has all expected fields."""
        from aac.contextual.trainer import ContextualConfig

        config = ContextualConfig()
        assert hasattr(config, "num_epochs")
        assert hasattr(config, "batch_size")
        assert hasattr(config, "lr")
        assert hasattr(config, "beta_init")
        assert hasattr(config, "beta_max")
        assert hasattr(config, "alpha_path")
        assert hasattr(config, "cond_lambda")
        assert hasattr(config, "beta_gamma")
        assert hasattr(config, "T_init")
        assert hasattr(config, "T_gamma")
        assert hasattr(config, "patience")
        assert hasattr(config, "K")
        assert hasattr(config, "m")

    def test_training_decreases_loss(self) -> None:
        """Test 2: Training on 3x3 grid with random CNN decreases loss over 80 epochs.

        Uses a single fixed training sample to eliminate inter-sample noise.
        Keeps beta and T constant to isolate gradient descent effect.
        Uses a fixed evaluation batch to avoid resampling noise dominating the
        per-epoch loss signal, and a wider window (first/last 20 epochs) to
        smooth Gumbel-softmax noise from LinearCompressor.
        """
        from aac.contextual.trainer import ContextualConfig, ContextualTrainer

        fixtures = self._make_training_fixtures()

        # Use only 1 fixed sample to eliminate stochastic sampling noise
        fixed_data = [fixtures["train_data"][0]]

        config = ContextualConfig(
            num_epochs=80,
            batch_size=1,
            lr=1e-2,
            beta_init=1.0,
            beta_max=1.0,  # keep constant
            beta_gamma=1.0,  # no annealing
            T_init=1.0,
            T_gamma=1.0,  # no annealing
            K=fixtures["K"],
            m=fixtures["m"],
            patience=200,  # disable early stopping
            seed=42,
        )

        trainer = ContextualTrainer(
            encoder=fixtures["encoder"],
            compressor=fixtures["compressor"],
            config=config,
        )

        metrics = trainer.train(
            train_data=fixed_data,
            graph_template=fixtures["template_graph"],
            anchor_indices=fixtures["anchor_indices"],
        )

        losses = metrics.per_epoch_loss
        assert len(losses) >= 60, f"Expected at least 60 epochs, got {len(losses)}"
        # Compare first 20 vs last 20 average loss (wider window for Gumbel noise)
        first_avg = sum(losses[:20]) / 20
        last_avg = sum(losses[-20:]) / 20
        assert last_avg < first_avg, (
            f"Loss should decrease: first 20 avg={first_avg:.6f}, last 20 avg={last_avg:.6f}"
        )

    def test_metrics_fields(self) -> None:
        """Test 3: ContextualMetrics contains expected fields."""
        from aac.contextual.trainer import ContextualConfig, ContextualMetrics, ContextualTrainer

        fixtures = self._make_training_fixtures()
        config = ContextualConfig(
            num_epochs=3,
            batch_size=4,
            K=fixtures["K"],
            m=fixtures["m"],
            seed=42,
        )

        trainer = ContextualTrainer(
            encoder=fixtures["encoder"],
            compressor=fixtures["compressor"],
            config=config,
        )

        metrics = trainer.train(
            train_data=fixtures["train_data"],
            graph_template=fixtures["template_graph"],
            anchor_indices=fixtures["anchor_indices"],
        )

        assert isinstance(metrics, ContextualMetrics)
        assert hasattr(metrics, "per_epoch_loss")
        assert hasattr(metrics, "per_epoch_time_sec")
        assert hasattr(metrics, "peak_memory_bytes")
        assert hasattr(metrics, "total_time_sec")
        assert len(metrics.per_epoch_loss) == 3
        assert len(metrics.per_epoch_time_sec) == 3
        assert all(t > 0 for t in metrics.per_epoch_time_sec), "Each epoch should take >0 seconds"
        assert metrics.total_time_sec > 0, "Total time should be positive"
        assert isinstance(metrics.peak_memory_bytes, int), "peak_memory_bytes should be int"

    def test_training_result_structure(self) -> None:
        """Test 4: Training returns metrics with 'per_epoch_loss' (list), 'total_time_sec' (float), 'peak_memory_bytes' (int)."""
        from aac.contextual.trainer import ContextualConfig, ContextualTrainer

        fixtures = self._make_training_fixtures()
        config = ContextualConfig(
            num_epochs=2,
            batch_size=4,
            K=fixtures["K"],
            m=fixtures["m"],
            seed=42,
        )

        trainer = ContextualTrainer(
            encoder=fixtures["encoder"],
            compressor=fixtures["compressor"],
            config=config,
        )

        metrics = trainer.train(
            train_data=fixtures["train_data"],
            graph_template=fixtures["template_graph"],
            anchor_indices=fixtures["anchor_indices"],
        )

        assert isinstance(metrics.per_epoch_loss, list)
        assert all(isinstance(x, float) for x in metrics.per_epoch_loss)
        assert isinstance(metrics.total_time_sec, float)
        assert isinstance(metrics.peak_memory_bytes, int)

    def test_beta_annealing(self) -> None:
        """Test 5: Beta increases from beta_init toward beta_max over training epochs."""
        from aac.contextual.trainer import ContextualConfig, ContextualTrainer

        fixtures = self._make_training_fixtures()
        config = ContextualConfig(
            num_epochs=10,
            batch_size=4,
            beta_init=1.0,
            beta_max=30.0,
            beta_gamma=1.2,
            K=fixtures["K"],
            m=fixtures["m"],
            seed=42,
        )

        trainer = ContextualTrainer(
            encoder=fixtures["encoder"],
            compressor=fixtures["compressor"],
            config=config,
        )

        metrics = trainer.train(
            train_data=fixtures["train_data"],
            graph_template=fixtures["template_graph"],
            anchor_indices=fixtures["anchor_indices"],
        )

        # final_beta should be > beta_init since gamma > 1
        assert metrics.final_beta > config.beta_init, (
            f"Beta should increase from {config.beta_init}, got final_beta={metrics.final_beta}"
        )
        # final_beta should not exceed beta_max
        assert metrics.final_beta <= config.beta_max + 1e-6, (
            f"Beta should not exceed {config.beta_max}, got {metrics.final_beta}"
        )
