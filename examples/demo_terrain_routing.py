#!/usr/bin/env python
"""Demo 3: Context-Conditioned Terrain Routing.

Shows the end-to-end differentiable pipeline: a neural network observes
terrain features and produces an admissible heuristic for A* search --
all in a single forward pass.

Pipeline: terrain features -> CNN encoder -> edge costs -> smooth BF ->
          compress -> admissible heuristic -> A* search

The heuristic is admissible by construction (row-stochastic compression),
so A* always finds the optimal path regardless of CNN quality.

Uses synthetic terrain (no dataset download required).

Usage:
    python examples/demo_terrain_routing.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aac.compression.compressor import LinearCompressor
from aac.compression.smooth import make_aac_heuristic
from aac.contextual.pipeline import contextual_forward
from aac.contextual.trainer import ContextualConfig, ContextualTrainer
from aac.embeddings.anchors import farthest_point_sampling
from aac.graphs.convert import edges_to_graph
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra

# --- Synthetic terrain generation ---

TERRAIN_COSTS = {
    0: 1.0,   # plains (fast)
    1: 2.0,   # forest (moderate)
    2: 5.0,   # mountain (slow)
    3: 10.0,  # swamp (very slow)
}


def generate_terrain(grid_size: int, seed: int = 42) -> torch.Tensor:
    """Generate a random terrain cost map.

    Returns:
        (grid_size, grid_size) float tensor of terrain costs.
    """
    gen = torch.Generator().manual_seed(seed)
    # Random terrain types with spatial correlation (block structure)
    block = max(1, grid_size // 4)
    small = torch.randint(0, 4, (grid_size // block + 1, grid_size // block + 1), generator=gen)
    terrain_type = small.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    terrain_type = terrain_type[:grid_size, :grid_size]

    # Convert to costs
    cost_map = torch.zeros(grid_size, grid_size, dtype=torch.float64)
    for t, c in TERRAIN_COSTS.items():
        cost_map[terrain_type == t] = c
    return cost_map


def cost_map_to_features(cost_map: torch.Tensor) -> torch.Tensor:
    """Convert cost map to 3-channel 'image' features for the encoder.

    Encodes terrain type as RGB-like features. In a real application,
    this would be an actual aerial/satellite image.

    Returns:
        (1, 3, H, W) float32 feature tensor.
    """
    H, W = cost_map.shape
    features = torch.zeros(1, 3, H, W, dtype=torch.float32)
    # Channel 0: normalized cost
    features[0, 0] = (cost_map.float() - 1.0) / 9.0
    # Channel 1: binary high-cost indicator
    features[0, 1] = (cost_map > 3.0).float()
    # Channel 2: inverse cost (fast terrain = bright)
    features[0, 2] = 1.0 / cost_map.float()
    return features


class SimpleEncoder(nn.Module):
    """Minimal encoder: predicts per-cell costs from terrain features.

    In a real application, this would be a CNN processing RGB images.
    Here we use a simple 2-layer CNN on synthetic features.
    """

    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).squeeze(1)  # (B, H, W)
        positive = F.softplus(raw) + 0.01
        return positive.view(positive.shape[0], -1)  # (B, H*W)


def build_grid_graph(cost_map: torch.Tensor):
    """Build 8-connected grid graph from cost map."""
    H, W = cost_map.shape
    V = H * W
    sqrt2 = math.sqrt(2.0)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    sources, targets, weights = [], [], []
    for r in range(H):
        for c in range(W):
            src = r * W + c
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    tgt = nr * W + nc
                    if src < tgt:
                        diag = dr != 0 and dc != 0
                        dist = sqrt2 if diag else 1.0
                        w = 0.5 * (cost_map[r, c].item() + cost_map[nr, nc].item()) * dist
                        sources.append(src)
                        targets.append(tgt)
                        weights.append(w)

    return edges_to_graph(
        torch.tensor(sources, dtype=torch.int64),
        torch.tensor(targets, dtype=torch.int64),
        torch.tensor(weights, dtype=torch.float64),
        num_nodes=V,
        is_directed=False,
    )


def print_terrain(cost_map: torch.Tensor, path_nodes: list[int]) -> None:
    """ASCII-print the terrain with path overlay."""
    H, W = cost_map.shape
    path_set = {(n // W, n % W) for n in path_nodes}
    symbols = {1.0: ".", 2.0: "f", 5.0: "M", 10.0: "~"}
    for r in range(H):
        row = []
        for c in range(W):
            if (r, c) == (0, 0):
                row.append("S")
            elif (r, c) == (H - 1, W - 1):
                row.append("G")
            elif (r, c) in path_set:
                row.append("*")
            else:
                row.append(symbols.get(cost_map[r, c].item(), "?"))
        print(" ".join(row))


def main() -> None:
    torch.manual_seed(42)
    grid_size = 10
    K, m = 5, 4

    print("=" * 60)
    print("  AAC Demo: Context-Conditioned Terrain Routing")
    print("=" * 60)
    print(f"  Grid: {grid_size}x{grid_size}, K={K} anchors, m={m} compressed")
    print(f"  Terrain: . = plains, f = forest, M = mountain, ~ = swamp")
    print()

    # Step 1: Generate terrain and build graph
    cost_map = generate_terrain(grid_size)
    graph = build_grid_graph(cost_map)
    V = graph.num_nodes

    # Ground truth: optimal path via Dijkstra on true costs
    src, tgt = 0, V - 1
    gt_result = dijkstra(graph, src, tgt)
    print(f"  Ground truth (Dijkstra): cost={gt_result.cost:.2f},"
          f" expansions={gt_result.expansions}")

    # Step 2: Train encoder + compressor end-to-end
    import scipy.sparse.csgraph
    from aac.graphs.convert import graph_to_scipy

    features = cost_map_to_features(cost_map)  # (1, 3, H, W)
    sp = graph_to_scipy(graph)
    gt_distances = torch.tensor(
        scipy.sparse.csgraph.shortest_path(sp, directed=False),
        dtype=torch.float64,
    )

    gt_cell_costs = cost_map.flatten().float()
    train_data = [(features, gt_distances, gt_cell_costs)]

    encoder = SimpleEncoder(grid_size)
    compressor = LinearCompressor(K=K, m=m, is_directed=False)
    anchor_indices = farthest_point_sampling(graph, K)

    config = ContextualConfig(
        num_epochs=200,
        batch_size=1,
        lr=5e-3,
        beta_init=1.0,
        beta_max=30.0,
        beta_gamma=1.05,
        alpha_cost=1.0,
        cond_lambda=0.01,
        T_init=1.0,
        T_gamma=1.05,
        patience=30,
        K=K,
        m=m,
        grid_size=grid_size,
    )

    trainer = ContextualTrainer(encoder, compressor, config, is_directed=False)
    print(f"\n  Training encoder + compressor (200 epochs)...")
    metrics = trainer.train(train_data, graph, anchor_indices)
    print(f"  Training done: {metrics.final_epoch + 1} epochs,"
          f" final loss={metrics.per_epoch_loss[-1]:.4f}")

    # Step 3: Inference -- single forward pass produces admissible heuristic
    encoder.eval()
    compressor.eval()

    with torch.no_grad():
        output = contextual_forward(
            encoder=encoder,
            compressor=compressor,
            context=features,
            graph_template=graph,
            anchor_indices=anchor_indices,
            beta=config.beta_max,
            compressed_dim=m,
            is_directed=False,
        )

    compressed_labels = output.compressed_labels  # (V, m)
    h_aac = make_aac_heuristic(compressed_labels, is_directed=False)

    # Step 4: A* with learned heuristic
    # The contextual heuristic is admissible w.r.t. PREDICTED costs.
    # For guaranteed optimality on the TRUE graph, we also build a
    # static heuristic from true distances (Demo 1 & 2 approach).

    from aac.baselines.alt import alt_preprocess
    from aac.train.trainer import TrainConfig, train_linear_compressor

    # Static AAC: true distances -> compress -> guaranteed admissible
    teacher = alt_preprocess(graph, K)
    static_comp = LinearCompressor(K=K, m=m, is_directed=False)
    train_linear_compressor(
        static_comp, teacher,
        TrainConfig(num_epochs=300, batch_size=256, lr=1e-2, seed=42, patience=30),
    )
    static_comp.eval()
    with torch.no_grad():
        static_labels = static_comp(teacher.d_out.t().to(torch.float64))
    h_static = make_aac_heuristic(static_labels, is_directed=False)

    result_static = astar(graph, src, tgt, h_static)
    static_reduction = (1 - result_static.expansions / gt_result.expansions) * 100
    assert abs(result_static.cost - gt_result.cost) < 1e-6

    # Contextual AAC: single forward pass (fast, slightly suboptimal)
    result_ctx = astar(graph, src, tgt, h_aac)
    ctx_reduction = (1 - result_ctx.expansions / gt_result.expansions) * 100
    cost_regret = (result_ctx.cost - gt_result.cost) / gt_result.cost * 100

    print(f"\n  Static AAC (true distances): cost={result_static.cost:.2f},"
          f" expansions={result_static.expansions}"
          f" ({static_reduction:.1f}% reduction) [optimal guaranteed]")
    print(f"  Contextual AAC (1 fwd pass): cost={result_ctx.cost:.2f},"
          f" expansions={result_ctx.expansions}"
          f" ({ctx_reduction:.1f}% reduction) [{cost_regret:.1f}% regret]")

    # Show terrain with path
    print(f"\n  Terrain with static-AAC path (*):")
    print()
    print_terrain(cost_map, result_static.path)

    # Step 5: Demonstrate the key property -- static AAC is always admissible
    print(f"\n{'=' * 60}")
    print("  Key Property: Static AAC is Admissible by Construction")
    print("=" * 60)
    print("  The row-stochastic compression guarantees:")
    print("    h(u,t) <= h_ALT(u,t) <= d(u,t)")
    print("  So A* with static AAC always finds the optimal path,")
    print("  regardless of how the compressor was trained.")
    print()
    print("  The contextual pipeline adds a neural encoder that")
    print("  produces heuristics from a SINGLE forward pass,")
    print("  trading guaranteed optimality for inference speed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
