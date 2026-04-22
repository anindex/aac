"""End-to-end Contextual forward pass: encoder -> smooth BF -> compress.

Connects the neural edge cost encoder (WarcraftCNN or CabspottingMLP)
with smooth Bellman-Ford and LinearCompressor to produce differentiable
compressed labels via row-stochastic landmark selection.

For simplicity, processes one sample at a time (loop over batch).
The small grid sizes (144-324 nodes) make this acceptable.

IMPORTANT: Does NOT differentiate through A* search. The pipeline
differentiates through label construction only. At test time, A* runs
on predicted edge costs with the safe heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from aac.contextual.encoders import build_grid_edge_index, cell_costs_to_edge_weights
from aac.contextual.smooth_bf import graph_with_weights, smooth_bellman_ford_batched
from aac.graphs.types import Graph


@dataclass
class ContextualOutput:
    """Output of contextual_forward.

    Attributes:
        compressed_labels: (V, m) compressed label vectors.
        edge_costs: (B, E) predicted edge costs from encoder.
        soft_distances: (K, V) differentiable distances from smooth BF.
        phi: (V, K) raw distances for undirected or (V, 2K) [d_out, d_in] for directed.
        cell_costs: (B, V) predicted cell costs from encoder (before edge conversion).
            Only set for CNN encoders (grid_size is not None). None for MLP encoders.
    """

    compressed_labels: torch.Tensor
    edge_costs: torch.Tensor
    soft_distances: torch.Tensor
    phi: torch.Tensor
    cell_costs: torch.Tensor | None = None


def contextual_forward(
    encoder: nn.Module,
    compressor: nn.Module,
    context: torch.Tensor,
    graph_template: Graph,
    anchor_indices: torch.Tensor,
    beta: float = 10.0,
    compressed_dim: int = 8,
    is_directed: bool | None = None,
    grid_size: int | None = None,
    tau: float = 1.0,
) -> ContextualOutput:
    """End-to-end Contextual forward pass.

    Step 1: encoder(context) -> cell_costs (B, H*W) or edge_costs
    Step 2: cell_costs -> edge_weights via warcraft averaging
    Step 3: For each sample, build graph with predicted weights and run smooth BF
    Step 4: Transpose soft distances to (V, K)
    Step 5: compressor(d_out_t) -> compressed_labels (V, m)

    With LinearCompressor the Hilbert embedding step is skipped; distances
    are compressed directly via row-stochastic landmark selection.

    NOTE: Processes one sample at a time (loop over batch). Small grids
    (144-324 nodes) make this acceptable.

    Args:
        encoder: Neural edge cost encoder (WarcraftCNN or CabspottingMLP).
        compressor: Compressor module (LinearCompressor) for label compression.
        context: Input context tensor (images for CNN, features for MLP).
        graph_template: Graph with topology and placeholder weights.
        anchor_indices: (K,) tensor of anchor vertex indices.
        beta: Smooth BF inverse temperature.
        compressed_dim: Target compression dimension m.
        is_directed: Override directionality. If None, uses graph_template.is_directed.
        grid_size: Grid size for CNN encoder (inferred from graph if None).

    Returns:
        ContextualOutput with compressed labels, edge costs, distances, and raw distances.
    """
    if is_directed is None:
        is_directed = graph_template.is_directed

    V = graph_template.num_nodes

    # Step 1: Encoder produces costs
    raw_costs = encoder(context)  # (B, H*W) for CNN or (B, E) for MLP

    # Step 2: Convert cell costs to edge weights if needed (CNN case)
    # Detect grid_size from graph or encoder
    if grid_size is None and hasattr(encoder, "grid_size"):
        grid_size = encoder.grid_size

    _cell_costs_out = None
    if grid_size is not None:
        # CNN encoder: reshape flat output to (B, H, W) for cell_costs_to_edge_weights
        B = raw_costs.shape[0]
        _cell_costs_out = raw_costs  # (B, H*W) -- save for cost supervision
        cell_costs_2d = raw_costs.view(B, grid_size, grid_size)
        edge_costs = cell_costs_to_edge_weights(cell_costs_2d, grid_size)  # (B, num_template_edges)

        # Get the edge indices from build_grid_edge_index to build graphs
        src_idx, tgt_idx, _dist_factors = build_grid_edge_index(grid_size)
    else:
        # MLP encoder: costs are already per-edge
        edge_costs = raw_costs
        B = edge_costs.shape[0]
        src_idx = None  # use graph_template topology directly

    # Step 3: Process samples. We use the last sample's graph for embedding
    # (in practice, all samples share topology but have different weights).
    # For training, we typically aggregate losses across the batch.
    # Here we process the LAST sample to produce the output labels,
    # but return all edge_costs for batch-level loss computation.

    # NOTE: We average edge costs across the batch to produce a single
    # "representative" set of labels for the forward pass output. This is
    # intentional for memory reasons: computing per-sample labels would require
    # running smooth BF B times. In training, the loss is computed per-sample
    # via the smoothed heuristic on the shared compressed labels, so gradient
    # signal still flows correctly through the encoder's edge cost predictions.
    # For inference (B=1), this reduces to using the single sample's costs.
    mean_edge_costs = edge_costs.mean(dim=0)  # (E_template,)

    # Build graph with predicted weights
    # The graph_template may have different edge count than edge_costs
    # if is_directed differs from how build_grid_edge_index works
    if src_idx is not None:
        # Rebuild graph from grid edge indices with predicted weights.
        # build_grid_edge_index already emits BOTH directions for each edge,
        # so we must use is_directed=True to avoid doubling edges.
        from aac.graphs.convert import edges_to_graph

        predicted_graph = edges_to_graph(
            src_idx,
            tgt_idx,
            mean_edge_costs.to(torch.float64),
            num_nodes=V,
            is_directed=True,
        )
    else:
        predicted_graph = graph_with_weights(
            graph_template, mean_edge_costs.to(torch.float64)
        )

    # Step 4: Smooth BF from anchors
    soft_distances = smooth_bellman_ford_batched(
        predicted_graph, anchor_indices, beta=beta
    )  # (K, V)

    # Step 5: Compress distances directly via LinearCompressor
    # LinearCompressor internally operates in float64 (Gumbel-softmax cast),
    # so ensure input distances are float64 for matmul compatibility.

    if not is_directed:
        # Undirected: compress d_out_t directly -> (V, m)
        d_out_t = soft_distances.t().to(torch.float64)  # (V, K)
        compressed_labels = compressor(d_out_t, tau=tau)  # (V, m) single tensor
        phi = d_out_t  # store raw distances for reference
    else:
        # Directed: compute reverse distances, compress both
        from aac.graphs.convert import transpose_graph
        transposed = transpose_graph(predicted_graph)
        soft_distances_rev = smooth_bellman_ford_batched(
            transposed, anchor_indices, beta=beta
        )  # (K, V) - distances in reversed graph = d_in in original
        d_out_t = soft_distances.t().to(torch.float64)  # (V, K)
        d_in_t = soft_distances_rev.t().to(torch.float64)  # (V, K)
        result = compressor(d_out_t, d_in_t, tau=tau)  # (y_fwd, y_bwd) tuple
        if isinstance(result, tuple):
            y_fwd, y_bwd = result
            compressed_labels = torch.cat([y_fwd, y_bwd], dim=1)  # (V, m)
        else:
            compressed_labels = result
        phi = torch.cat([d_out_t, d_in_t], dim=1)  # (V, 2K) for reference

    return ContextualOutput(
        compressed_labels=compressed_labels,
        edge_costs=edge_costs,
        soft_distances=soft_distances,
        phi=phi,
        cell_costs=_cell_costs_out,
    )


def contextual_forward_mlp(
    encoder: nn.Module,
    compressor: nn.Module,
    features: torch.Tensor,
    graph_template: Graph,
    anchor_indices: torch.Tensor,
    beta: float = 10.0,
    compressed_dim: int = 8,
    is_directed: bool | None = None,
) -> ContextualOutput:
    """Contextual forward pass for MLP encoder (edge features directly).

    Similar to contextual_forward but skips cell-to-edge conversion.
    MLP produces edge costs directly from features.

    Args:
        encoder: CabspottingMLP or similar edge feature encoder.
        compressor: Compressor module (LinearCompressor) for label compression.
        features: (B, E, F) edge feature tensor.
        graph_template: Graph with topology.
        anchor_indices: (K,) anchor indices.
        beta: Smooth BF temperature.
        compressed_dim: Compressed label dimension m.
        is_directed: Override directionality.

    Returns:
        ContextualOutput.
    """
    return contextual_forward(
        encoder=encoder,
        compressor=compressor,
        context=features,
        graph_template=graph_template,
        anchor_indices=anchor_indices,
        beta=beta,
        compressed_dim=compressed_dim,
        is_directed=is_directed,
        grid_size=None,  # No grid conversion needed
    )
