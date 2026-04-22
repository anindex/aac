"""Track 3: Warcraft Contextual experiment runner with DataSP comparison.

Uses real Pogancic et al. (ICLR 2020) Warcraft shortest path dataset:
- 10K train / 1K val / 1K test, 12x12 grids with 96x96 RGB images
- 5 terrain types with costs [0.8, 1.2, 5.3, 7.7, 9.2]
- Ground-truth shortest paths provided as binary masks

Trains Contextual (encoder + compressor) end-to-end on real terrain maps,
evaluates with METR-07 path accuracy metrics (path match, Jaccard, cost regret),
and runs DataSP baseline via subprocess on the same data splits (EXP-03).
"""

from __future__ import annotations

import csv
import logging
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from experiments.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class WarcraftRunner(BaseRunner):
    """Track 3: Warcraft Contextual with DataSP comparison.

    EXP-03: AAC Contextual vs DataSP on real Warcraft terrain maps.
    METR-07: Path match rate, Jaccard overlap, cost regret.
    METR-08: Training time and peak GPU memory.

    Supported methods: contextual, datasp, dijkstra.
    """

    SUPPORTED_METHODS = ["contextual", "datasp", "dijkstra"]

    def run(self, cfg: DictConfig) -> None:
        """Execute Warcraft track experiment.

        Loads real Pogancic terrain data, trains Contextual (or runs
        DataSP/Dijkstra baseline), evaluates path accuracy, and writes results.

        Args:
            cfg: Hydra-composed configuration with track, method, and training fields.
        """
        from aac.graphs.loaders.warcraft import load_warcraft_dataset

        output_dir = Path(cfg.output_dir) / "warcraft"
        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = cfg.track.data_dir
        grid_size = cfg.track.grid_size

        # Load real Pogancic dataset (pre-split train/val/test)
        try:
            dataset = load_warcraft_dataset(data_dir, grid_size)
        except FileNotFoundError as e:
            logger.warning("Warcraft data not found: %s", e)
            return

        method_name = cfg.method.name

        if method_name == "contextual":
            results = self._run_contextual(cfg, dataset, grid_size)
        elif method_name == "datasp":
            results = self._run_datasp(cfg, data_dir, grid_size)
        elif method_name == "dijkstra":
            results = self._run_dijkstra(dataset, grid_size)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Write results CSV
        if results:
            from experiments.reporting.csv_writer import write_csv_metadata

            csv_path = output_dir / f"warcraft_{method_name}.csv"
            fieldnames = ["sample_idx", "method", "path_match", "jaccard",
                          "cost_regret", "expansions", "dijkstra_expansions"]
            with open(csv_path, "w", newline="") as f:
                write_csv_metadata(f)
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
            logger.info("Results written to %s", csv_path)

            # Aggregate metrics for TensorBoard
            matches = [r["path_match"] for r in results]
            jaccards = [r["jaccard"] for r in results]
            regrets = [r["cost_regret"] for r in results
                       if not np.isnan(r["cost_regret"])]
            expansions = [r.get("expansions", 0) for r in results
                          if r.get("expansions") is not None]
            dij_expansions = [r.get("dijkstra_expansions", 0) for r in results
                              if r.get("dijkstra_expansions") is not None]

            summary = {
                "path_match_rate": float(np.mean(matches)) if matches else 0.0,
                "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
                "mean_cost_regret": float(np.mean(regrets)) if regrets else 0.0,
                "num_test_maps": len(results),
            }
            if expansions and dij_expansions:
                mean_exp = float(np.mean(expansions))
                mean_dij = float(np.mean(dij_expansions))
                summary["mean_expansions"] = mean_exp
                summary["mean_dijkstra_expansions"] = mean_dij
                if mean_dij > 0:
                    summary["expansion_reduction_pct"] = (
                        (1.0 - mean_exp / mean_dij) * 100
                    )
            self.log_summary(f"warcraft/{method_name}", summary)

    def _run_contextual(
        self,
        cfg: DictConfig,
        dataset: dict,
        grid_size: int,
    ) -> list[dict]:
        """Train Contextual and evaluate on test maps using real Pogancic data.

        Args:
            cfg: Full configuration.
            dataset: Dict with 'train', 'val', 'test' splits from load_warcraft_dataset.
            grid_size: Grid dimension (H = W).

        Returns:
            List of per-map result dicts with path metrics.
        """
        import scipy.sparse.csgraph

        from aac.compression.compressor import LinearCompressor
        from aac.contextual.encoders import WarcraftCNN
        from aac.contextual.trainer import ContextualConfig, ContextualTrainer
        from aac.embeddings.anchors import farthest_point_sampling
        from aac.graphs.convert import graph_to_scipy
        from aac.graphs.loaders.warcraft import build_warcraft_graph
        from aac.search.dijkstra import dijkstra

        K = cfg.method.K
        m = cfg.method.m

        train_maps = dataset["train"]["maps"]      # (N, 96, 96, 3) uint8
        train_weights = dataset["train"]["weights"]  # (N, H, W) float64

        # Build training data: (context_image, gt_distances, gt_cell_costs)
        train_data: list[tuple[torch.Tensor, ...]] = []
        first_graph = None

        for i in range(len(train_weights)):
            graph, _ = build_warcraft_graph(train_weights[i])
            if first_graph is None:
                first_graph = graph

            # Compute ground-truth all-pairs distances
            sp = graph_to_scipy(graph)
            dist_matrix = scipy.sparse.csgraph.shortest_path(sp, directed=False)
            gt_dist = torch.tensor(dist_matrix, dtype=torch.float64)

            # Use real RGB image as context (normalized to [0, 1])
            # Include batch dim: (3, H, W) -> (1, 3, H, W)
            rgb = torch.from_numpy(train_maps[i]).permute(2, 0, 1).float() / 255.0

            # GT cell costs (flat) for cost supervision
            gt_costs = torch.tensor(
                train_weights[i].flatten(), dtype=torch.float32
            )
            train_data.append((rgb.unsqueeze(0), gt_dist, gt_costs))

        if first_graph is None:
            logger.warning("No training maps loaded")
            return []

        # Select anchors on first training graph
        anchor_indices = farthest_point_sampling(first_graph, K)

        # Create encoder and compressor
        img_size = cfg.track.get("img_size", 96)
        encoder = WarcraftCNN(grid_size=grid_size, img_size=img_size)
        compressor = LinearCompressor(K=K, m=m, is_directed=False)

        # Configure trainer
        train_cfg_dict = cfg.method.get("training", {})
        vb_config = ContextualConfig(
            num_epochs=train_cfg_dict.get("num_epochs", 100),
            batch_size=train_cfg_dict.get("batch_size", 24),
            lr=train_cfg_dict.get("lr", 1e-3),
            beta_init=train_cfg_dict.get("beta_init", 1.0),
            beta_max=train_cfg_dict.get("beta_max", 30.0),
            beta_gamma=train_cfg_dict.get("beta_gamma", 1.05),
            alpha_cost=train_cfg_dict.get("alpha_cost", 1.0),
            cond_lambda=train_cfg_dict.get("cond_lambda", 0.01),
            T_init=train_cfg_dict.get("T_init", 1.0),
            T_gamma=train_cfg_dict.get("T_gamma", 1.05),
            patience=train_cfg_dict.get("patience", 15),
            K=K,
            m=m,
            grid_size=grid_size,
        )

        trainer = ContextualTrainer(
            encoder=encoder,
            compressor=compressor,
            config=vb_config,
            is_directed=False,
        )

        # Train
        metrics = trainer.train(train_data, first_graph, anchor_indices)
        logger.info(
            "Contextual training complete: %d epochs, %.1fs, peak mem %d bytes",
            metrics.final_epoch + 1,
            metrics.total_time_sec,
            metrics.peak_memory_bytes,
        )

        # Log METR-08: training time and peak memory
        self.writer.add_scalar(
            "warcraft/contextual/training_time_sec", metrics.total_time_sec, 0
        )
        self.writer.add_scalar(
            "warcraft/contextual/peak_memory_bytes", metrics.peak_memory_bytes, 0
        )

        # Evaluate on test maps
        # DataSP evaluation protocol: run Dijkstra on predicted-cost graph,
        # compare path with GT optimal path on GT graph.
        test_maps = dataset["test"]["maps"]
        test_weights = dataset["test"]["weights"]
        _test_paths = dataset["test"]["paths"]

        results = []
        encoder.eval()
        compressor.eval()

        from aac.contextual.encoders import build_grid_edge_index, cell_costs_to_edge_weights
        from aac.graphs.convert import edges_to_graph

        for i in range(len(test_weights)):
            graph, _ = build_warcraft_graph(test_weights[i])

            # Ground-truth shortest path via Dijkstra (0,0) -> (H-1, W-1)
            src_node, tgt_node = 0, graph.num_nodes - 1
            gt_result = dijkstra(graph, src_node, tgt_node)

            # Predict edge costs from terrain image
            rgb = torch.from_numpy(test_maps[i]).permute(2, 0, 1).float() / 255.0
            context = rgb.unsqueeze(0)  # (1, 3, H_img, W_img)

            with torch.no_grad():
                # Get predicted cell costs from encoder
                cell_costs = encoder(context)  # (1, 144)
                cell_costs_2d = cell_costs.view(1, grid_size, grid_size)
                pred_edge_costs = cell_costs_to_edge_weights(
                    cell_costs_2d, grid_size
                )  # (1, E)

            # Build predicted-cost graph and run Dijkstra on it
            # build_grid_edge_index emits BOTH directions, so use is_directed=True
            # to avoid doubling edges in edges_to_graph
            src_idx, tgt_idx, _ = build_grid_edge_index(grid_size)
            pred_graph = edges_to_graph(
                src_idx, tgt_idx,
                pred_edge_costs.squeeze(0).to(torch.float64),
                num_nodes=graph.num_nodes,
                is_directed=True,
            )
            pred_result = dijkstra(pred_graph, src_node, tgt_node)

            # Evaluate predicted path on GT graph (true cost of predicted path)
            # Walk the predicted path edges on the GT graph to get true cost
            pred_path = pred_result.path
            true_cost_of_pred_path = _path_cost_on_graph(pred_path, graph)

            # Compute METR-07 metrics
            path_metrics = _compute_path_metrics(
                pred_path, gt_result.path,
                true_cost_of_pred_path, gt_result.cost,
            )

            results.append(
                {
                    "sample_idx": i,
                    "method": "contextual",
                    "path_match": float(path_metrics["match"]),
                    "jaccard": path_metrics["jaccard"],
                    "cost_regret": path_metrics["cost_regret"],
                    "expansions": pred_result.expansions,
                    "dijkstra_expansions": gt_result.expansions,
                }
            )

        return results

    def _run_datasp(
        self,
        cfg: DictConfig,
        data_dir: str,
        grid_size: int,
    ) -> list[dict]:
        """Run DataSP baseline via subprocess.

        Args:
            cfg: Full configuration.
            data_dir: Path to data directory.
            grid_size: Grid dimension.

        Returns:
            List of result dicts or empty if DataSP fails.
        """
        datasp_dir = Path("baselines/dataSP")
        if not datasp_dir.exists():
            raise RuntimeError(
                f"DataSP not found at {datasp_dir}. "
                "Run scripts/clone_datasp.sh first."
            )

        cmd = [
            "python",
            str(datasp_dir / "exp_warcraft_1to1.py"),
            "--data_dir",
            str(data_dir),
            "--grid_size",
            str(grid_size),
            "--seed",
            str(cfg.seed),
        ]

        logger.info("Running DataSP: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=str(datasp_dir),
            )
            if result.returncode != 0:
                logger.error("DataSP failed: %s", result.stderr)
                return [
                    {
                        "sample_idx": -1,
                        "method": "datasp",
                        "path_match": float("nan"),
                        "jaccard": float("nan"),
                        "cost_regret": float("nan"),
                    }
                ]

            return self._parse_datasp_output(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.error("DataSP execution error: %s", e)
            return [
                {
                    "sample_idx": -1,
                    "method": "datasp",
                    "path_match": float("nan"),
                    "jaccard": float("nan"),
                    "cost_regret": float("nan"),
                }
            ]

    def _parse_datasp_output(self, stdout: str) -> list[dict]:
        """Parse DataSP stdout for metrics.

        Args:
            stdout: Standard output from DataSP subprocess.

        Returns:
            List of result dicts parsed from output.
        """
        results = []
        for line in stdout.strip().split("\n"):
            line_lower = line.lower()
            if "match" in line_lower or "jaccard" in line_lower or "regret" in line_lower:
                logger.info("DataSP output: %s", line)
                try:
                    parts = line.split(",")
                    metrics: dict = {"sample_idx": -1, "method": "datasp"}
                    for part in parts:
                        key_val = part.strip().split(":")
                        if len(key_val) == 2:
                            key = key_val[0].strip().lower().replace(" ", "_")
                            val = float(key_val[1].strip())
                            metrics[key] = val
                    if any(isinstance(v, float) for k, v in metrics.items()
                           if k not in ("sample_idx", "method")):
                        results.append(metrics)
                except (ValueError, IndexError):
                    continue

        if not results:
            results.append(
                {
                    "sample_idx": -1,
                    "method": "datasp",
                    "path_match": float("nan"),
                    "jaccard": float("nan"),
                    "cost_regret": float("nan"),
                }
            )
        return results

    def _run_dijkstra(
        self,
        dataset: dict,
        grid_size: int,
    ) -> list[dict]:
        """Run Dijkstra on ground-truth edge costs as sanity baseline.

        All metrics should be perfect (match=1, jaccard=1, regret=0).

        Args:
            dataset: Loaded dataset dict with 'test' split.
            grid_size: Grid dimension.

        Returns:
            List of per-map result dicts.
        """
        from aac.graphs.loaders.warcraft import build_warcraft_graph
        from aac.search.dijkstra import dijkstra

        test_weights = dataset["test"]["weights"]
        results = []

        for i in range(len(test_weights)):
            graph, _ = build_warcraft_graph(test_weights[i])
            src_node, tgt_node = 0, graph.num_nodes - 1
            gt_result = dijkstra(graph, src_node, tgt_node)

            results.append(
                {
                    "sample_idx": i,
                    "method": "dijkstra",
                    "path_match": 1.0,
                    "jaccard": 1.0,
                    "cost_regret": 0.0,
                    "expansions": gt_result.expansions,
                    "dijkstra_expansions": gt_result.expansions,
                }
            )
        return results


def _path_cost_on_graph(path: list[int], graph) -> float:
    """Compute the true cost of walking a path on a given graph.

    Looks up each edge (u, v) in the graph's CSR structure and sums weights.

    Args:
        path: Vertex sequence.
        graph: Graph with CSR storage.

    Returns:
        Total edge cost along the path.
    """
    if len(path) < 2:
        return 0.0

    total = 0.0
    crow = graph.crow_indices
    col = graph.col_indices
    vals = graph.values

    for u, v in zip(path[:-1], path[1:]):
        # Find edge (u, v) in CSR row u
        row_start = int(crow[u].item())
        row_end = int(crow[u + 1].item())
        found = False
        for idx in range(row_start, row_end):
            if int(col[idx].item()) == v:
                total += float(vals[idx].item())
                found = True
                break
        if not found:
            # Edge not found in this direction, try reverse (undirected)
            row_start = int(crow[v].item())
            row_end = int(crow[v + 1].item())
            for idx in range(row_start, row_end):
                if int(col[idx].item()) == u:
                    total += float(vals[idx].item())
                    found = True
                    break
        if not found:
            raise ValueError(f"Edge ({u}, {v}) not found in graph")

    return total


def _compute_path_metrics(
    predicted_path: list[int],
    gt_path: list[int],
    predicted_cost: float,
    optimal_cost: float,
) -> dict:
    """Compute METR-07 path accuracy metrics.

    Args:
        predicted_path: Predicted shortest path as vertex sequence.
        gt_path: Ground-truth shortest path as vertex sequence.
        predicted_cost: Cost of predicted path.
        optimal_cost: Cost of optimal path.

    Returns:
        Dict with keys: match (bool), jaccard (float), cost_regret (float).
    """
    pred_edges = (
        set(zip(predicted_path[:-1], predicted_path[1:]))
        if len(predicted_path) > 1
        else set()
    )
    gt_edges = (
        set(zip(gt_path[:-1], gt_path[1:]))
        if len(gt_path) > 1
        else set()
    )

    match = pred_edges == gt_edges

    if not pred_edges and not gt_edges:
        jaccard = 1.0
    elif not pred_edges or not gt_edges:
        jaccard = 0.0
    else:
        intersection = pred_edges & gt_edges
        union = pred_edges | gt_edges
        jaccard = len(intersection) / len(union)

    if optimal_cost > 0 and not math.isinf(optimal_cost):
        cost_regret = (predicted_cost - optimal_cost) / optimal_cost
    else:
        cost_regret = 0.0

    return {"match": match, "jaccard": jaccard, "cost_regret": cost_regret}
