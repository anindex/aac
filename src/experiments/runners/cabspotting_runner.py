"""Track 3: Cabspotting taxi trajectory runner for Contextual evaluation.

Trains CabspottingMLP to predict edge travel times from temporal features,
evaluates with METR-07 metrics (path match, Jaccard, cost regret),
and runs DataSP baseline on the same data splits for comparison.

DataSP Cabspotting hyperparameters (matched for fair comparison):
    - Input features: [day_of_week (4 one-hot), is_holiday, time_start] = 6 dims
    - Graph: ~355 nodes, ~2178 edges
    - Beta (smoothing): 30.0
    - Train/test: 70/30 split by driver
    - Epochs: 100, early stopping patience=4
"""

from __future__ import annotations

import csv
import logging
import subprocess
import time
from pathlib import Path

import torch
from omegaconf import DictConfig

from experiments.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class CabspottingRunner(BaseRunner):
    """Track 3: Cabspotting taxi trajectory experiments (AAC Contextual vs DataSP).

    Trains a CabspottingMLP to predict per-edge travel times from edge features
    (day_of_week, is_holiday, time_start), then evaluates path quality using
    METR-07 metrics against ground-truth shortest paths.

    Supports DataSP comparison via subprocess execution of their Cabspotting
    experiment code, and Dijkstra as a sanity baseline.

    IMPORTANT: Uses contextual_forward_mlp (NOT contextual_forward) because
    CabspottingMLP produces edge costs directly from features, with no
    cell-to-edge conversion needed. This is the key difference from WarcraftRunner.
    """

    SUPPORTED_METHODS = ["contextual", "datasp", "dijkstra"]

    def run(self, cfg: DictConfig) -> None:
        """Execute Cabspotting experiment for the configured method.

        Args:
            cfg: Hydra-composed configuration with track and method sections.
        """
        data_dir = Path(cfg.track.data_dir)
        data = self._load_cabspotting_data(
            str(data_dir), cfg.track.get("train_split", 0.7)
        )

        if data is None:
            logger.warning(
                "Cabspotting data not found at %s. Download from "
                "crawdad.org/epfl/mobility and preprocess using "
                "baselines/dataSP/cabspotting_preprocessing/ notebooks.",
                data_dir,
            )
            return

        method = cfg.method.name
        if method == "contextual":
            results = self._run_contextual(cfg, data)
        elif method == "datasp":
            results = self._run_datasp(cfg, str(data_dir), data)
        elif method == "dijkstra":
            results = self._run_dijkstra(cfg, data)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Write results CSV
        self._write_results_csv(cfg, results)

        # Log summary to TensorBoard
        self._log_results(cfg, results)

    def _load_cabspotting_data(
        self, data_dir: str, train_split: float = 0.7
    ) -> dict | None:
        """Load preprocessed Cabspotting data from disk.

        Expected files in data_dir:
            - graph_edges.pt: (2, E) int64 edge index tensor [sources, targets]
            - edge_features.pt: (N_samples, E, F) float32 edge feature tensor
            - ground_truth_times.pt: (N_samples, E) float32 travel times per sample
            - Optional: driver_ids.pt: (N_samples,) int64 driver IDs for split

        If data files not found, returns None.

        Args:
            data_dir: Path to preprocessed Cabspotting data directory.
            train_split: Fraction of samples for training (default 0.7).

        Returns:
            Dict with keys: graph_edges, edge_features, ground_truth_times,
            train_mask, test_mask, num_nodes, num_edges. Or None if data missing.
        """
        data_path = Path(data_dir)

        required_files = ["graph_edges.pt", "edge_features.pt", "ground_truth_times.pt"]
        for fname in required_files:
            if not (data_path / fname).exists():
                return None

        graph_edges = torch.load(data_path / "graph_edges.pt", weights_only=True)
        edge_features = torch.load(data_path / "edge_features.pt", weights_only=True)
        ground_truth_times = torch.load(
            data_path / "ground_truth_times.pt", weights_only=True
        )

        num_nodes = int(graph_edges.max().item()) + 1
        num_edges = graph_edges.shape[1]
        num_samples = edge_features.shape[0]

        # Split by sample index (proxy for driver split)
        # DataSP splits 70/30 by driver; we split by sample index for reproducibility
        if (data_path / "driver_ids.pt").exists():
            driver_ids = torch.load(data_path / "driver_ids.pt", weights_only=True)
            unique_drivers = driver_ids.unique()
            n_train_drivers = int(len(unique_drivers) * train_split)
            train_drivers = set(unique_drivers[:n_train_drivers].tolist())
            train_mask = torch.tensor(
                [d.item() in train_drivers for d in driver_ids], dtype=torch.bool
            )
        else:
            n_train = int(num_samples * train_split)
            train_mask = torch.zeros(num_samples, dtype=torch.bool)
            train_mask[:n_train] = True

        test_mask = ~train_mask

        return {
            "graph_edges": graph_edges,
            "edge_features": edge_features,
            "ground_truth_times": ground_truth_times,
            "train_mask": train_mask,
            "test_mask": test_mask,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
        }

    def _run_contextual(self, cfg: DictConfig, data: dict) -> list[dict]:
        """Run Contextual training and evaluation on Cabspotting data.

        Uses CabspottingMLP encoder and contextual_forward_mlp pipeline
        for end-to-end differentiable label construction.

        Args:
            cfg: Experiment configuration.
            data: Loaded Cabspotting data dict.

        Returns:
            List of per-sample result dicts with path_match, jaccard, cost_regret.
        """
        import scipy.sparse.csgraph

        from aac.compression.compressor import LinearCompressor
        from aac.contextual.encoders import CabspottingMLP
        from aac.contextual.trainer import ContextualConfig, ContextualTrainer
        from aac.embeddings.anchors import farthest_point_sampling
        from aac.graphs.convert import edges_to_graph
        from aac.search.dijkstra import dijkstra

        graph_edges = data["graph_edges"]
        edge_features = data["edge_features"]
        gt_times = data["ground_truth_times"]
        train_mask = data["train_mask"]
        test_mask = data["test_mask"]
        num_nodes = data["num_nodes"]

        # Build graph template from edge topology with unit weights
        sources = graph_edges[0]
        targets = graph_edges[1]
        unit_weights = torch.ones(sources.shape[0], dtype=torch.float64)
        graph_template = edges_to_graph(
            sources, targets, unit_weights, num_nodes=num_nodes, is_directed=True
        )

        # Select anchors
        K = cfg.method.get("K", 5)
        m = cfg.method.get("m", 8)
        anchors = farthest_point_sampling(graph_template, K)

        # Create encoder and compressor
        input_dim = cfg.track.get("input_dim", 6)
        encoder = CabspottingMLP(input_dim=input_dim, hidden_dim=64)
        compressor = LinearCompressor(K=K, m=m, is_directed=True)

        # Use beta_override from track config if present
        beta = cfg.track.get("beta_override", 30.0)

        # Prepare training data: list of (features, gt_distances) tuples
        # For each train sample, compute pairwise distances from gt edge costs
        train_data = []
        train_indices = torch.where(train_mask)[0]
        for idx in train_indices:
            feat = edge_features[idx]  # (E, F)
            gt_edge_costs = gt_times[idx]  # (E,)

            # Build graph with GT costs for distance computation
            gt_graph = edges_to_graph(
                sources, targets, gt_edge_costs.to(torch.float64),
                num_nodes=num_nodes, is_directed=True,
            )

            # Compute pairwise distances via scipy
            from aac.graphs.convert import graph_to_scipy

            scipy_csr = graph_to_scipy(gt_graph)
            dist_matrix = torch.tensor(
                scipy.sparse.csgraph.shortest_path(scipy_csr, directed=True),
                dtype=torch.float64,
            )

            # Add batch dimension: (E, F) -> (1, E, F) so pipeline
            # treats dim 0 as batch (not edge dimension)
            train_data.append((feat.unsqueeze(0), dist_matrix))

        # Configure trainer
        trainer_config = ContextualConfig(
            num_epochs=cfg.method.get("num_epochs", 100),
            batch_size=cfg.method.get("batch_size", 12),
            lr=cfg.method.get("lr", 1e-4),
            beta_init=1.0,
            beta_max=beta,
            beta_gamma=1.05,
            patience=cfg.method.get("patience", 4),
            K=K,
            m=m,
        )

        trainer = ContextualTrainer(
            encoder=encoder,
            compressor=compressor,
            config=trainer_config,
            is_directed=True,
        )

        # Reset GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_start = time.perf_counter()
        train_metrics = trainer.train(train_data, graph_template, anchors)
        train_time = time.perf_counter() - train_start

        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = int(torch.cuda.max_memory_allocated())

        # Log METR-08: training time and GPU memory
        logger.info(
            "Contextual training: %.1fs, %d epochs, peak GPU memory: %d bytes",
            train_time,
            train_metrics.final_epoch + 1,
            peak_memory,
        )
        self.writer.add_scalar("cabspotting/contextual/train_time_sec", train_time, 0)
        self.writer.add_scalar(
            "cabspotting/contextual/peak_memory_bytes", peak_memory, 0
        )

        # Evaluate on test samples
        results = []
        test_indices = torch.where(test_mask)[0]
        encoder.eval()
        compressor.eval()

        # Fixed-seed RNG for reproducible query pairs across methods
        query_rng = torch.Generator().manual_seed(cfg.get("seed", 42))

        with torch.no_grad():
            for sample_idx, idx in enumerate(test_indices):
                feat = edge_features[idx].unsqueeze(0)  # (1, E, F)
                gt_edge_costs = gt_times[idx]  # (E,)

                # Predict edge costs
                predicted_costs = encoder(feat).squeeze(0)  # (E,)

                # Build predicted and GT graphs
                pred_graph = edges_to_graph(
                    sources, targets,
                    predicted_costs.to(torch.float64),
                    num_nodes=num_nodes, is_directed=True,
                )
                gt_graph = edges_to_graph(
                    sources, targets,
                    gt_edge_costs.to(torch.float64),
                    num_nodes=num_nodes, is_directed=True,
                )

                # Deterministic query pair (same seed for all methods)
                src = int(torch.randint(
                    0, num_nodes, (1,), generator=query_rng
                ).item())
                tgt = int(torch.randint(
                    0, num_nodes, (1,), generator=query_rng
                ).item())
                if tgt == src:
                    tgt = (src + 1) % num_nodes

                # Run Dijkstra on both graphs
                pred_result = dijkstra(pred_graph, src, tgt)
                gt_result = dijkstra(gt_graph, src, tgt)

                # Compute METR-07 metrics
                metrics = self._compute_path_metrics(
                    pred_result.path, gt_result.path,
                    pred_result.cost, gt_result.cost,
                )
                metrics["sample_id"] = int(idx.item())
                metrics["method"] = "contextual"
                results.append(metrics)

        return results

    def _run_datasp(
        self, cfg: DictConfig, data_dir: str, split_info: dict
    ) -> list[dict]:
        """Run DataSP Cabspotting experiment via subprocess.

        Args:
            cfg: Experiment configuration.
            data_dir: Path to Cabspotting data directory.
            split_info: Data dict with split information.

        Returns:
            List of result dicts (parsed from DataSP output).
        """
        datasp_dir = Path("baselines/dataSP")
        if not datasp_dir.exists():
            raise RuntimeError(
                "DataSP baseline not found at baselines/dataSP/. "
                "Clone it: git clone https://github.com/AlanLahoud/dataSP.git "
                "baselines/dataSP"
            )

        script = datasp_dir / "exp_cabspotting.py"
        if not script.exists():
            raise RuntimeError(
                f"DataSP script not found: {script}. "
                "Check the DataSP repository structure."
            )

        cmd = [
            "python",
            str(script),
            "--data_dir", data_dir,
            "--seed", str(cfg.get("seed", 42)),
        ]

        logger.info("Running DataSP Cabspotting experiment: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=str(datasp_dir),
            )

            if result.returncode != 0:
                logger.error("DataSP failed: %s", result.stderr[:500])
                return [
                    {
                        "sample_id": -1,
                        "method": "datasp",
                        "path_match": float("nan"),
                        "jaccard": float("nan"),
                        "cost_regret": float("nan"),
                    }
                ]

            # Parse DataSP output -- expect summary line with metrics
            return self._parse_datasp_output(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("DataSP timed out after 3600 seconds")
            return [
                {
                    "sample_id": -1,
                    "method": "datasp",
                    "path_match": float("nan"),
                    "jaccard": float("nan"),
                    "cost_regret": float("nan"),
                }
            ]
        except Exception as e:
            logger.error("DataSP execution error: %s", e)
            return [
                {
                    "sample_id": -1,
                    "method": "datasp",
                    "path_match": float("nan"),
                    "jaccard": float("nan"),
                    "cost_regret": float("nan"),
                }
            ]

    def _run_dijkstra(self, cfg: DictConfig, data: dict) -> list[dict]:
        """Run Dijkstra on ground-truth travel times as sanity baseline.

        Should produce perfect metrics (path_match=1.0, jaccard=1.0, cost_regret=0.0).

        Args:
            cfg: Experiment configuration.
            data: Loaded Cabspotting data dict.

        Returns:
            List of per-sample result dicts.
        """
        from aac.graphs.convert import edges_to_graph
        from aac.search.dijkstra import dijkstra

        graph_edges = data["graph_edges"]
        gt_times = data["ground_truth_times"]
        test_mask = data["test_mask"]
        num_nodes = data["num_nodes"]

        sources = graph_edges[0]
        targets = graph_edges[1]

        results = []
        test_indices = torch.where(test_mask)[0]

        # Same fixed-seed RNG as _run_contextual for identical query pairs
        query_rng = torch.Generator().manual_seed(cfg.get("seed", 42))

        for sample_idx, idx in enumerate(test_indices):
            gt_edge_costs = gt_times[idx]

            gt_graph = edges_to_graph(
                sources, targets,
                gt_edge_costs.to(torch.float64),
                num_nodes=num_nodes, is_directed=True,
            )

            # Deterministic query pair (same seed as _run_contextual)
            src = int(torch.randint(
                0, num_nodes, (1,), generator=query_rng
            ).item())
            tgt = int(torch.randint(
                0, num_nodes, (1,), generator=query_rng
            ).item())
            if tgt == src:
                tgt = (src + 1) % num_nodes

            result = dijkstra(gt_graph, src, tgt)

            # Dijkstra on GT is the reference, so metrics should be perfect
            metrics = self._compute_path_metrics(
                result.path, result.path,
                result.cost, result.cost,
            )
            metrics["sample_id"] = int(idx.item())
            metrics["method"] = "dijkstra"
            results.append(metrics)

        return results

    @staticmethod
    def _compute_path_metrics(
        predicted_path: list[int],
        gt_path: list[int],
        predicted_cost: float,
        optimal_cost: float,
    ) -> dict:
        """Compute METR-07 path accuracy metrics.

        Args:
            predicted_path: List of vertex IDs in the predicted path.
            gt_path: List of vertex IDs in the ground-truth shortest path.
            predicted_cost: Cost of the predicted path.
            optimal_cost: Cost of the optimal path.

        Returns:
            Dict with path_match, jaccard, and cost_regret.
        """
        # Path match: exact edge set match
        pred_edges = set(
            zip(predicted_path[:-1], predicted_path[1:])
        ) if len(predicted_path) > 1 else set()
        gt_edges = set(
            zip(gt_path[:-1], gt_path[1:])
        ) if len(gt_path) > 1 else set()

        path_match = 1.0 if pred_edges == gt_edges else 0.0

        # Jaccard: edge set overlap
        if not pred_edges and not gt_edges:
            jaccard = 1.0
        elif not pred_edges or not gt_edges:
            jaccard = 0.0
        else:
            intersection = len(pred_edges & gt_edges)
            union = len(pred_edges | gt_edges)
            jaccard = intersection / union if union > 0 else 0.0

        # Cost regret: (predicted - optimal) / optimal
        if optimal_cost > 0:
            cost_regret = (predicted_cost - optimal_cost) / optimal_cost
        else:
            cost_regret = 0.0 if predicted_cost == 0 else float("inf")

        return {
            "path_match": path_match,
            "jaccard": jaccard,
            "cost_regret": cost_regret,
        }

    @staticmethod
    def _parse_datasp_output(stdout: str) -> list[dict]:
        """Parse DataSP stdout for experiment metrics.

        Looks for lines like:
            Match: 0.95, Jaccard: 0.92, Cost Regret: 0.001

        Args:
            stdout: DataSP subprocess stdout.

        Returns:
            List of result dicts.
        """
        results = []
        for line in stdout.strip().split("\n"):
            line_lower = line.lower()
            if "match" in line_lower and "jaccard" in line_lower:
                try:
                    parts = line.split(",")
                    metrics: dict = {"sample_id": -1, "method": "datasp"}
                    for part in parts:
                        key_val = part.strip().split(":")
                        if len(key_val) == 2:
                            key = key_val[0].strip().lower().replace(" ", "_")
                            val = float(key_val[1].strip())
                            metrics[key] = val
                    results.append(metrics)
                except (ValueError, IndexError):
                    continue

        if not results:
            # Return NaN metrics if parsing failed
            results.append(
                {
                    "sample_id": -1,
                    "method": "datasp",
                    "path_match": float("nan"),
                    "jaccard": float("nan"),
                    "cost_regret": float("nan"),
                }
            )

        return results

    def _write_results_csv(self, cfg: DictConfig, results: list[dict]) -> None:
        """Write per-sample results to CSV.

        Args:
            cfg: Experiment configuration (for output path).
            results: List of per-sample result dicts.
        """
        if not results:
            return

        output_dir = Path(cfg.log_dir) / cfg.track.name / cfg.method.name
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "results.csv"

        from experiments.reporting.csv_writer import write_csv_metadata

        fieldnames = ["sample_id", "method", "path_match", "jaccard", "cost_regret"]
        with open(csv_path, "w", newline="") as f:
            write_csv_metadata(f)
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        logger.info("Results written to %s", csv_path)

    def _log_results(self, cfg: DictConfig, results: list[dict]) -> None:
        """Log summary metrics to TensorBoard.

        Args:
            cfg: Experiment configuration.
            results: List of per-sample result dicts.
        """
        if not results:
            return

        tag_prefix = f"cabspotting/{cfg.method.name}"

        # Compute mean metrics (excluding NaN)
        valid_results = [
            r for r in results
            if not (
                r.get("path_match") != r.get("path_match")  # NaN check
                or r.get("jaccard") != r.get("jaccard")
                or r.get("cost_regret") != r.get("cost_regret")
            )
        ]

        if valid_results:
            mean_path_match = sum(r["path_match"] for r in valid_results) / len(
                valid_results
            )
            mean_jaccard = sum(r["jaccard"] for r in valid_results) / len(valid_results)
            mean_cost_regret = sum(r["cost_regret"] for r in valid_results) / len(
                valid_results
            )

            self.writer.add_scalar(
                f"{tag_prefix}/mean_path_match", mean_path_match, 0
            )
            self.writer.add_scalar(f"{tag_prefix}/mean_jaccard", mean_jaccard, 0)
            self.writer.add_scalar(
                f"{tag_prefix}/mean_cost_regret", mean_cost_regret, 0
            )

            summary = {
                "mean_path_match": mean_path_match,
                "mean_jaccard": mean_jaccard,
                "mean_cost_regret": mean_cost_regret,
                "num_samples": len(valid_results),
            }
            self.log_summary(tag_prefix, summary)

            logger.info(
                "Cabspotting %s summary: path_match=%.4f, jaccard=%.4f, "
                "cost_regret=%.4f (%d samples)",
                cfg.method.name,
                mean_path_match,
                mean_jaccard,
                mean_cost_regret,
                len(valid_results),
            )
