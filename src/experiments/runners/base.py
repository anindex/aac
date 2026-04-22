"""Base experiment runner with shared metrics, timing, TensorBoard logging, and preprocessing."""

from __future__ import annotations

from collections.abc import Callable

import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from experiments.metrics.collector import PreprocessingMetrics
from aac.search.types import SearchResult


class BaseRunner:
    """Base class for experiment runners with shared metrics, timing, and TensorBoard logging.

    Provides:
    - TensorBoard SummaryWriter for per-query and summary metric logging (REPRO-05)
    - Method validation against SUPPORTED_METHODS whitelist
    - Preprocessing dispatch for AAC, ALT, and FastMap methods
    - Batch throughput measurement (METR-06)

    Subclasses should override ``SUPPORTED_METHODS`` to restrict which methods
    are valid for that track, and implement ``run(cfg)`` for the experiment logic.
    """

    # Subclasses should override to restrict supported methods.
    # None means all methods are supported.
    SUPPORTED_METHODS: list[str] | None = None

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.writer = SummaryWriter(
            log_dir=f"{cfg.log_dir}/{cfg.track.name}/{cfg.method.name}"
        )
        self._validate_method(cfg.method.name)

    def _validate_method(self, method_name: str) -> None:
        """Validate that the method is supported by this runner.

        Args:
            method_name: Name of the method to validate.

        Raises:
            ValueError: If method is not in SUPPORTED_METHODS.
        """
        if self.SUPPORTED_METHODS is not None and method_name not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Method '{method_name}' is not supported by {self.__class__.__name__}. "
                f"Supported methods: {self.SUPPORTED_METHODS}"
            )

    def run(self, cfg: DictConfig) -> None:
        """Execute the experiment. Must be overridden by subclasses.

        Args:
            cfg: Hydra-composed configuration.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError

    def log_hparams(self, hparams: dict, metrics: dict) -> None:
        """Log hyperparameters and associated metrics to TensorBoard.

        Args:
            hparams: Dictionary of hyperparameter names to values.
            metrics: Dictionary of metric names to values.
        """
        self.writer.add_hparams(hparams, metrics)

    def log_query_metric(
        self,
        tag_prefix: str,
        query_idx: int,
        result: SearchResult,
        latency_ms: float,
    ) -> None:
        """Log per-query metrics to TensorBoard.

        Args:
            tag_prefix: Tag prefix (e.g., "NY/aac").
            query_idx: Query index used as the global step.
            result: SearchResult from the search function.
            latency_ms: Query latency in milliseconds.
        """
        self.writer.add_scalar(f"{tag_prefix}/expansions", result.expansions, query_idx)
        self.writer.add_scalar(f"{tag_prefix}/cost", result.cost, query_idx)
        self.writer.add_scalar(f"{tag_prefix}/latency_ms", latency_ms, query_idx)

    def log_summary(self, tag_prefix: str, summary: dict) -> None:
        """Log aggregate summary metrics to TensorBoard.

        Args:
            tag_prefix: Tag prefix for the summary scalars.
            summary: Dictionary of summary metric names to values.
        """
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{tag_prefix}/summary/{key}", value, 0)

    def preprocess_aac(
        self, graph, cfg: DictConfig
    ) -> tuple[Callable[[int, int], float], PreprocessingMetrics, torch.Tensor]:
        """AAC preprocessing: anchors -> SSSP -> embed -> compress -> train.

        Uses teacher heuristic (ALT lower bound) as training target instead of
        O(V^2) pairwise distances. This is gradient-equivalent to using true
        distances because h_teacher is constant w.r.t. compressor parameters.

        Selects embedding type based on graph directedness:
        - Undirected: Hilbert simplex embedding
        - Directed: Tropical/Funk embedding

        Args:
            graph: Graph in CSR format.
            cfg: Configuration with method.K0, method.m, method.training.

        Returns:
            Tuple of (heuristic_fn, preprocess_metrics, compressed_labels).
        """
        import time

        from aac.compression.compressor import LinearCompressor, make_linear_heuristic
        from aac.embeddings.anchors import farthest_point_sampling
        from aac.embeddings.sssp import compute_teacher_labels
        from aac.train.trainer import TrainConfig, train_linear_compressor

        K0 = cfg.method.K0
        m = cfg.method.m

        # Compute LCC for anchor selection and training
        from experiments.utils import compute_strong_lcc
        lcc_nodes, lcc_seed = compute_strong_lcc(graph)

        t0 = time.perf_counter()
        anchors = farthest_point_sampling(
            graph, K0, seed_vertex=lcc_seed,
            valid_vertices=torch.tensor(lcc_nodes, dtype=torch.int64),
        )
        t1 = time.perf_counter()
        teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)
        t2 = time.perf_counter()

        # LinearCompressor: row-stochastic compression preserving per-landmark diffs
        compressor = LinearCompressor(K=K0, m=m, is_directed=graph.is_directed)
        train_cfg = TrainConfig(
            num_epochs=cfg.method.training.num_epochs,
            batch_size=cfg.method.training.batch_size,
            lr=cfg.method.training.lr,
            cond_lambda=cfg.method.training.cond_lambda,
            T_init=cfg.method.training.T_init,
            gamma=cfg.method.training.gamma,
            seed=cfg.seed,
        )

        lcc_vertices = torch.tensor(lcc_nodes, dtype=torch.int64)
        train_linear_compressor(
            compressor, teacher_labels, train_cfg,
            valid_vertices=lcc_vertices,
        )
        t3 = time.perf_counter()

        d_out_t = teacher_labels.d_out.t()
        d_in_t = teacher_labels.d_in.t()
        with torch.no_grad():
            if graph.is_directed:
                y_fwd, y_bwd = compressor(d_out_t, d_in_t)
                y_fwd, y_bwd = y_fwd.detach(), y_bwd.detach()
            else:
                y = compressor(d_out_t)
                y_fwd = y_bwd = y.detach()
        heuristic = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)
        compressed_labels = torch.cat([y_fwd, y_bwd], dim=1)

        preprocess_metrics = PreprocessingMetrics(
            anchor_selection_sec=t1 - t0,
            sssp_sec=t2 - t1,
            training_sec=t3 - t2,
            total_sec=t3 - t0,
        )
        return heuristic, preprocess_metrics, compressed_labels

    def preprocess_alt(
        self, graph, cfg: DictConfig
    ) -> tuple[Callable[[int, int], float], PreprocessingMetrics]:
        """ALT preprocessing. Returns (heuristic_fn, preprocess_metrics).

        Args:
            graph: Graph in CSR format.
            cfg: Configuration with method.num_landmarks or method.m.

        Returns:
            Tuple of (heuristic_fn, preprocess_metrics).
        """
        import time

        from aac.baselines.alt import alt_preprocess, make_alt_heuristic

        # Restrict landmarks to LCC for fairness (same as AAC preprocessing)
        from experiments.utils import compute_strong_lcc
        lcc_nodes, lcc_seed = compute_strong_lcc(graph)
        lcc_vertices = torch.tensor(lcc_nodes, dtype=torch.int64)

        t0 = time.perf_counter()
        num_landmarks = cfg.method.get("num_landmarks", cfg.method.get("m", 16))
        teacher_labels = alt_preprocess(
            graph, num_landmarks, seed_vertex=lcc_seed,
            valid_vertices=lcc_vertices,
        )
        t1 = time.perf_counter()
        heuristic = make_alt_heuristic(teacher_labels)
        preprocess_metrics = PreprocessingMetrics(
            anchor_selection_sec=0.0,  # included in total
            sssp_sec=t1 - t0,
            training_sec=0.0,
            total_sec=t1 - t0,
        )
        return heuristic, preprocess_metrics

    def preprocess_fastmap(
        self, graph, cfg: DictConfig
    ) -> tuple[Callable[[int, int], float], PreprocessingMetrics]:
        """FastMap preprocessing. Returns (heuristic_fn, preprocess_metrics).

        Args:
            graph: Graph in CSR format.
            cfg: Configuration with method.num_dims or method.m.

        Returns:
            Tuple of (heuristic_fn, preprocess_metrics).
        """
        import time

        from aac.baselines.fastmap import fastmap_preprocess, make_fastmap_heuristic

        t0 = time.perf_counter()
        num_dims = cfg.method.get("num_dims", cfg.method.get("m", 16))
        coords = fastmap_preprocess(graph, num_dims)
        t1 = time.perf_counter()
        heuristic = make_fastmap_heuristic(coords)
        preprocess_metrics = PreprocessingMetrics(
            anchor_selection_sec=0.0,
            sssp_sec=0.0,
            training_sec=0.0,
            total_sec=t1 - t0,
        )
        return heuristic, preprocess_metrics

    def measure_batch_throughput(
        self,
        graph,
        heuristic: Callable[[int, int], float],
        queries: list[tuple[int, int]],
        cfg: DictConfig,
    ) -> dict[int, float]:
        """Measure batch throughput at configured batch sizes (METR-06).

        Args:
            graph: Graph in CSR format.
            heuristic: Admissible heuristic function.
            queries: List of (source, target) query pairs.
            cfg: Configuration with batch_throughput_sizes and seed.

        Returns:
            Dict mapping batch_size to queries per second.
        """
        from experiments.metrics.collector import batch_throughput

        batch_sizes = list(cfg.get("batch_throughput_sizes", [1, 8, 32, 128, 1024]))
        return batch_throughput(
            graph, heuristic, queries, batch_sizes=batch_sizes, seed=cfg.seed
        )

    def close(self) -> None:
        """Flush and close the TensorBoard SummaryWriter."""
        self.writer.close()
