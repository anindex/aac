"""Track 1: DIMACS exact weighted point-to-point search runner.

Orchestrates the full pipeline: load graph, preprocess method, run queries,
collect metrics, measure batch throughput, run ablation, log to TensorBoard,
and write CSV results.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from omegaconf import DictConfig

from experiments.runners.base import BaseRunner


class DIMACSRunner(BaseRunner):
    """Track 1: DIMACS exact weighted point-to-point search.

    Runs AAC vs ALT vs FastMap vs Dijkstra at equal bytes/vertex.
    EXP-01: Main comparison tables
    EXP-05: Equal-budget fairness (Pareto frontier)
    EXP-07: Ablation (compressed vs uncompressed, A* vs greedy best-first)
    """

    def run(self, cfg: DictConfig) -> None:
        """Execute DIMACS track experiment for a single method across all graphs.

        For each graph in the track config:
        1. Load DIMACS graph from .gr/.co files
        2. Generate query pairs from largest connected component
        3. Run Dijkstra baseline for reference costs
        4. Preprocess the configured method (AAC/ALT/FastMap/Dijkstra)
        5. Run queries with timing, collect per-query metrics
        6. Check admissibility against Dijkstra reference
        7. Measure batch throughput (METR-06)
        8. Write CSV and log to TensorBoard

        Args:
            cfg: Hydra-composed configuration with track.graphs, method, timing.
        """
        from aac.graphs.loaders.dimacs import load_dimacs
        from aac.search.astar import astar
        from aac.search.dijkstra import dijkstra
        from experiments.metrics.admissibility import check_admissibility
        from experiments.metrics.collector import MetricsCollector, PreprocessingMetrics
        from experiments.metrics.timing import time_query
        from experiments.utils import generate_queries, memory_bytes_per_vertex

        output_dir = Path(cfg.output_dir) / "dimacs"
        output_dir.mkdir(parents=True, exist_ok=True)

        for graph_cfg in cfg.track.graphs:
            graph_name = graph_cfg.name
            gr_path = f"{cfg.track.data_dir}/{graph_cfg.gr_file}"
            co_path = f"{cfg.track.data_dir}/{graph_cfg.co_file}"

            # Load graph
            graph = load_dimacs(gr_path, co_path)

            # Generate queries from largest connected component
            queries = generate_queries(graph, cfg.num_queries, seed=cfg.seed)

            # Run Dijkstra baseline for reference costs
            dijkstra_results = [dijkstra(graph, s, t) for s, t in queries]
            dijkstra_costs = [r.cost for r in dijkstra_results]

            # Preprocess method
            method_name = cfg.method.name
            if method_name == "aac":
                heuristic, preprocess_metrics, _ = self.preprocess_aac(graph, cfg)
            elif method_name == "alt":
                heuristic, preprocess_metrics = self.preprocess_alt(graph, cfg)
            elif method_name == "fastmap":
                heuristic, preprocess_metrics = self.preprocess_fastmap(graph, cfg)
            elif method_name == "dijkstra":
                def heuristic(node: int, target: int) -> float:
                    return 0.0
                preprocess_metrics = PreprocessingMetrics(0, 0, 0, 0)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            # Run queries with timing
            collector = MetricsCollector()
            search_results = []
            for i, (s, t) in enumerate(queries):
                result = astar(graph, s, t, heuristic=heuristic)
                timing = time_query(
                    lambda s=s, t=t: astar(graph, s, t, heuristic=heuristic),
                    warmup_runs=cfg.timing.warmup_runs,
                    num_runs=cfg.timing.num_runs,
                )
                collector.add_query(i, s, t, result, timing.median_ms)
                search_results.append(result)
                self.log_query_metric(
                    f"{graph_name}/{method_name}", i, result, timing.median_ms
                )

            # Check admissibility
            admissibility = check_admissibility(search_results, dijkstra_costs)

            # Measure batch throughput (METR-06)
            throughput = self.measure_batch_throughput(graph, heuristic, queries, cfg)

            # Write results
            summary = collector.summary()
            summary["graph"] = graph_name
            summary["method"] = method_name
            summary["preprocess_total_sec"] = preprocess_metrics.total_sec
            summary["preprocess_anchor_sec"] = preprocess_metrics.anchor_selection_sec
            summary["preprocess_sssp_sec"] = preprocess_metrics.sssp_sec
            summary["preprocess_training_sec"] = preprocess_metrics.training_sec
            summary["admissibility_violations"] = admissibility.num_violations

            # Add batch throughput to summary
            for bs, qps in throughput.items():
                summary[f"throughput_batch_{bs}"] = qps

            # Compute memory (dtype_size=4 for float32 per METR-05)
            if method_name == "aac":
                m = cfg.method.m
                summary["memory_bytes_per_vertex"] = memory_bytes_per_vertex(m, 4)
            elif method_name == "alt":
                m = cfg.method.get("num_landmarks", cfg.method.get("m", 16))
                summary["memory_bytes_per_vertex"] = memory_bytes_per_vertex(2 * m, 4)
            elif method_name == "fastmap":
                m = cfg.method.get("num_dims", 16)
                summary["memory_bytes_per_vertex"] = memory_bytes_per_vertex(m, 4)
            else:
                summary["memory_bytes_per_vertex"] = 0

            collector.to_csv(str(output_dir / f"{method_name}_{graph_name}.csv"))
            self.log_summary(f"{graph_name}/{method_name}", summary)

        self.log_hparams(
            {
                "method": cfg.method.name,
                "seed": cfg.seed,
                "num_queries": cfg.num_queries,
            },
            {"total_graphs": len(cfg.track.graphs)},
        )

    def run_ablation(self, cfg: DictConfig) -> None:
        """EXP-07: Ablation study comparing compressed vs uncompressed, A* vs greedy.

        Ablation compares:
        1. AAC (compressed, K0->m) vs ALT (uncompressed, same K0 landmarks) at same K0.
           This isolates the effect of compression: same teacher labels, with/without compression.
        2. A* (exact, optimal) vs greedy best-first (expand node with smallest h, no g-cost).
           This isolates the effect of exact search vs heuristic-only search.

        The greedy best-first approximation (scaling h by 1e12) uses the existing A*
        infrastructure. When h dominates g, A* degenerates to greedy best-first.

        Results are written to results/dimacs/ablation/ directory.

        Args:
            cfg: Hydra-composed configuration with track.graphs, method.K0, method.m.
        """
        from aac.baselines.alt import alt_preprocess, make_alt_heuristic
        from aac.graphs.loaders.dimacs import load_dimacs
        from aac.search.astar import astar
        from aac.search.dijkstra import dijkstra
        from experiments.metrics.admissibility import check_admissibility
        from experiments.metrics.collector import MetricsCollector
        from experiments.utils import generate_queries

        output_dir = Path(cfg.output_dir) / "dimacs" / "ablation"
        output_dir.mkdir(parents=True, exist_ok=True)

        K0 = cfg.method.get("K0", 64)
        m = cfg.method.get("m", 16)

        for graph_cfg in cfg.track.graphs:
            graph_name = graph_cfg.name
            gr_path = f"{cfg.track.data_dir}/{graph_cfg.gr_file}"
            co_path = f"{cfg.track.data_dir}/{graph_cfg.co_file}"

            graph = load_dimacs(gr_path, co_path)
            queries = generate_queries(graph, cfg.num_queries, seed=cfg.seed)

            # Dijkstra reference
            dijkstra_costs = [dijkstra(graph, s, t).cost for s, t in queries]

            # --- Ablation 1: Compressed (AAC) vs Uncompressed (ALT at same K0) ---

            # AAC compressed heuristic
            aac_heuristic, aac_metrics, _ = self.preprocess_aac(graph, cfg)

            # ALT uncompressed heuristic at same K0 (identity compression)
            alt_teacher = alt_preprocess(graph, K0)
            alt_heuristic = make_alt_heuristic(alt_teacher)

            for label, heuristic in [
                ("aac_compressed", aac_heuristic),
                ("alt_uncompressed_K0", alt_heuristic),
            ]:
                collector = MetricsCollector()
                search_results = []
                for i, (s, t) in enumerate(queries):
                    r = astar(graph, s, t, heuristic=heuristic)
                    collector.add_query(i, s, t, r, 0.0)
                    search_results.append(r)
                summary = collector.summary()
                summary["graph"] = graph_name
                summary["method"] = label
                summary["K0"] = K0
                summary["m"] = m if label == "aac_compressed" else K0
                admissibility = check_admissibility(search_results, dijkstra_costs)
                summary["admissibility_violations"] = admissibility.num_violations
                collector.to_csv(str(output_dir / f"{label}_{graph_name}.csv"))

            # --- Ablation 2: A* (exact) vs Greedy best-first (h-only) ---
            greedy_h = _make_greedy_heuristic(aac_heuristic)

            for label, h_fn in [
                ("aac_astar", aac_heuristic),
                ("aac_greedy", greedy_h),
            ]:
                collector = MetricsCollector()
                search_results = []
                for i, (s, t) in enumerate(queries):
                    r = astar(graph, s, t, heuristic=h_fn)
                    collector.add_query(i, s, t, r, 0.0)
                    search_results.append(r)
                summary = collector.summary()
                summary["graph"] = graph_name
                summary["method"] = label
                admissibility = check_admissibility(search_results, dijkstra_costs)
                summary["admissibility_violations"] = admissibility.num_violations
                collector.to_csv(str(output_dir / f"{label}_{graph_name}.csv"))


def _make_greedy_heuristic(
    h: Callable[[int, int], float],
) -> Callable[[int, int], float]:
    """Wrap heuristic to make greedy best-first: f = h only (no g-cost).

    Approximates greedy by making h very large relative to g:
    A* with h'(n,t) = 1e12 * h(n,t) effectively ignores g-cost.

    Args:
        h: Original heuristic function h(node, target) -> float.

    Returns:
        Greedy heuristic that scales h by 1e12.
    """

    def greedy_h(node: int, target: int) -> float:
        return 1e12 * h(node, target)

    return greedy_h
