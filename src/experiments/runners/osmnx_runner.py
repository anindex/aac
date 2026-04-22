"""Track 2: OSMnx road graphs for PHIL comparison runner.

Loads cached city graphs, runs AAC or Dijkstra, measures metrics,
and includes PHIL reported numbers for comparison in outputs.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from experiments.runners.base import BaseRunner


class OSMnxRunner(BaseRunner):
    """Track 2: OSMnx road graphs for PHIL comparison.

    EXP-02: Greedy best-first with AAC vs PHIL reported numbers.
    Also runs exact A* with AAC to show admissibility advantage.

    Supported methods: aac, dijkstra only.
    ALT and FastMap are not run on OSMnx track (PHIL comparison
    uses different methodology).
    """

    SUPPORTED_METHODS = ["aac", "dijkstra"]

    def run(self, cfg: DictConfig) -> None:
        """Execute OSMnx track experiment for a single method across all cities.

        For each city in the track config:
        1. Load cached OSMnx graph from pickle
        2. Generate query pairs from largest connected component
        3. Run Dijkstra baseline for reference costs
        4. Preprocess the configured method (AAC/Dijkstra)
        5. Run queries with timing, collect per-query metrics
        6. Check admissibility against Dijkstra reference
        7. Measure batch throughput (METR-06)
        8. Include PHIL reported numbers for comparison
        9. Write CSV and log to TensorBoard

        Args:
            cfg: Hydra-composed configuration with track.cities, method, timing.
        """
        import pickle

        from aac.baselines import PHIL_REPORTED
        from aac.graphs.loaders.osmnx import _networkx_digraph_to_graph
        from aac.search.astar import astar
        from aac.search.dijkstra import dijkstra
        from experiments.metrics.admissibility import check_admissibility
        from experiments.metrics.collector import MetricsCollector, PreprocessingMetrics
        from experiments.metrics.timing import time_query
        from experiments.utils import generate_queries

        output_dir = Path(cfg.output_dir) / "osmnx"
        output_dir.mkdir(parents=True, exist_ok=True)

        for city_cfg in cfg.track.cities:
            city_name = city_cfg.name
            cache_path = f"{cfg.track.data_dir}/{city_name}.pkl"

            # Load cached OSMnx graph
            with open(cache_path, "rb") as f:
                nx_graph = pickle.load(f)
            graph = _networkx_digraph_to_graph(nx_graph)

            queries = generate_queries(graph, cfg.num_queries, seed=cfg.seed)

            # Dijkstra reference
            dijkstra_results = [dijkstra(graph, s, t) for s, t in queries]
            dijkstra_costs = [r.cost for r in dijkstra_results]

            # Method dispatch (already validated by BaseRunner._validate_method)
            method_name = cfg.method.name
            if method_name == "aac":
                heuristic, preprocess_metrics, _ = self.preprocess_aac(graph, cfg)
            elif method_name == "dijkstra":
                def heuristic(node: int, target: int) -> float:
                    return 0.0
                _preprocess_metrics = PreprocessingMetrics(0, 0, 0, 0)
            # No else needed: _validate_method already caught unsupported methods

            # Run exact A* with heuristic
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
                    f"{city_name}/{method_name}", i, result, timing.median_ms
                )

            # Admissibility check
            admissibility = check_admissibility(search_results, dijkstra_costs)

            # Batch throughput (METR-06)
            throughput = self.measure_batch_throughput(graph, heuristic, queries, cfg)

            summary = collector.summary()
            summary["graph"] = city_name
            summary["method"] = method_name
            summary["admissibility_violations"] = admissibility.num_violations

            # Add batch throughput
            for bs, qps in throughput.items():
                summary[f"throughput_batch_{bs}"] = qps

            # Add PHIL reported numbers for comparison
            if city_name in PHIL_REPORTED:
                summary["phil_reported"] = PHIL_REPORTED[city_name]

            collector.to_csv(str(output_dir / f"{city_name}_{method_name}.csv"))
            self.log_summary(f"{city_name}/{method_name}", summary)
