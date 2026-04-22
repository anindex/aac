# `src/experiments/` -- Hydra-configured experiment runners

This package is the **single configuration-driven entry point** for the per-track experiments cited in the paper. Each scientific track (DIMACS road networks, OSMnx city graphs, Warcraft contextual planning, Cabspotting) is implemented as one runner whose behaviour is fully specified by composable Hydra configuration files under [`configs/`](configs/).

For the per-experiment artifact index -- which CSV/log backs which paper Table or Figure -- see [`results/README.md`](../../results/README.md). For installation and the headline reproduction command, see the project [`README.md`](../../README.md).

## Quick start

```bash
# DIMACS NY with the default AAC method config
python -m experiments track=dimacs method=aac

# Override scalars from the CLI (Hydra dot-notation)
python -m experiments track=dimacs method=aac seed=123 num_queries=500

# Switch the method on the same track
python -m experiments track=dimacs method=alt
python -m experiments track=osmnx  method=aac
python -m experiments track=warcraft method=contextual
```

The `python -m experiments` invocation is dispatched by [`__main__.py`](__main__.py) -> [`run.py`](run.py), which is a thin Hydra wrapper:

1. Composes the configuration ([`configs/config.yaml`](configs/config.yaml) + selected `track/*.yaml` + selected `method/*.yaml`).
2. Calls `seed_everything(cfg.seed)` from [`utils.py`](utils.py).
3. Looks up the right runner via `runners.get_runner(cfg.track.name)`.
4. Runs the experiment and writes outputs under `cfg.output_dir` (defaults to `results/`).

## Layout

```
src/experiments/
├── __init__.py
├── __main__.py        # `python -m experiments` entry point
├── run.py             # Hydra @hydra.main decorator; dispatches to runner
├── utils.py           # seed_everything, query generation, common helpers
├── configs/           # Hydra config tree (see below)
├── runners/           # one runner per track
├── metrics/           # admissibility check, timing harness, metric collector
└── reporting/         # CSV writer, latex_tables, matplotlib figure builders
```

## `configs/` -- Hydra configuration tree

| Path | Role |
|---|---|
| [`configs/config.yaml`](configs/config.yaml) | Root config. Sets defaults for `track`, `method`, `seed`, `num_queries`, `output_dir`, `log_dir`, and the timing harness (`warmup_runs`, `num_runs`, `report_percentiles`) |
| [`configs/track/dimacs.yaml`](configs/track/dimacs.yaml) | DIMACS USA road graphs (NY, BAY, COL, FLA) -- file paths and per-graph metadata |
| [`configs/track/osmnx.yaml`](configs/track/osmnx.yaml) | OSMnx road graphs (Modena, Manhattan, Berlin, Los Angeles, Netherlands) |
| [`configs/track/warcraft.yaml`](configs/track/warcraft.yaml) | Warcraft 12x12 grid maps for the contextual variant |
| [`configs/track/cabspotting.yaml`](configs/track/cabspotting.yaml) | Cabspotting taxi traces (legacy contextual track) |
| [`configs/method/aac.yaml`](configs/method/aac.yaml) | AAC: `K0`, `m`, training schedule (epochs, batch size, lr, temperature curriculum) |
| [`configs/method/alt.yaml`](configs/method/alt.yaml) | FPS-ALT classical landmark heuristic |
| [`configs/method/dijkstra.yaml`](configs/method/dijkstra.yaml) | Dijkstra reference oracle |
| [`configs/method/fastmap.yaml`](configs/method/fastmap.yaml) | FastMap embedding (cautionary out-of-domain comparator on directed graphs) |
| [`configs/method/contextual.yaml`](configs/method/contextual.yaml) | Differentiable contextual variant for Warcraft / Cabspotting |

To override a single value from the CLI, use Hydra's dot-notation (e.g. `method.K0=128 method.m=32`). To run the full multi-budget sweep, override the `sweep_*` lists in the method config.

## `runners/` -- per-track experiment loops

Every runner subclasses [`runners/base.py`](runners/base.py) `BaseRunner`, which standardises preprocessing, query generation, the metric collection loop, output directory layout, and CSV serialisation.

| Runner | File | Track |
|---|---|---|
| `DIMACSRunner` | [`runners/dimacs_runner.py`](runners/dimacs_runner.py) | DIMACS USA road networks |
| `OSMnxRunner` | [`runners/osmnx_runner.py`](runners/osmnx_runner.py) | OSMnx city / country graphs |
| `WarcraftRunner` | [`runners/warcraft_runner.py`](runners/warcraft_runner.py) | Warcraft 12x12 contextual grids |
| `CabspottingRunner` | [`runners/cabspotting_runner.py`](runners/cabspotting_runner.py) | Cabspotting taxi-trace contextual track |

The dispatch table is in [`runners/__init__.py`](runners/__init__.py); `get_runner(track_name)` returns the right class.

## `metrics/` -- what is measured

| Module | Role |
|---|---|
| [`metrics/admissibility.py`](metrics/admissibility.py) | Per-query admissibility check (`h(u, t) ≤ d(u, t)`) -- used for the zero-violation guarantee reported throughout the paper |
| [`metrics/timing.py`](metrics/timing.py) | Warm-up + timed-phase harness for p50 / p95 query latency |
| [`metrics/collector.py`](metrics/collector.py) | Aggregates per-query records into the schema written by `reporting/csv_writer.py` |

## `reporting/` -- outputs

| Module | Role |
|---|---|
| [`reporting/csv_writer.py`](reporting/csv_writer.py) | Writes per-experiment CSVs into `results/<track>/...` (the schema each track produces is the schema the paper's Table/Figure generators expect) |
| [`reporting/latex_tables.py`](reporting/latex_tables.py) | Builds the data-driven `paper/table_*.tex` fragments from the CSVs |
| [`reporting/figures.py`](reporting/figures.py) | Builds the data-driven figures used in `paper/figures/` |

## How this fits into the wider repo

- The headline reproduction driver [`scripts/reproduce_paper.py`](../../scripts/reproduce_paper.py) calls a mixture of `python -m experiments ...` invocations and standalone scripts under `scripts/`. Use it for the full pipeline.
- For a single experiment in isolation, prefer the `python -m experiments track=... method=...` form documented above.
- For where each output ends up and which paper Table/Figure consumes it, see [`results/README.md`](../../results/README.md).

## Cross-references

- **Configuration -> runner mapping**: [`runners/__init__.py`](runners/__init__.py) (`get_runner(track_name)`).
- **Per-experiment output -> paper Table/Figure mapping**: [`../../results/README.md`](../../results/README.md).
- **Per-table provenance -> source CSV mapping**: the `%%% PAPER-TABLE-PROVENANCE` headers at the top of each `paper/table_*.tex` file. The drift detector [`../../scripts/check_paper_consistency.py`](../../scripts/check_paper_consistency.py) parses these headers and verifies every numeric cell against its CSV.
- **Per-script role index**: [`../../scripts/README.md`](../../scripts/README.md).
- **Installation and quick start**: [`../../README.md`](../../README.md).
