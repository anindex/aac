# `scripts/` -- top-level runners, generators, and the reproduction driver

This directory holds every active script invoked by the canonical paper
reproduction pipeline. For each entry below, the **Output**
column points at the result file under `results/` (or `paper/figures/` for
figure scripts) that the script is responsible for producing.

For where each output is consumed by the paper, see
[`../results/README.md`](../results/README.md). For the per-table
provenance / drift verification protocol, see
[`check_paper_consistency.py`](check_paper_consistency.py).

## Reproduction driver

| Script | Purpose |
|---|---|
| [`reproduce_paper.py`](reproduce_paper.py) | Single-command paper reproduction. `--tables-only` regenerates plots and tables from existing CSVs (with a CSV preflight); the default mode runs every experiment from scratch. |

## Experiment runners (referenced from `reproduce_paper.py` STEPS)

DIMACS road networks:

| Script | Output |
|---|---|
| [`run_timing_and_stats.py`](run_timing_and_stats.py) | `results/dimacs/{timing_p50_p95,preprocessing_breakdown,per_query_paired,wilcoxon_pvalues,aac_*}.csv` |
| [`run_multiseed_pareto.py`](run_multiseed_pareto.py) | `results/dimacs/pareto_sweep_{NY,FLA}{,_multiseed,_perseed}.csv` |
| [`run_multiseed_wilcoxon.py`](run_multiseed_wilcoxon.py) | `results/dimacs/wilcoxon_pvalues_multiseed.csv` |
| [`run_matched_budget_wilcoxon.py`](run_matched_budget_wilcoxon.py) | `results/dimacs/{wilcoxon_matched_budget,wilcoxon_stouffer}.csv` |
| [`run_validation_pareto.py`](run_validation_pareto.py) | `results/{dimacs,osmnx}/pareto_valsplit_*.csv` |
| [`run_fastmap_admissibility.py`](run_fastmap_admissibility.py) | `results/dimacs/fastmap_admissibility{,_summary}.csv` |

OSMnx city / country graphs:

| Script | Output |
|---|---|
| [`run_osmnx_experiments.py`](run_osmnx_experiments.py) | `results/osmnx/large_scale_results{,_agg}.csv` |

Synthetic + non-road:

| Script | Output |
|---|---|
| [`run_synthetic_experiments.py`](run_synthetic_experiments.py) | `results/synthetic/{community,powerlaw}_results.csv` |
| [`run_nonroad_real.py`](run_nonroad_real.py) | `results/synthetic/ogbn_arxiv_results.csv` |
| [`compute_tost_equivalence.py`](compute_tost_equivalence.py) | `results/synthetic/tost_equivalence.csv` (TOST equivalence p-values at $\delta{=}1$ pp; backs paper Sec.5.9 / Appendix E TOST claims) |
| [`compute_covering_radius.py`](compute_covering_radius.py) | `results/covering_radius.csv` |
| [`run_toy_p7_gap_vs_covering.py`](run_toy_p7_gap_vs_covering.py) | `results/toy_p7/{all_subsets,highlight}.csv` (closed-form 1D toy backing Figure `fig:toy-p7` in Sec.3.5) |

Hybrid + Greedy-Max:

| Script | Output |
|---|---|
| [`eval_hybrid.py`](eval_hybrid.py) | `results/hybrid/hybrid_evaluation.csv` |
| [`run_matched_hybrid_nonroad.py`](run_matched_hybrid_nonroad.py) | `results/hybrid_nonroad/matched_budget_hybrid.csv` |
| [`run_greedy_max_nonroad.py`](run_greedy_max_nonroad.py) | `results/greedy_max_nonroad/greedy_max.csv` |

CDH reference benchmark:

| Script | Output |
|---|---|
| [`run_cdh_baseline.py`](run_cdh_baseline.py) | `results/cdh_baseline/sbm_cdh.csv` (SBM, undirected) |
| [`run_cdh_road.py`](run_cdh_road.py) | `results/cdh_baseline/{modena,ny}_cdh.csv` (directed road graphs cited in the paper) |

Ablations and regularizers:

| Script | Output |
|---|---|
| [`run_ablation_selection.py`](run_ablation_selection.py) | `results/ablation_selection/exp{1,2,3}_*.csv` (selection / admissibility / compression curve) |
| [`run_random_restart_fps.py`](run_random_restart_fps.py) | `results/random_restart_fps/{modena,manhattan}.csv` |
| [`run_lambda_uniq_ablation.py`](run_lambda_uniq_ablation.py) | `results/lambda_uniq_ablation/modena_results.csv` |
| [`run_coverage_aware.py`](run_coverage_aware.py) | `results/coverage_aware/coverage_aware_results.csv` (road graphs) |
| [`run_coverage_aware_nonroad.py`](run_coverage_aware_nonroad.py) | `results/coverage_aware/coverage_aware_nonroad_*.csv` |
| [`run_query_distribution_experiments.py`](run_query_distribution_experiments.py) | `results/query_distributions/query_mode_{results,summary}.csv` |

Training-objective drift:

| Script | Output |
|---|---|
| [`diagnose_sbm_aac_alt.py`](diagnose_sbm_aac_alt.py) | Inline forced-first-m diagnostic (Table `tab:training-drift`); no CSV |
| [`run_same_pool_first_m.py`](run_same_pool_first_m.py) | `results/same_pool_first_m/{sbm,ba,NY,modena,manhattan,berlin,los_angeles}.csv` |
| [`run_training_drift_multi.py`](run_training_drift_multi.py) | `results/training_drift_multi/drift_{sbm,ba}_B{32,64}.csv` |
| [`run_training_drift_road.py`](run_training_drift_road.py) | `results/training_drift_road/drift_ny_B64.csv` |
| [`run_training_drift_hp.py`](run_training_drift_hp.py) | `results/training_drift_hp/sbm_b32_results.csv` |

Contextual (Warcraft):

| Script | Output |
|---|---|
| [`run_multiseed_ablation.py`](run_multiseed_ablation.py) | `results/warcraft/ablation_results_{cnn_a1,resnet_a10}_{multiseed,perseed}.csv` CLI flags `--encoder {cnn,resnet}` and `--alpha-cost FLOAT` select the encoder family and cost-supervision weight; the paper's CNN row uses `--encoder cnn --alpha-cost 1` and the ResNet row uses `--encoder resnet --alpha-cost 10` (Table `tab:contextual-ablation`). |
| [`run_gradient_analysis.py`](run_gradient_analysis.py) | `results/warcraft/gradient_flow.csv` (per-epoch encoder-vs-compressor gradient norms; backs Appendix B's "compressor gradient norms are 2--4 orders of magnitude smaller than encoder norms" claim via the ratio `encoder_grad_norm / compressor_grad_norm`). |

## Figure / table generators (regenerated under `--tables-only`)

| Script | Output |
|---|---|
| [`generate_table_osmnx.py`](generate_table_osmnx.py) | `paper/table_osmnx.tex` (the only canonical CSV -> TeX generator targeting `paper/`) |
| [`generate_table_osmnx_full.py`](generate_table_osmnx_full.py) | `paper/table_osmnx_full.tex` |
| [`generate_paper_figures.py`](generate_paper_figures.py) | `results/paper/{pareto_frontier.pdf,latency_comparison.pdf,table_main_dimacs.tex,table_pareto_detail.tex}` (a transient parallel-tree of artifacts; `paper/` copies are hand-maintained) |
| [`plot_pareto_seedbands.py`](plot_pareto_seedbands.py) | `paper/figures/pareto_frontier.pdf` (Figure `fig:pareto`) |
| [`plot_training_drift_multi.py`](plot_training_drift_multi.py) | `paper/figures/training_drift.pdf` (Figure `fig:training-drift`) |
| [`plot_amortized_cost.py`](plot_amortized_cost.py) | `paper/figures/amortized_cost.pdf` (Figure `fig:amortized-cost`) |
| [`generate_landmark_viz.py`](generate_landmark_viz.py) | `paper/figures/landmark_selection.pdf` (Figure `fig:landmarks`) |
| [`generate_landmark_viz_sbm.py`](generate_landmark_viz_sbm.py) | `paper/figures/sbm_landmark_placement.pdf` (Figure `fig:landmarks-sbm`) |
| [`plot_toy_p7_divergence.py`](plot_toy_p7_divergence.py) | `paper/figures/toy_p7_divergence.pdf` (Figure `fig:toy-p7`, Sec.3.5; reads `results/toy_p7/highlight.csv`) |

## Verification (always runs unless `--no-verify`)

| Script | Purpose |
|---|---|
| [`verify_path_optimality.py`](verify_path_optimality.py) | Audits per-query path optimality across DIMACS aac CSVs; writes `results/dimacs/path_optimality_audit.csv` |
| [`verify_ogbn_admissibility.py`](verify_ogbn_admissibility.py) | OGB-arXiv per-cell admissibility audit (5 seeds x 3 budgets); writes `results/synthetic/ogbn_arxiv_admissibility.csv` (Appendix E.3) |
| [`check_scc_reachability.py`](check_scc_reachability.py) | Section 2.1 SCC reachability audit; writes `results/scc_reachability.csv` |
| [`check_paper_consistency.py`](check_paper_consistency.py) | Drift detector that cross-checks every `paper/table_*.tex` cell against its CSV via the `%%% PAPER-TABLE-PROVENANCE` headers |

## Bootstrap (one-off dataset preparation; not in `STEPS`)

These run once per fresh clone to materialise the input datasets under
`data/`. They are not invoked by `reproduce_paper.py`.

| Script | Purpose |
|---|---|
| [`download_dimacs.py`](download_dimacs.py) | Downloads the DIMACS 9th Challenge USA road graph files (NY, BAY, COL, FLA) to `data/dimacs/` |
| [`download_osmnx.py`](download_osmnx.py) | Downloads and caches the OSMnx city / country graphs (Modena, Manhattan, Berlin, Los Angeles, Netherlands) to `data/osmnx/*.npz` |
| [`generate_warcraft_data.py`](generate_warcraft_data.py) | Generates synthetic Warcraft 12x12 terrain maps for the contextual experiments |
