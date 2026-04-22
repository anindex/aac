#!/usr/bin/env python3
"""Single-command paper reproduction with built-in verification.

Orchestrates all experiment and table-generation scripts in dependency
order. Two canonical entry points:

  python scripts/reproduce_paper.py                # Full pipeline (~hours;
                                                   # runs every experiment).
  python scripts/reproduce_paper.py --tables-only  # Plot-and-table only
                                                   # (seconds; CSVs must
                                                   # already exist on disk).

The ``--tables-only`` path is the single source of truth for regenerating
every paper figure and every ``\\input``'d table from existing result
CSVs; it preflights CSV availability (failing fast with actionable
diagnostics if any prerequisite CSV is missing) and runs the
``check_paper_consistency.py`` drift detector as the final step.

Other flags:

  --track T      Run one experiment track only. The 11 valid tracks are
                 derived from STEPS: ablation, cdh, contextual, covering,
                 dimacs, figures, hybrid, osmnx, query, synthetic,
                 training_drift (plus ``verify`` for the drift check).
  --no-verify    Skip the existence/consistency verification at the end.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Experiment pipeline: (step_name, script_args, track, heavy)
#   heavy=True  -> skipped in --tables-only mode
#   track       -> which --track value enables this step ("all" = always)
# ---------------------------------------------------------------------------

STEPS: list[tuple[str, list[str], str, bool]] = [
    # ---- DIMACS core (Tables 1-3, 16; Pareto figure 5) ----
    ("DIMACS timing & Wilcoxon",
     ["scripts/run_timing_and_stats.py"],
     "dimacs", True),
    ("DIMACS Pareto sweep (5 seeds)",
     ["scripts/run_multiseed_pareto.py"],
     "dimacs", True),
    ("DIMACS matched-budget Wilcoxon (5 seeds)",
     ["scripts/run_multiseed_wilcoxon.py"],
     "dimacs", True),
    ("DIMACS per-cell matched-budget Wilcoxon (Table 16)",
     ["scripts/run_matched_budget_wilcoxon.py"],
     "dimacs", True),
    ("DIMACS FastMap admissibility (Appendix D)",
     ["scripts/run_fastmap_admissibility.py"],
     "dimacs", True),
    ("DIMACS validation-split Pareto (Table 6)",
     ["scripts/run_validation_pareto.py"],
     "dimacs", True),
    ("DIMACS per-query path-optimality audit",
     ["scripts/verify_path_optimality.py"],
     "dimacs", False),
    ("DIMACS SCC reachability audit (\u00a72.1)",
     ["scripts/check_scc_reachability.py"],
     "dimacs", False),

    # ---- OSMnx (Tables 7, 8; Section 5.4) ----
    ("OSMnx experiments (5 seeds, 4 graphs)",
     ["scripts/run_osmnx_experiments.py",
      "--graphs", "modena", "manhattan", "berlin", "los_angeles"],
     "osmnx", True),

    # ---- Ablations (Tables 21, 22, 11) ----
    ("Selection / compression / admissibility ablations",
     ["scripts/run_ablation_selection.py"],
     "ablation", True),
    ("Random-restart FPS baseline (Table 21 FPS-RR column)",
     ["scripts/run_random_restart_fps.py"],
     "ablation", True),

    # ---- Hybrid (Table 10, Table 9) ----
    ("Hybrid max(AAC, ALT) evaluation (Table 10, road graphs)",
     ["scripts/eval_hybrid.py"],
     "hybrid", True),
    ("Matched-budget hybrid on non-road graphs (Table 9)",
     ["scripts/run_matched_hybrid_nonroad.py"],
     "hybrid", True),
    ("Greedy-Max oracle on non-road graphs (Table 9 column)",
     ["scripts/run_greedy_max_nonroad.py"],
     "hybrid", True),

    # ---- Query distributions (Table 15) ----
    ("Query distribution sensitivity",
     ["scripts/run_query_distribution_experiments.py"],
     "query", True),

    # ---- Synthetic graphs (\u00a75.9.1\u20135.9.3, Table 2 non-road rows) ----
    ("Synthetic SBM + BA experiments",
     ["scripts/run_synthetic_experiments.py"],
     "synthetic", True),
    ("OGB-arXiv non-road experiment (pre-registered, Appendix E)",
     ["scripts/run_nonroad_real.py"],
     "synthetic", True),
    ("OGB-arXiv per-cell admissibility audit (Appendix E.3, 15/15 cells)",
     ["scripts/verify_ogbn_admissibility.py"],
     "synthetic", True),
    ("TOST equivalence p-values from matched-budget paired data (\u00a75.9, Appendix E)",
     ["scripts/compute_tost_equivalence.py"],
     "synthetic", False),

    # ---- CDH reference benchmark (Table 4, \u00a75.9.4) ----
    ("CDH reference benchmark on SBM/Modena/NY (Table 4)",
     ["scripts/run_cdh_baseline.py"],
     "cdh", True),
    ("CDH road-graph extension (\u00a75.9.4 supplement)",
     ["scripts/run_cdh_road.py"],
     "cdh", True),

    # ---- Same-pool first-m diagnostic (Table 17, \u00a75.10) ----
    ("ALT-pool (first m) diagnostic (Table 17)",
     ["scripts/run_same_pool_first_m.py"],
     "training_drift", True),

    # ---- Training-objective drift (\u00a75.10, Figure 6, Tables 12, 19, 20) ----
    ("Multi-graph training-drift sweep (Figure 6 data)",
     ["scripts/run_training_drift_multi.py"],
     "training_drift", True),
    ("Road-graph training-drift on NY-DIMACS (Table 19)",
     ["scripts/run_training_drift_road.py"],
     "training_drift", True),
    ("Hyperparameter and init sweep on SBM B=32 (Table 20)",
     ["scripts/run_training_drift_hp.py"],
     "training_drift", True),
    ("Inline forced-first-m diagnostic on SBM (Table 12)",
     ["scripts/diagnose_sbm_aac_alt.py"],
     "training_drift", True),

    # ---- Covering-radius computation (Theorem 5; \u00a75.7) ----
    ("Compute covering radii (Theorem 5)",
     ["scripts/compute_covering_radius.py"],
     "covering", True),

    # ---- \u03bb_uniq / R_uniq ablation (Appendix C) ----
    ("lambda_uniq ablation",
     ["scripts/run_lambda_uniq_ablation.py"],
     "ablation", True),

    # ---- Coverage-aware regularization (\u00a75.10 + Table 18) ----
    ("Coverage-aware loss on road graphs (\u00a75.10)",
     ["scripts/run_coverage_aware.py"],
     "coverage", True),
    ("Coverage-aware loss on non-road graphs (Table 18)",
     ["scripts/run_coverage_aware_nonroad.py"],
     "coverage", True),

    # ---- Contextual / Warcraft (Appendix B) ----
    ("Warcraft contextual ablation: CNN alpha_c=1 (3 seeds)",
     ["scripts/run_multiseed_ablation.py",
      "--encoder", "cnn", "--alpha-cost", "1"],
     "contextual", True),
    ("Warcraft contextual ablation: ResNet alpha_c=10 (3 seeds, Appendix B)",
     ["scripts/run_multiseed_ablation.py",
      "--encoder", "resnet", "--alpha-cost", "10"],
     "contextual", True),
    ("Warcraft gradient-flow analysis (Appendix B: 2-4 orders claim)",
     ["scripts/run_gradient_analysis.py"],
     "contextual", True),

    # ---- Table generation from CSVs (figures track) ----
    ("Generate Pareto / DIMACS tables to results/paper/",
     ["scripts/generate_paper_figures.py",
      "--output-dir", "results/paper/", "--data-dir", "results/dimacs/"],
     "figures", False),
    ("Generate OSMnx summary table (writes paper/table_osmnx.tex)",
     ["scripts/generate_table_osmnx.py"],
     "figures", False),
    ("Generate OSMnx full table (writes paper/table_osmnx_full.tex)",
     ["scripts/generate_table_osmnx_full.py"],
     "figures", False),

    # ---- Figures emitted directly to paper/figures/ (LaTeX import path) ----
    ("Pareto frontier with seed bands (Figure 5)",
     ["scripts/plot_pareto_seedbands.py",
      "--output", "paper/figures/pareto_frontier.pdf"],
     "figures", False),
    ("Multi-graph training-drift figure (Figure 6)",
     ["scripts/plot_training_drift_multi.py"],
     "figures", False),
    ("Amortized cost figure (Figure 7)",
     ["scripts/plot_amortized_cost.py"],
     "figures", False),
    ("SBM landmark placement figure (Figure 2)",
     ["scripts/generate_landmark_viz_sbm.py"],
     "figures", False),

    # ---- Landmark selection visualization (canonical paper figure) ----
    ("Landmark selection visualization (Figure 1)",
     ["scripts/generate_landmark_viz.py",
      "--output", "paper/figures/landmark_selection.pdf"],
     "figures", False),

    # ---- Toy P_7 closed-form (Section 3.5, Figure ref:toy-p7) ----
    ("Toy P_7 closed-form (gap-to-teacher vs covering radius)",
     ["scripts/run_toy_p7_gap_vs_covering.py"],
     "figures", False),
    ("Toy P_7 divergence figure (Section 3.5)",
     ["scripts/plot_toy_p7_divergence.py",
      "--output", "paper/figures/toy_p7_divergence.pdf"],
     "figures", False),

    # ---- Verification: paper-table CSV consistency drift detector ----
    ("Paper-table consistency check (numerical drift detector)",
     ["scripts/check_paper_consistency.py", "--quiet"],
     "verify", False),

    # The following supplementary-figure generators have been moved under
    # scripts/legacy/ and are not invoked from this pipeline:
    #   * scripts/legacy/generate_method_diagram.py    (the AAC pipeline
    #     diagram is now drawn inline as TikZ in paper/main.tex; the
    #     legacy paper/figures/method_overview.pdf was also retired)
    #   * scripts/legacy/generate_expansion_heatmap.py (replaced by tab:hybrid)
    #   * scripts/legacy/generate_heuristic_contour.py
    #   * scripts/legacy/generate_search_comparison.py
    # See scripts/legacy/README.md.  The pipeline above is sufficient to
    # reproduce every figure cited in main.tex.
]

ALL_TRACKS = sorted({t for _, _, t, _ in STEPS})

# Expected outputs for verification.
#
# TABLES: every paper/table_*.tex \input by paper/main.tex.  All 20 tables are
# tracked in git.  Some are written directly by generators (table_osmnx*.tex);
# the rest are hand-edited LaTeX whose data comes from CSVs and the
# verification step only checks that the file exists and is non-empty.  The
# two "results/paper/*.tex" entries below are the output of
# generate_paper_figures.py, which writes to results/paper/ rather than
# paper/ (the paper/ versions are hand-edited).
EXPECTED_TEX = [
    # Headline matched-memory comparison (Table 2; \input by main.tex)
    "paper/table_main_matched_memory.tex",
    # CDH reference benchmark (Table 4; \input by main.tex)
    "paper/table_cdh_reference.tex",
    # DIMACS retrospective + appendix tables
    "paper/table_main_dimacs.tex",
    "paper/table_pareto_detail.tex",
    "paper/table_valsplit.tex",
    "paper/table_dimacs_wilcoxon_percell.tex",
    "paper/table_multi_axis_cost.tex",
    # OSMnx tables (produced by generate_table_osmnx*.py)
    "paper/table_osmnx.tex",
    "paper/table_osmnx_full.tex",
    # Hybrid tables
    "paper/table_hybrid.tex",
    "paper/table_matched_hybrid.tex",
    # Ablation tables
    "paper/table_ablation_selection.tex",
    "paper/table_ablation_compression.tex",
    "paper/table_ablation_admissibility.tex",
    "paper/table_query_distribution.tex",
    # Training-drift and same-pool diagnostics
    "paper/table_same_pool_firstm.tex",
    "paper/table_training_drift_road.tex",
    "paper/table_training_drift_hp.tex",
    # Coverage-aware non-road sweep
    "paper/table_coverage_aware_nonroad.tex",
    # Warcraft contextual
    "paper/table_contextual_ablation.tex",
]

EXPECTED_CSV = [
    # DIMACS core (timing + significance + Pareto + paired audit)
    "results/dimacs/timing_p50_p95.csv",
    "results/dimacs/preprocessing_breakdown.csv",
    "results/dimacs/per_query_paired.csv",
    "results/dimacs/wilcoxon_pvalues.csv",
    "results/dimacs/wilcoxon_pvalues_multiseed.csv",
    "results/dimacs/wilcoxon_stouffer.csv",
    "results/dimacs/wilcoxon_matched_budget.csv",
    "results/dimacs/pareto_sweep_NY.csv",            # legacy compat alias
    "results/dimacs/pareto_sweep_FLA.csv",           # legacy compat alias
    "results/dimacs/pareto_sweep_NY_multiseed.csv",  # canonical aggregate
    "results/dimacs/pareto_sweep_FLA_multiseed.csv", # canonical aggregate
    "results/dimacs/pareto_sweep_NY_perseed.csv",    # consumed by plot_pareto_seedbands.py
    "results/dimacs/pareto_sweep_FLA_perseed.csv",   # consumed by plot_pareto_seedbands.py
    "results/dimacs/pareto_valsplit_NY.csv",
    "results/dimacs/pareto_valsplit_FLA.csv",
    "results/dimacs/path_optimality_audit.csv",
    "results/dimacs/fastmap_admissibility.csv",
    "results/dimacs/fastmap_admissibility_summary.csv",
    "results/dimacs/aac_NY.csv",
    "results/dimacs/aac_BAY.csv",
    "results/dimacs/aac_COL.csv",
    "results/dimacs/aac_FLA.csv",
    # OSMnx (per-seed + aggregated)
    "results/osmnx/large_scale_results.csv",
    "results/osmnx/large_scale_results_agg.csv",
    "results/osmnx/pareto_valsplit_modena.csv",
    "results/osmnx/pareto_valsplit_manhattan.csv",
    # Ablations
    "results/ablation_selection/exp1_selection_strategy.csv",
    "results/ablation_selection/exp2_admissibility_robustness.csv",
    "results/ablation_selection/exp3_compression_curve.csv",
    "results/random_restart_fps/modena.csv",
    "results/random_restart_fps/manhattan.csv",
    # Hybrid
    "results/hybrid/hybrid_evaluation.csv",
    "results/hybrid_nonroad/matched_budget_hybrid.csv",
    "results/greedy_max_nonroad/greedy_max.csv",
    # Regularizers
    "results/lambda_uniq_ablation/modena_results.csv",
    "results/coverage_aware/coverage_aware_results.csv",
    "results/coverage_aware/coverage_aware_nonroad_results.csv",
    "results/coverage_aware/coverage_aware_nonroad_ba.csv",
    "results/coverage_aware/coverage_aware_nonroad_sbm_ba.csv",
    # Query distributions (per-cell + summary)
    "results/query_distributions/query_mode_results.csv",
    "results/query_distributions/query_mode_summary.csv",
    # Synthetic + non-road
    "results/synthetic/community_results.csv",
    "results/synthetic/powerlaw_results.csv",
    "results/synthetic/ogbn_arxiv_results.csv",
    # TOST equivalence outputs (computed from matched_budget_hybrid.csv via
    # scripts/compute_tost_equivalence.py); back the paper's "TOST accepts at
    # B=128 only" / "no cell achieves equivalence" claims (Sections 5.9, E).
    "results/synthetic/tost_equivalence.csv",
    # Per-cell admissibility audit on OGB-arXiv (5 seeds x 3 budgets);
    # backs the paper's "15/15 cells, zero admissibility violations" claim
    # (Section 5.9.3 / Appendix E.3).
    "results/synthetic/ogbn_arxiv_admissibility.csv",
    # CDH reference benchmark (Table tab:cdh-reference: SBM / Modena / NY)
    "results/cdh_baseline/sbm_cdh.csv",
    "results/cdh_baseline/modena_cdh.csv",
    "results/cdh_baseline/ny_cdh.csv",
    # Same-pool first-m diagnostic (Table tab:same-pool-firstm: 7 graphs)
    "results/same_pool_first_m/sbm.csv",
    "results/same_pool_first_m/ba.csv",
    "results/same_pool_first_m/NY.csv",
    "results/same_pool_first_m/modena.csv",
    "results/same_pool_first_m/manhattan.csv",
    "results/same_pool_first_m/berlin.csv",
    "results/same_pool_first_m/los_angeles.csv",
    # Training drift (Figure fig:training-drift + Tables 19, 20)
    "results/training_drift_road/drift_ny_B64.csv",
    "results/training_drift_hp/sbm_b32_results.csv",
    "results/training_drift_multi/drift_sbm_B32.csv",
    "results/training_drift_multi/drift_sbm_B64.csv",
    "results/training_drift_multi/drift_ba_B32.csv",
    "results/training_drift_multi/drift_ba_B64.csv",
    # Warcraft contextual (Table tab:contextual-ablation + Appendix B claims)
    # Warcraft canonical CNN ablation: the canonical CNN+alpha=1 run writes
    # the three legacy aliases below; tagged duplicates (cnn_a1_*.csv) would
    # also be written by a fresh canonical run but are not preflighted here
    # because the on-disk historical record only retains the legacy aliases.
    "results/warcraft/ablation_results_multiseed.csv",            # CNN alpha_c=1 aggregate (canonical alias; consumed by paper/table_contextual_ablation.tex)
    "results/warcraft/ablation_results.csv",                       # CNN alpha_c=1 backward-compat alias
    "results/warcraft/ablation_results_perseed.csv",               # CNN alpha_c=1 per-seed
    # Warcraft ResNet ablation (Appendix B ResNet row): tagged outputs
    # are the canonical record (no legacy aliases for non-canonical runs).
    "results/warcraft/ablation_results_resnet_a10_multiseed.csv",  # ResNet alpha_c=10 aggregate (consumed by paper/table_contextual_ablation.tex)
    "results/warcraft/ablation_results_resnet_a10_perseed.csv",    # ResNet alpha_c=10 per-seed
    "results/warcraft/ablation_results_resnet_a10.csv",            # ResNet alpha_c=10 flat summary
    "results/warcraft/gradient_flow.csv",                          # backs "compressor gradient
                                                                    # norms are 2--4 orders of
                                                                    # magnitude smaller than
                                                                    # encoder norms" (Appendix B)
    # Covering-radius computation
    "results/covering_radius.csv",
    # Toy P_7 closed-form (Section 3.5, Figure ref:toy-p7)
    "results/toy_p7/all_subsets.csv",
    "results/toy_p7/highlight.csv",
    # SCC reachability audit (Section 2.1)
    "results/scc_reachability.csv",
]

# JSON outputs that the paper cites or that document a baseline reproduction.
EXPECTED_JSON = [
    # DataSP reproduction reported in Appendix B (cost regret 0.384, path-match
    # 10.8%); kept as a JSON snapshot of the third-party reproduction.
    "results/warcraft/datasp_reproduced.json",
    "results/warcraft/datasp_fair_eval.json",
]

# Figures \includegraphics-d by main.tex (must live in paper/figures/).
# All are git-tracked. EXPECTED_PDF_GENERATED are produced by the figure
# scripts under scripts/ (see the "figures" track in STEPS); the verification
# step regenerates and re-checks them when --tables-only is used.
EXPECTED_PDF_GENERATED = [
    "paper/figures/pareto_frontier.pdf",
    "paper/figures/training_drift.pdf",
    "paper/figures/amortized_cost.pdf",
    "paper/figures/sbm_landmark_placement.pdf",
    "paper/figures/landmark_selection.pdf",
    "paper/figures/toy_p7_divergence.pdf",
]
# Note: the legacy paper/figures/method_overview.pdf was retired in the
# The AAC pipeline diagram is now drawn inline
# as a TikZ picture in paper/main.tex (Figure~\ref{fig:pipeline}), so it is
# font-matched to the body text and version-controlled.
EXPECTED_PDF_CURATED: list[str] = []
EXPECTED_PDF = EXPECTED_PDF_GENERATED + EXPECTED_PDF_CURATED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_step(cmd: list[str], description: str, env: dict) -> bool:
    """Run a subprocess command with status banner. Returns True on success."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    full_cmd = [sys.executable] + cmd
    print(f"  $ {' '.join(full_cmd)}")
    result = subprocess.run(full_cmd, env=env)
    if result.returncode == 0:
        print(f"  [OK] {description}")
        return True
    print(f"  [FAIL] exit code {result.returncode}")
    return False


def check_file(path: str) -> tuple[bool, str]:
    """Check that a file exists with non-zero size."""
    if not os.path.exists(path):
        return False, "MISSING"
    size = os.path.getsize(path)
    if size == 0:
        return False, "EMPTY (0 bytes)"
    return True, f"OK ({size:,} bytes)"


# Per-CSV -> producing track. Used by the --tables-only preflight to print
# actionable "Run: python scripts/reproduce_paper.py --track <T>" messages
# when an upstream experiment CSV is missing.
CSV_TO_TRACK: dict[str, str] = {
    # DIMACS core
    "results/dimacs/timing_p50_p95.csv": "dimacs",
    "results/dimacs/preprocessing_breakdown.csv": "dimacs",
    "results/dimacs/per_query_paired.csv": "dimacs",
    "results/dimacs/wilcoxon_pvalues.csv": "dimacs",
    "results/dimacs/wilcoxon_pvalues_multiseed.csv": "dimacs",
    "results/dimacs/wilcoxon_stouffer.csv": "dimacs",
    "results/dimacs/wilcoxon_matched_budget.csv": "dimacs",
    "results/dimacs/pareto_sweep_NY.csv": "dimacs",
    "results/dimacs/pareto_sweep_FLA.csv": "dimacs",
    "results/dimacs/pareto_sweep_NY_multiseed.csv": "dimacs",
    "results/dimacs/pareto_sweep_FLA_multiseed.csv": "dimacs",
    "results/dimacs/pareto_sweep_NY_perseed.csv": "dimacs",
    "results/dimacs/pareto_sweep_FLA_perseed.csv": "dimacs",
    "results/dimacs/pareto_valsplit_NY.csv": "dimacs",
    "results/dimacs/pareto_valsplit_FLA.csv": "dimacs",
    "results/dimacs/path_optimality_audit.csv": "dimacs",
    "results/dimacs/fastmap_admissibility.csv": "dimacs",
    "results/dimacs/fastmap_admissibility_summary.csv": "dimacs",
    "results/dimacs/aac_NY.csv": "dimacs",
    "results/dimacs/aac_BAY.csv": "dimacs",
    "results/dimacs/aac_COL.csv": "dimacs",
    "results/dimacs/aac_FLA.csv": "dimacs",
    # OSMnx
    "results/osmnx/large_scale_results.csv": "osmnx",
    "results/osmnx/large_scale_results_agg.csv": "osmnx",
    "results/osmnx/pareto_valsplit_modena.csv": "dimacs",
    "results/osmnx/pareto_valsplit_manhattan.csv": "dimacs",
    # Ablations
    "results/ablation_selection/exp1_selection_strategy.csv": "ablation",
    "results/ablation_selection/exp2_admissibility_robustness.csv": "ablation",
    "results/ablation_selection/exp3_compression_curve.csv": "ablation",
    "results/random_restart_fps/modena.csv": "ablation",
    "results/random_restart_fps/manhattan.csv": "ablation",
    # Hybrid
    "results/hybrid/hybrid_evaluation.csv": "hybrid",
    "results/hybrid_nonroad/matched_budget_hybrid.csv": "hybrid",
    "results/greedy_max_nonroad/greedy_max.csv": "hybrid",
    # Regularizers
    "results/lambda_uniq_ablation/modena_results.csv": "ablation",
    "results/coverage_aware/coverage_aware_results.csv": "coverage",
    "results/coverage_aware/coverage_aware_nonroad_results.csv": "coverage",
    "results/coverage_aware/coverage_aware_nonroad_ba.csv": "coverage",
    "results/coverage_aware/coverage_aware_nonroad_sbm_ba.csv": "coverage",
    # Query distributions
    "results/query_distributions/query_mode_results.csv": "query",
    "results/query_distributions/query_mode_summary.csv": "query",
    # Synthetic + non-road
    "results/synthetic/community_results.csv": "synthetic",
    "results/synthetic/powerlaw_results.csv": "synthetic",
    "results/synthetic/ogbn_arxiv_results.csv": "synthetic",
    "results/synthetic/tost_equivalence.csv": "synthetic",
    "results/synthetic/ogbn_arxiv_admissibility.csv": "synthetic",
    # CDH reference benchmark
    "results/cdh_baseline/sbm_cdh.csv": "cdh",
    "results/cdh_baseline/modena_cdh.csv": "cdh",
    "results/cdh_baseline/ny_cdh.csv": "cdh",
    # Same-pool first-m diagnostic
    "results/same_pool_first_m/sbm.csv": "training_drift",
    "results/same_pool_first_m/ba.csv": "training_drift",
    "results/same_pool_first_m/NY.csv": "training_drift",
    "results/same_pool_first_m/modena.csv": "training_drift",
    "results/same_pool_first_m/manhattan.csv": "training_drift",
    "results/same_pool_first_m/berlin.csv": "training_drift",
    "results/same_pool_first_m/los_angeles.csv": "training_drift",
    # Training drift
    "results/training_drift_road/drift_ny_B64.csv": "training_drift",
    "results/training_drift_hp/sbm_b32_results.csv": "training_drift",
    "results/training_drift_multi/drift_sbm_B32.csv": "training_drift",
    "results/training_drift_multi/drift_sbm_B64.csv": "training_drift",
    "results/training_drift_multi/drift_ba_B32.csv": "training_drift",
    "results/training_drift_multi/drift_ba_B64.csv": "training_drift",
    # Warcraft contextual
    "results/warcraft/ablation_results_multiseed.csv": "contextual",
    "results/warcraft/ablation_results.csv": "contextual",
    "results/warcraft/ablation_results_perseed.csv": "contextual",
    "results/warcraft/ablation_results_resnet_a10_multiseed.csv": "contextual",
    "results/warcraft/ablation_results_resnet_a10_perseed.csv": "contextual",
    "results/warcraft/ablation_results_resnet_a10.csv": "contextual",
    "results/warcraft/gradient_flow.csv": "contextual",
    # Roots
    "results/covering_radius.csv": "covering",
    "results/scc_reachability.csv": "dimacs",
    "results/toy_p7/all_subsets.csv": "figures",
    "results/toy_p7/highlight.csv": "figures",
}


def preflight_tables_only() -> bool:
    """Verify every input CSV/JSON exists before running --tables-only.

    Returns True iff every prerequisite is on disk; otherwise prints a
    grouped, actionable diagnostic listing the track that produces each
    missing artifact and exits via the caller.
    """
    missing: list[tuple[str, str]] = []  # (path, track)
    for path in EXPECTED_CSV:
        if not os.path.exists(path):
            track = CSV_TO_TRACK.get(path, "<unknown>")
            missing.append((path, track))
    for path in EXPECTED_JSON:
        if not os.path.exists(path):
            missing.append((path, "contextual"))
    if not missing:
        return True
    print("\n" + "=" * 60)
    print("  PREFLIGHT FAILED: --tables-only requires existing CSVs")
    print("=" * 60)
    by_track: dict[str, list[str]] = {}
    for path, track in missing:
        by_track.setdefault(track, []).append(path)
    for track, paths in sorted(by_track.items()):
        print(f"\n  Missing inputs from track '{track}':")
        for p in paths:
            print(f"    - {p}")
        if track != "<unknown>":
            print(
                f"    Run: python scripts/reproduce_paper.py --track {track}"
            )
    print(
        "\n  --tables-only regenerates plots and tables from existing CSVs;\n"
        "  it does not run heavy experiments. Re-run the listed track(s)\n"
        "  first (or the full pipeline) and then re-invoke --tables-only.\n"
    )
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce all paper artifacts with built-in verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tracks: " + ", ".join(ALL_TRACKS) + "\n\n"
            "Examples:\n"
            "  python scripts/reproduce_paper.py                    # Full\n"
            "  python scripts/reproduce_paper.py --tables-only      # Fast\n"
            "  python scripts/reproduce_paper.py --track dimacs      # One track\n"
            "  python scripts/reproduce_paper.py --track figures     # Regen figs\n"
        ),
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Skip heavy experiments; regenerate tables/figures from existing CSVs",
    )
    parser.add_argument(
        "--track",
        choices=ALL_TRACKS,
        default=None,
        help="Run only the specified experiment track",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the output-existence verification step",
    )
    args = parser.parse_args()

    python = sys.executable
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    results: dict[str, str] = {}

    print("\n" + "=" * 60)
    print("  REPRODUCE PAPER ARTIFACTS")
    print("=" * 60)
    if args.tables_only:
        print("  Mode: --tables-only (regenerate from existing CSVs)")
    elif args.track:
        print(f"  Mode: --track {args.track}")
    else:
        print("  Mode: full reproduction (all tracks)")
    print("=" * 60)

    # ---- CSV preflight (only when --tables-only) ----
    # --tables-only regenerates plots and tables from existing CSVs only;
    # if any prerequisite CSV is missing we abort fast with an actionable
    # per-track diagnostic so users don't see opaque downstream failures.
    if args.tables_only and not preflight_tables_only():
        return 1

    # ---- Run pipeline ----
    for name, cmd, track, heavy in STEPS:
        # Decide whether to run this step
        if args.tables_only and heavy:
            results[name] = "SKIP"
            continue
        if args.track and track != args.track and track != "all":
            # In single-track mode, also run "figures" track after experiment
            if not (args.track != "figures" and track == "figures"):
                results[name] = "SKIP"
                continue

        ok = run_step(cmd, name, env)
        results[name] = "PASS" if ok else "FAIL"

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("  REPRODUCTION SUMMARY")
    print(f"{'=' * 60}")
    for name, status in results.items():
        marker = {"PASS": "[+]", "FAIL": "[X]", "SKIP": "[ ]"}[status]
        print(f"  {marker} {name:<45s} {status}")

    any_failure = "FAIL" in results.values()

    # ---- Verification ----
    if args.no_verify:
        print("\n  Verification: SKIPPED (--no-verify)")
    else:
        print(f"\n{'=' * 60}")
        print("  OUTPUT VERIFICATION")
        print(f"{'=' * 60}")
        all_expected = (
            [(p, ".tex") for p in EXPECTED_TEX]
            + [(p, ".csv") for p in EXPECTED_CSV]
            + [(p, ".json") for p in EXPECTED_JSON]
            + [(p, ".pdf") for p in EXPECTED_PDF]
        )
        verify_ok = True
        for path, ext in all_expected:
            ok, detail = check_file(path)
            status = detail if ok else f"FAIL: {detail}"
            print(f"  {path:<55s} {status}")
            if not ok:
                verify_ok = False

        if not verify_ok:
            any_failure = True

    # ---- Final ----
    print(f"\n{'=' * 60}")
    if any_failure:
        print("  Result: SOME CHECKS FAILED")
    else:
        print("  Result: ALL CHECKS PASSED")
    print(f"{'=' * 60}")
    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
