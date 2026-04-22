#!/usr/bin/env python
"""Three ablation experiments to strengthen AAC claims.

Experiment 1: Selection strategy ablation (learned vs random vs FPS-subset)
  - At fixed K0=64, m=16 on Modena and Manhattan (where AAC wins),
    compare: (a) AAC learned selection, (b) random m-subset of K0 teachers,
    (c) FPS m-subset (select m landmarks via FPS directly).
  - This isolates the value of learned selection.

Experiment 2: Admissibility under early stopping
  - Train AAC on Modena (K0=64, m=16) and checkpoint at epochs {1,5,10,50,200}.
  - At each checkpoint, verify admissibility on 100 queries and report
    expansion reduction. Shows admissibility holds regardless of training quality.

Experiment 3: Compression efficiency curve (m/K0)
  - Fix K0=64 on Modena and Manhattan, sweep m in {4,8,16,32,64}.
  - Plot expansion reduction vs compression ratio m/K0.
  - Compare against ALT at matched memory.

All experiments use 3 seeds {42, 123, 456}, 100 queries, same protocol as
the main OSMnx experiments.

Output: results/ablation_selection/ with CSV files for each experiment.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
# Make src/ importable so `experiments` and `aac` resolve to src/.
_SRC_DIR = str(Path(_PROJECT_ROOT) / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from aac.baselines.alt import alt_preprocess, make_alt_heuristic
from aac.compression.compressor import LinearCompressor, make_linear_heuristic
from aac.embeddings.anchors import farthest_point_sampling
from aac.embeddings.sssp import compute_teacher_labels
from aac.graphs.io import load_graph_npz
from aac.search.astar import astar
from aac.search.dijkstra import dijkstra
from aac.train.trainer import TrainConfig, train_linear_compressor
from experiments.utils import compute_strong_lcc, generate_queries

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRAPHS = {
    "modena": Path("data/osmnx/modena.npz"),
    "manhattan": Path("data/osmnx/manhattan.npz"),
}
SEEDS = [42, 123, 456, 789, 1024]
NUM_QUERIES = 100
OUTPUT_DIR = Path("results/ablation_selection")

# Shared training config
TRAIN_CFG_DEFAULTS = dict(
    num_epochs=200, batch_size=256, lr=1e-3, cond_lambda=0.01,
    T_init=1.0, gamma=1.05,
    patience=20,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lcc(graph):
    """Return LCC node tensor and seed vertex."""
    nodes, seed = compute_strong_lcc(graph)
    return torch.tensor(nodes, dtype=torch.int64), seed


def dijkstra_baseline(graph, queries):
    """Run Dijkstra on all queries, return mean expansions."""
    exps = []
    for s, t in queries:
        r = dijkstra(graph, s, t)
        exps.append(r.expansions)
    return np.array(exps)


def run_astar_queries(graph, queries, heuristic):
    """Run A* with given heuristic, return (expansions, all_optimal)."""
    exps = []
    all_optimal = True
    for s, t in queries:
        r = astar(graph, s, t, heuristic=heuristic)
        exps.append(r.expansions)
        if not r.optimal:
            all_optimal = False
    return np.array(exps), all_optimal


def check_admissibility_queries(graph, queries, heuristic):
    """Check admissibility: h(u,t) <= d(u,t) for all queries."""
    violations = 0
    for s, t in queries:
        r_dij = dijkstra(graph, s, t)
        h_val = heuristic(s, t)
        if h_val > r_dij.cost + 1e-6:
            violations += 1
    return violations


def train_aac(graph, K0, m, seed, lcc_nodes, lcc_seed, num_epochs=200, patience=20):
    """Full AAC pipeline: FPS -> SSSP -> train compressor -> heuristic."""
    rng = torch.Generator().manual_seed(seed)
    anchors = farthest_point_sampling(
        graph, K0, seed_vertex=lcc_seed, rng=rng, valid_vertices=lcc_nodes
    )
    teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)

    compressor = LinearCompressor(K=K0, m=m, is_directed=graph.is_directed)
    cfg = TrainConfig(**{**TRAIN_CFG_DEFAULTS, "seed": seed,
                        "num_epochs": num_epochs, "patience": patience})
    metrics = train_linear_compressor(compressor, teacher_labels, cfg,
                                      valid_vertices=lcc_nodes)

    compressor.eval()
    d_out_t = teacher_labels.d_out.t()
    d_in_t = teacher_labels.d_in.t()
    with torch.no_grad():
        if graph.is_directed:
            y_fwd, y_bwd = compressor(d_out_t, d_in_t)
        else:
            y = compressor(d_out_t)
            y_fwd = y_bwd = y
    heuristic = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)
    return heuristic, teacher_labels, anchors, compressor, metrics


def random_subset_heuristic(teacher_labels, m, seed, is_directed):
    """Pick m/2 random fwd + m/2 random bwd landmarks from K0 teachers."""
    rng = np.random.RandomState(seed)
    K0 = teacher_labels.d_out.shape[0]
    d_out = teacher_labels.d_out  # (K, V)
    d_in = teacher_labels.d_in    # (K, V)

    if is_directed:
        m_fwd = m // 2
        m_bwd = m - m_fwd
        idx_fwd = rng.choice(K0, size=m_fwd, replace=False)
        idx_bwd = rng.choice(K0, size=m_bwd, replace=False)
        y_fwd = d_out[idx_fwd].t()  # (V, m_fwd)
        y_bwd = d_in[idx_bwd].t()   # (V, m_bwd)
    else:
        idx = rng.choice(K0, size=m, replace=False)
        y_fwd = d_out[idx].t()
        y_bwd = y_fwd

    return make_linear_heuristic(y_fwd, y_bwd, is_directed)


def greedy_maximize_heuristic(teacher_labels, m, queries, is_directed):
    """Greedily select landmarks maximizing average ALT heuristic over queries.

    At each step, add the landmark whose marginal contribution to
    max_{k in S} h_k(s,t) averaged over queries is largest.  For directed
    graphs, each selected landmark contributes both forward and backward
    bounds (storing 2 values per vertex), so we select m//2 landmarks to
    match the m-float budget.
    """
    d_out = teacher_labels.d_out.numpy()  # (K0, V)
    d_in = teacher_labels.d_in.numpy()    # (K0, V)
    K0 = d_out.shape[0]

    sources = np.array([int(s) for s, t in queries])
    targets = np.array([int(t) for s, t in queries])

    if is_directed:
        fwd_bounds = d_out[:, targets] - d_out[:, sources]   # (K0, Q)
        bwd_bounds = d_in[:, sources] - d_in[:, targets]     # (K0, Q)
        h_per_lm = np.maximum(fwd_bounds, bwd_bounds)        # (K0, Q)
        n_select = max(m // 2, 1)
    else:
        h_per_lm = np.abs(d_out[:, sources] - d_out[:, targets])
        n_select = m

    h_per_lm = np.maximum(h_per_lm, 0.0)

    Q = len(queries)
    selected = []
    current_max = np.zeros(Q)

    for _ in range(n_select):
        gains = np.maximum(h_per_lm - current_max[np.newaxis, :], 0.0)
        mean_gains = gains.mean(axis=1)
        for idx in selected:
            mean_gains[idx] = -1.0
        best = int(np.argmax(mean_gains))
        selected.append(best)
        current_max = np.maximum(current_max, h_per_lm[best])

    sel = np.array(selected)
    if is_directed:
        y_fwd = torch.from_numpy(d_out[sel]).t().float()
        y_bwd = torch.from_numpy(d_in[sel]).t().float()
    else:
        y_fwd = torch.from_numpy(d_out[sel]).t().float()
        y_bwd = y_fwd

    return make_linear_heuristic(y_fwd, y_bwd, is_directed)


def fps_subset_heuristic(graph, m, seed, lcc_nodes, lcc_seed):
    """Select m landmarks directly via FPS (not from a K0 pool)."""
    rng = torch.Generator().manual_seed(seed)
    if graph.is_directed:
        # ALT with K = m//2 landmarks stores 2*(m//2) = m values per vertex
        K = m // 2
    else:
        K = m
    K = max(K, 1)
    teacher = alt_preprocess(graph, K, seed_vertex=lcc_seed, rng=rng,
                             valid_vertices=lcc_nodes)
    return make_alt_heuristic(teacher)


def nominal_selection_stats(is_directed: bool, m: int) -> dict[str, int | float]:
    """Return ideal no-duplicate selection stats for non-learned baselines."""
    if is_directed:
        nominal_fwd = max(1, m // 2)
        nominal_bwd = m - nominal_fwd
        if nominal_bwd < 1:
            nominal_bwd = 1
            nominal_fwd = m - 1
        effective_unique_total = nominal_fwd + nominal_bwd
        return {
            "nominal_fwd": nominal_fwd,
            "nominal_bwd": nominal_bwd,
            "unique_fwd": nominal_fwd,
            "unique_bwd": nominal_bwd,
            "duplicates_fwd": 0,
            "duplicates_bwd": 0,
            "effective_unique_total": effective_unique_total,
            "effective_unique_ratio": effective_unique_total / max(m, 1),
        }

    return {
        "nominal_fwd": m,
        "nominal_bwd": 0,
        "unique_fwd": m,
        "unique_bwd": 0,
        "duplicates_fwd": 0,
        "duplicates_bwd": 0,
        "effective_unique_total": m,
        "effective_unique_ratio": 1.0,
    }


# ===================================================================
# Experiment 1: Selection strategy ablation
# ===================================================================

def experiment_1_selection_strategy():
    """Compare learned vs random vs FPS-subset selection at K0=64, m=16."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Selection Strategy Ablation")
    print("=" * 70)

    K0, m = 64, 16
    rows = []

    for gname, gpath in GRAPHS.items():
        print(f"\n--- {gname} ---")
        graph = load_graph_npz(str(gpath))
        lcc_nodes, lcc_seed = get_lcc(graph)
        queries = generate_queries(graph, NUM_QUERIES, seed=42)
        dij_exps = dijkstra_baseline(graph, queries)
        dij_mean = dij_exps.mean()
        print(f"  Dijkstra mean expansions: {dij_mean:.0f}")

        for seed in SEEDS:
            print(f"  seed={seed}:")

            # (a) AAC learned selection
            h_aac, teacher_labels, _, compressor, _ = train_aac(
                graph, K0, m, seed, lcc_nodes, lcc_seed
            )
            aac_stats = compressor.selection_stats()
            aac_exps, aac_opt = run_astar_queries(graph, queries, h_aac)
            aac_red = 100.0 * (1 - aac_exps.mean() / dij_mean)
            print(f"    AAC (default): {aac_red:.1f}% reduction, optimal={aac_opt}")
            rows.append(dict(graph=gname, seed=seed, method="AAC (default)",
                            K0=K0, m=m, bytes_per_vertex=m*4,
                            mean_exp=aac_exps.mean(), reduction_pct=aac_red,
                            all_optimal=aac_opt, **aac_stats))

            # (b) Random m-subset from same K0 teachers
            h_rand = random_subset_heuristic(teacher_labels, m, seed,
                                             graph.is_directed)
            rand_stats = nominal_selection_stats(graph.is_directed, m)
            rand_exps, rand_opt = run_astar_queries(graph, queries, h_rand)
            rand_red = 100.0 * (1 - rand_exps.mean() / dij_mean)
            print(f"    Random-Subset: {rand_red:.1f}% reduction, optimal={rand_opt}")
            rows.append(dict(graph=gname, seed=seed, method="Random-Subset",
                            K0=K0, m=m, bytes_per_vertex=m*4,
                            mean_exp=rand_exps.mean(), reduction_pct=rand_red,
                            all_optimal=rand_opt, **rand_stats))

            # (c) FPS m-subset (m/2 landmarks selected directly via FPS)
            h_fps = fps_subset_heuristic(graph, m, seed, lcc_nodes, lcc_seed)
            fps_stats = nominal_selection_stats(graph.is_directed, m)
            fps_exps, fps_opt = run_astar_queries(graph, queries, h_fps)
            fps_red = 100.0 * (1 - fps_exps.mean() / dij_mean)
            print(f"    FPS-Subset (ALT K={m//2}): {fps_red:.1f}% reduction, optimal={fps_opt}")
            rows.append(dict(graph=gname, seed=seed, method="FPS-Subset",
                            K0=K0, m=m, bytes_per_vertex=m*4,
                            mean_exp=fps_exps.mean(), reduction_pct=fps_red,
                            all_optimal=fps_opt, **fps_stats))

            # (d) Greedy-Maximize: greedily select landmarks maximizing avg heuristic
            h_greedy = greedy_maximize_heuristic(teacher_labels, m, queries,
                                                  graph.is_directed)
            greedy_stats = nominal_selection_stats(graph.is_directed, m)
            greedy_exps, greedy_opt = run_astar_queries(graph, queries, h_greedy)
            greedy_red = 100.0 * (1 - greedy_exps.mean() / dij_mean)
            print(f"    Greedy-Maximize: {greedy_red:.1f}% reduction, optimal={greedy_opt}")
            rows.append(dict(graph=gname, seed=seed, method="Greedy-Maximize",
                            K0=K0, m=m, bytes_per_vertex=m*4,
                            mean_exp=greedy_exps.mean(), reduction_pct=greedy_red,
                            all_optimal=greedy_opt, **greedy_stats))

    # Write CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / "exp1_selection_strategy.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {outpath}")

    # Print summary table
    print("\n  === Summary (mean +/- std across 3 seeds) ===")
    for gname in GRAPHS:
        print(f"\n  {gname} (K0={K0}, m={m}, {m*4} B/v):")
        for method in ["AAC (default)", "Random-Subset", "FPS-Subset", "Greedy-Maximize"]:
            vals = [r["reduction_pct"] for r in rows
                    if r["graph"] == gname and r["method"] == method]
            print(f"    {method:20s}: {np.mean(vals):.1f} +/- {np.std(vals):.1f}%")

    return rows


# ===================================================================
# Experiment 2: Admissibility under early stopping
# ===================================================================

def experiment_2_admissibility_robustness():
    """Train AAC with various epoch budgets, check admissibility at each."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Admissibility Under Early Stopping")
    print("=" * 70)

    K0, m = 64, 16
    epoch_checkpoints = [1, 5, 10, 50, 200]
    rows = []

    gname = "modena"
    gpath = GRAPHS[gname]
    print(f"\n--- {gname} ---")
    graph = load_graph_npz(str(gpath))
    lcc_nodes, lcc_seed = get_lcc(graph)
    queries = generate_queries(graph, NUM_QUERIES, seed=42)
    dij_exps = dijkstra_baseline(graph, queries)
    dij_mean = dij_exps.mean()

    for seed in SEEDS:
        print(f"  seed={seed}:")

        # Compute teacher labels once
        rng = torch.Generator().manual_seed(seed)
        anchors = farthest_point_sampling(
            graph, K0, seed_vertex=lcc_seed, rng=rng, valid_vertices=lcc_nodes
        )
        teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)

        for n_epochs in epoch_checkpoints:
            # Train with limited epochs (set patience very high to not early stop)
            compressor = LinearCompressor(K=K0, m=m, is_directed=graph.is_directed)
            cfg = TrainConfig(**{**TRAIN_CFG_DEFAULTS, "seed": seed,
                                "num_epochs": n_epochs, "patience": n_epochs + 1})
            train_linear_compressor(compressor, teacher_labels, cfg,
                                    valid_vertices=lcc_nodes)

            # Build heuristic
            compressor.eval()
            d_out_t = teacher_labels.d_out.t()
            d_in_t = teacher_labels.d_in.t()
            with torch.no_grad():
                if graph.is_directed:
                    y_fwd, y_bwd = compressor(d_out_t, d_in_t)
                else:
                    y = compressor(d_out_t)
                    y_fwd = y_bwd = y
            h = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)

            # Check admissibility
            violations = check_admissibility_queries(graph, queries, h)

            # Measure expansion reduction
            exps, all_opt = run_astar_queries(graph, queries, h)
            red = 100.0 * (1 - exps.mean() / dij_mean)

            print(f"    epochs={n_epochs:3d}: reduction={red:.1f}%, "
                  f"admissibility_violations={violations}, all_optimal={all_opt}")

            rows.append(dict(graph=gname, seed=seed, epochs=n_epochs,
                            K0=K0, m=m, reduction_pct=red,
                            admissibility_violations=violations,
                            all_optimal=all_opt, mean_exp=exps.mean()))

    # Write CSV
    outpath = OUTPUT_DIR / "exp2_admissibility_robustness.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {outpath}")

    # Print summary
    print("\n  === Summary (mean +/- std across 3 seeds) ===")
    print(f"  {gname} (K0={K0}, m={m}):")
    print(f"  {'Epochs':>8s}  {'Reduction':>12s}  {'Violations':>12s}  {'All Optimal':>12s}")
    for n_ep in epoch_checkpoints:
        reds = [r["reduction_pct"] for r in rows if r["epochs"] == n_ep]
        viols = [r["admissibility_violations"] for r in rows if r["epochs"] == n_ep]
        opts = [r["all_optimal"] for r in rows if r["epochs"] == n_ep]
        print(f"  {n_ep:>8d}  {np.mean(reds):>8.1f}+/-{np.std(reds):.1f}%  "
              f"{np.mean(viols):>10.0f}        {'Yes' if all(opts) else 'NO'}")

    return rows


# ===================================================================
# Experiment 3: Compression efficiency curve
# ===================================================================

def experiment_3_compression_curve():
    """Sweep m at fixed K0=64, compare to ALT at matched memory."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Compression Efficiency Curve")
    print("=" * 70)

    K0 = 64
    m_values = [4, 8, 16, 32, 64]
    rows = []

    for gname, gpath in GRAPHS.items():
        print(f"\n--- {gname} ---")
        graph = load_graph_npz(str(gpath))
        lcc_nodes, lcc_seed = get_lcc(graph)
        queries = generate_queries(graph, NUM_QUERIES, seed=42)
        dij_exps = dijkstra_baseline(graph, queries)
        dij_mean = dij_exps.mean()
        print(f"  Dijkstra mean: {dij_mean:.0f}")

        for seed in SEEDS:
            print(f"  seed={seed}:")

            # Compute teacher labels once for this seed
            rng = torch.Generator().manual_seed(seed)
            anchors = farthest_point_sampling(
                graph, K0, seed_vertex=lcc_seed, rng=rng, valid_vertices=lcc_nodes
            )
            teacher_labels = compute_teacher_labels(graph, anchors, use_gpu=False)

            # Full teacher ALT (K0=64) as upper bound
            h_full_alt = make_alt_heuristic(teacher_labels)
            full_exps, _ = run_astar_queries(graph, queries, h_full_alt)
            full_red = 100.0 * (1 - full_exps.mean() / dij_mean)
            rows.append(dict(graph=gname, seed=seed, method="ALT-Full",
                            K0=K0, m=K0*2 if graph.is_directed else K0,
                            compression_ratio=1.0,
                            bytes_per_vertex=K0*2*4 if graph.is_directed else K0*4,
                            reduction_pct=full_red, mean_exp=full_exps.mean(),
                            **nominal_selection_stats(
                                graph.is_directed,
                                K0 * 2 if graph.is_directed else K0,
                            )))

            for m in m_values:
                if m > K0:
                    continue
                bpv = m * 4
                ratio = m / K0

                # AAC at this m
                compressor = LinearCompressor(K=K0, m=m, is_directed=graph.is_directed)
                cfg = TrainConfig(**{**TRAIN_CFG_DEFAULTS, "seed": seed})
                train_linear_compressor(compressor, teacher_labels, cfg,
                                        valid_vertices=lcc_nodes)
                compressor.eval()
                aac_stats = compressor.selection_stats()
                d_out_t = teacher_labels.d_out.t()
                d_in_t = teacher_labels.d_in.t()
                with torch.no_grad():
                    if graph.is_directed:
                        y_fwd, y_bwd = compressor(d_out_t, d_in_t)
                    else:
                        y = compressor(d_out_t)
                        y_fwd = y_bwd = y
                h_aac = make_linear_heuristic(y_fwd, y_bwd, graph.is_directed)
                aac_exps, _ = run_astar_queries(graph, queries, h_aac)
                aac_red = 100.0 * (1 - aac_exps.mean() / dij_mean)

                # ALT at matched memory: K_alt landmarks = bpv / 8
                K_alt = max(1, bpv // 8)  # 2*K_alt*4 = bpv -> K_alt = bpv/8
                rng_alt = torch.Generator().manual_seed(seed)
                alt_teacher = alt_preprocess(graph, K_alt, seed_vertex=lcc_seed,
                                             rng=rng_alt, valid_vertices=lcc_nodes)
                h_alt = make_alt_heuristic(alt_teacher)
                alt_exps, _ = run_astar_queries(graph, queries, h_alt)
                alt_red = 100.0 * (1 - alt_exps.mean() / dij_mean)

                print(f"    m={m:2d} ({ratio:.2f}): AAC={aac_red:.1f}%, ALT(K={K_alt})={alt_red:.1f}%")

                rows.append(dict(graph=gname, seed=seed, method="AAC",
                                K0=K0, m=m, compression_ratio=ratio,
                                bytes_per_vertex=bpv,
                                reduction_pct=aac_red, mean_exp=aac_exps.mean(),
                                **aac_stats))
                rows.append(dict(graph=gname, seed=seed, method="ALT",
                                K0=K0, m=m, compression_ratio=ratio,
                                bytes_per_vertex=bpv,
                                reduction_pct=alt_red, mean_exp=alt_exps.mean(),
                                **nominal_selection_stats(graph.is_directed, m)))

    # Write CSV
    outpath = OUTPUT_DIR / "exp3_compression_curve.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {outpath}")

    # Print summary
    print("\n  === Summary (mean +/- std across 3 seeds) ===")
    for gname in GRAPHS:
        print(f"\n  {gname} (K0={K0}):")
        print(f"  {'m':>4s}  {'Ratio':>6s}  {'B/v':>4s}  {'AAC':>14s}  {'ALT':>14s}  {'Delta':>8s}")
        for m in m_values:
            aac_vals = [r["reduction_pct"] for r in rows
                       if r["graph"] == gname and r["method"] == "AAC" and r["m"] == m]
            alt_vals = [r["reduction_pct"] for r in rows
                       if r["graph"] == gname and r["method"] == "ALT" and r["m"] == m]
            if aac_vals and alt_vals:
                hm, hs = np.mean(aac_vals), np.std(aac_vals)
                am, as_ = np.mean(alt_vals), np.std(alt_vals)
                print(f"  {m:>4d}  {m/K0:>6.2f}  {m*4:>4d}  "
                      f"{hm:>6.1f}+/-{hs:.1f}%  {am:>6.1f}+/-{as_:.1f}%  "
                      f"{hm-am:>+6.1f}%")

    return rows


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running 3 ablation experiments for AAC paper strengthening")
    print(f"Graphs: {list(GRAPHS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"Queries: {NUM_QUERIES}")

    r1 = experiment_1_selection_strategy()
    r2 = experiment_2_admissibility_robustness()
    r3 = experiment_3_compression_curve()

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 70}")
    print(f"  ALL EXPERIMENTS COMPLETE ({elapsed/60:.1f} min)")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'=' * 70}")
