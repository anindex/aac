# Pre-Registration Record: OGB-arXiv Evaluation

This file contains the verbatim pre-registered prediction for the OGB-arXiv
matched-memory evaluation, referenced from the paper (Appendix E, Sec.5.9.3).

## Prediction (filed before any OGB-arXiv evaluation)

> On OGB-arXiv (citation graph, ~170K nodes, non-metric, high clustering),
> AAC should beat ALT at every budget level with a gap of roughly +2 to +6
> percentage points at 32 B/v narrowing to +1 to +3% at 128 B/v; the hybrid
> max(h_AAC, h_ALT) should be the top performer; admissibility violations
> should be zero.

### Basis for prediction

- Covering-radius theory (Theorem 2 in the paper): FPS coverage is
  near-optimal on metric graphs, but OGB-arXiv has non-metric citation
  structure where FPS coverage is expected to be loose.
- Synthetic-graph findings (SBM/BA): ALT leads by 0.1-1.3%, but these are
  metric graphs. Non-metric structure should open more headroom.

## Observed outcome

| Budget (B/v) | Predicted gap (AAC−ALT) | Observed gap (AAC−ALT) | Verdict |
|-------------|------------------------|----------------------|---------|
| 32 | +2 to +6% | −1.13% (ALT ahead) | **Direction inverted** |
| 64 | +1.5 to +4.5% | +1.21 +/- 0.54% | Magnitude 4x low |
| 128 | +1 to +3% | +0.87 +/- 0.20% | Magnitude 2-3x low |

**Verdict:** Pre-registration **falsified**. The predicted bands are not
survived at any budget. The convention-independent sub-prediction (zero
admissibility violations) holds in 15/15 cells.

## Accounting convention

Matched memory at B bytes per vertex on OGB-arXiv (undirected after
symmetrization) uses AAC m=B/4 vs ALT K=B/4 (undirected rule). This is
the convention stated in Section 5.2 and Appendix E.2 of the paper.

## Result files

- `results/synthetic/ogbn_arxiv_results.csv` -- per-seed expansion counts
- `results/synthetic/ogbn_arxiv_admissibility.csv` -- per-cell admissibility audit
- `results/synthetic/ogbn_arxiv_log.txt` -- raw run log

## Reproduction

```bash
python scripts/run_nonroad_real.py        # Main OGB-arXiv experiment
python scripts/verify_ogbn_admissibility.py  # 15-cell admissibility audit
```
