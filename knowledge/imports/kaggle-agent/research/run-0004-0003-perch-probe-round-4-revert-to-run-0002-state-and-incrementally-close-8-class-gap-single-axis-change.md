# Research: run-0004 — Partial Revert Recovery, No Improvement Over Leader

## Verdict: Continue-Iterating (Override)

The report stage returned `submission-required`. This is **overridden to continue-iterating** because:

1. **Round count is 4**, still below the ≥5 submission bar.
2. **Family name** `perch_cached_probe` triggers the probe exclusion.
3. **No improvement over leader**: val ROC-AUC 0.6753 < run-0002's 0.6777.
4. **Only one structural technique** (coverage expansion) has been applied across 4 rounds — no calibration, ensemble, PP, or domain adaptation yet.
5. **Not a clean forward step**: the revert recovered prior fusion correctly but did not advance val AUC.

## Root Cause: Revert Restored Pipeline Health but Did Not Advance Performance

Round 4 successfully reverted the destabilizing round-3 changes: prior fusion val was restored from 0.475 to 0.662 (matching run-0002), and fitted class count improved from 67 to 71/75 (+4 classes recovered). However:

- **val_soundscape_macro_roc_auc decreased** from 0.6777 (run-0002) to 0.6753 (-0.0024). Adding 4 fitted classes did not translate to a holdout gain.
- **The remaining 4-class gap is likely data-driven**, not config-driven. Previous config changes recovered classes that round-3 lost, but the original 8-class gap from run-0002 may require new data or embedding vectors rather than parameter tuning.
- **The pipeline is confirmed healthy**: prior fusion restored, training completes, metrics are in expected ranges. The infrastructure problem from round-3 is resolved.

## Key Metrics Comparison

| Metric | Run-0002 (Leader) | Run-0003 (Broken) | Run-0004 (Current) | Delta vs Leader |
|--------|-------------------|-------------------|---------------------|-----------------|
| val_soundscape_macro_roc_auc | 0.6777 | 0.6758 | 0.6753 | -0.0024 |
| val_prior_fusion_macro_roc_auc | 0.662 | 0.475 | 0.662 | ~0 |
| soundscape_macro_roc_auc | 0.998 | 0.997 | 0.999 | +0.001 |
| prior_fusion_macro_roc_auc | 0.487 | 0.475 | 0.487 | ~0 |
| padded_cmap | 0.057 | 0.057 | 0.054 | -0.003 |
| fitted_class_count | 67/75 | 63/75 | 71/75 | +4 |

The prior fusion restoration confirms the round-3 regression was entirely config-driven. The new fitted class count (71) is higher than run-0002 (67), but val AUC is lower — suggesting the 4 newly fitted classes may be rare or poorly represented in the holdout split.

## Progress Tracker

| Round | Focus | Status |
|-------|-------|--------|
| 1 | Baseline | Done — 0.665 val, 52/75 classes |
| 2 | Coverage expansion | Done — 0.678 val, 67/75 classes (**current best**) |
| 3 | Coverage + calibration (coupled) | **Regressed** — 0.676 val, 63/75, prior fusion broken |
| 4 | Revert + incremental coverage | **Done** — 0.675 val, 71/75, pipeline restored but no AUC gain |
| 5 | Isolated calibration on run-0002 base | **Next move** |
| 6 | Domain robustness | Pending |
| 7 | Ensemble complementarity | Pending |
| 8+ | Post-processing and submission probe | Pending |

## Actionable Guidance

**Immediate (Round 5)**: Run isolated calibration on the **run-0002 code state** (the performance leader), not run-0004. Single-axis change only: temperature scaling, Platt scaling, or per-class threshold optimization. Do not touch class coverage config — run-0002's 67/75 is the stable baseline.

**Data diagnosis**: Before attempting further coverage expansion, audit which 4 classes remain unfitted and whether their embedding vectors exist in the Perch cache. If data is missing, document the gap and decide whether to (a) generate augmented samples, (b) accept 71/75 coverage, or (c) filter the evaluation class set.

**Do not couple changes**: The round-3 lesson is confirmed — multi-axis changes in one round are dangerous. Every future round must be a single structural change with a clear hypothesis.
