# Research: run-0003 — Regression on Coverage and Prior Fusion

## Verdict: Continue-Iterating (Override)

The report stage returned `submission-required`. This is **overridden to continue-iterating** because:

1. **Regression, not improvement**: val ROC-AUC dropped from 0.6777 to 0.6758.
2. **Fitted class count regressed**: 67 → 63 out of 75 active classes.
3. **Prior fusion collapsed**: val_prior_fusion dropped from 0.662 to 0.475.
4. **Round count is 3**, still below the 5-round submission bar.
5. **Family name** `perch_cached_probe` triggers the probe exclusion.

This run does not represent forward progress. It represents a destabilizing code change that must be reverted.

## Root Cause: Code Changes Destabilized Probe Fitting

Round 3 attempted to close the remaining 8-class gap (67→75) and apply calibration simultaneously. Instead, the fitted class count dropped from 67 to 63 — **4 classes that were previously learned were lost**. The Bayesian prior path also collapsed from 0.662 to 0.475 on the same holdout split, suggesting the config changes broke more than just the probe head.

The regression pattern (lost fitted classes + broken prior path) points to a config or data-loading change that affected class filtering, sample weighting, or the training label matrix. The calibration additions may also have altered the loss function in a way that prevented some classes from being fitted.

## Key Metrics

| Metric | Run-0002 (Best) | Run-0003 (Current) | Delta |
|--------|-----------------|-------------------|-------|
| val_soundscape_macro_roc_auc | 0.6777 | 0.6758 | -0.0019 |
| val_prior_fusion_macro_roc_auc | 0.662 | 0.475 | -0.187 |
| soundscape_macro_roc_auc | 0.998 | 0.997 | -0.001 |
| prior_fusion_macro_roc_auc | 0.487 | 0.475 | -0.012 |
| padded_cmap | 0.057 | 0.057 | ~0 |
| fitted_class_count | 67/75 | 63/75 | -4 |

The prior fusion val collapse (-0.187) is the most alarming signal. It means the Bayesian prior — which requires no training — is producing worse holdout predictions. This is almost certainly a data pipeline or evaluation config bug, not a model training issue.

## Progress Tracker

| Round | Focus | Status |
|-------|-------|--------|
| 1 | Baseline | Done — 0.665 val, 52/75 classes |
| 2 | Coverage expansion | Done — 0.678 val, 67/75 classes (**current best**) |
| 3 | Close 8-class gap + calibration | **Regressed** — 0.676 val, 63/75 classes, prior fusion broken |
| 4 | Revert + incremental coverage fix | **Next move** |
| 5 | Isolated calibration | Pending |
| 6 | Domain robustness | Pending |
| 7 | Ensemble complementarity | Pending |
| 8+ | Post-processing and submission probe | Pending |

## Actionable Guidance

**Immediate**: Revert to the run-0002 code state as the base for round 4. Diagnose the round 3 regression by diffing configs and code before making any new changes. Close the 8-class gap with a single minimal change, then validate that fitted_class_count reaches 75 before touching anything else.
