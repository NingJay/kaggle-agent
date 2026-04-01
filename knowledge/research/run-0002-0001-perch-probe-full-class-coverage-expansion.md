# Research: run-0002-0001-perch-probe-full-class-coverage-expansion

## Verdict: Override to Continue-Iterating

The report stage returned `submission-required`. This is **overridden to continue-iterating** because the result fails the submission bar (04_submission_bar.md):

1. **Round count**: This is iteration round 2. Submission requires ≥5 rounds.
2. **Family name**: `perch_cached_probe` falls under the probe exclusion trigger.
3. **Applied techniques**: Coverage expansion is structural progress but calibration, ensemble, PP, and domain adaptation have not yet been applied.
4. **Iteration trajectory**: Two data points (0.665 → 0.678) is a trend, not a validated pipeline.

## Root Cause: Coverage Gap Narrowed but Not Closed

The coverage expansion from 52 to 67 fitted classes (out of 75 active) produced a +0.013 gain in val ROC-AUC (0.665 → 0.678). The remaining 8-class gap is still a structural blocker. The learned probe continues to outperform prior fusion on the same holdout split (0.678 vs 0.662), confirming the embedding head adds discriminative value for covered classes.

The train-side prior_fusion collapse (0.487) vs holdout strength (0.662) is an expected DG signature — the Bayesian prior generalizes to unseen classes precisely because it doesn't overfit the training distribution. The full soundscape AUC (0.998) is resubstitution and should not be acted on.

## Key Metrics

| Metric | Value | Trust | Use |
|--------|-------|-------|-----|
| val_soundscape_macro_roc_auc | 0.678 | Primary | Keep/discard signal |
| val_prior_fusion_macro_roc_auc | 0.662 | Comparison | Probe vs prior |
| soundscape_macro_roc_auc | 0.998 | Sanity-check | Resubstitution only |
| prior_fusion_macro_roc_auc | 0.487 | Calibration | Train-side reference |
| padded_cmap | 0.057 | Archival | Not actionable |

## Progress Tracker

| Round | Focus | Status |
|-------|-------|--------|
| 1 | Baseline | Done — 0.665 val, 52/75 classes |
| 2 | Coverage expansion | Done — 0.678 val, 67/75 classes |
| 3 | Close remaining 8-class gap + calibration | **Next move** |
| 4 | Domain robustness | Pending |
| 5 | Ensemble / multi-lane blend | Pending |
| 6+ | Post-processing and submission probe | Pending |

## Adopt Now

1. Override to continue-iterating. Round 2 complete.
2. Close the 8-class coverage gap. Determine if missing classes lack embedding data or are filtered by config.
3. Record the second anchor row: val 0.678, prior fusion holdout 0.662, LB placeholders.
4. Update experiment_conclusions.md with run-0002 results.
5. Maintain val_soundscape_macro_roc_auc as sole primary signal.

## Consider

- Round 3: close coverage to 75/75, then calibration and loss tuning.
- Round 4: domain robustness (DG gap is real: 0.998 train vs 0.678 val).
- Round 5: ensemble complementarity.
- Round 6+: post-processing and submission probe only after structural work.
- SED lane in parallel: PCEN, secondary-label masking, waveform augmentation.

## Reject

- Submitting now (fails all submission bar criteria).
- Threshold tuning or PP on incomplete coverage.
- Resubstitution chasing (0.998).
- padded_cMAP optimization.
- Cosmetic sweeps before structural fixes.
