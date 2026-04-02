# Research: run-0001-perch-baseline

## Verdict: Override to Continue-Iterating

The report stage returned `submission-required`. This is **overridden to continue-iterating** because the baseline fails the submission bar (04_submission_bar.md) on all four criteria:

1. **First run** — no iteration history exists.
2. **Simple probe** — cached embeddings + linear head, no ensemble/calibration/PP/DA.
3. **No iteration history** — this is round 1.
4. **Family name contains "baseline"** — explicit exclusion trigger.

## Root Cause: Class Coverage Deficit

**Only 52 of 75 active classes are fitted**, leaving 23 classes (31%) with zero learned representation. This is the single structural priority. No amount of threshold tuning, calibration, or post-processing can fix classes the model was never trained on.

The train-val gap (0.992 vs 0.665) is an expected domain generalization signature, not overfitting. The prior fusion train-side collapse (0.487) vs val-side strength (0.662) confirms the Bayesian prior generalizes better to unseen classes than the fitted probe — precisely because the probe only covers 52/75 classes.

## Key Metrics

| Metric | Value | Trust | Use |
|--------|-------|-------|-----|
| val_soundscape_macro_roc_auc | 0.665 | Primary | Keep/discard signal |
| val_prior_fusion_macro_roc_auc | 0.662 | Comparison | Probe vs prior |
| soundscape_macro_roc_auc | 0.992 | Sanity-check | Resubstitution only |
| prior_fusion_macro_roc_auc | 0.487 | Calibration | Train-side reference |
| padded_cmap | 0.062 | Archival | Not actionable |

The learned probe edges out prior fusion on the same holdout split (0.665 vs 0.662), confirming the pipeline architecture works and the embedding head adds discriminative signal for the classes it does cover.

## Adopt Now

1. **Override verdict to continue-iterating.** Round 1 complete. Submission requires minimum 5 rounds.
2. **Expand class coverage to 75 active classes.** Investigate whether the 23-class gap is config-driven (default.yaml filtering) or data-driven (missing training samples in the embedding cache).
3. **Record the first anchor row** with local val (0.665), prior fusion holdout (0.662), and LB placeholders.
4. **Maintain val_soundscape_macro_roc_auc as sole primary signal.**
5. **Update experiment_conclusions.md** with run-0001 results and coverage gap as the active blocking issue.

## Iteration Roadmap

| Round | Focus | Status |
|-------|-------|--------|
| 1 | Baseline | Done — 0.665 val, coverage gap identified |
| 2 | Expand class coverage to 75 | **Next move** |
| 3 | Calibration and loss tuning | Pending |
| 4 | Domain robustness | Pending |
| 5 | Ensemble / multi-lane blend | Pending |
| 6+ | Post-processing and submission probe | Pending |

Submission requires minimum 5 rounds with applied techniques. Current state: round 1 complete. **Do not submit.**

## Rejected Moves

- Submitting this baseline (fails all 4 submission bar criteria)
- Threshold tuning or post-processing on a model with 31% class coverage
- Resubstitution metric chasing (0.992 is sanity-check only)
- Cosmetic hyperparameter sweeps before fixing the structural coverage gap
- padded_cMAP optimization
