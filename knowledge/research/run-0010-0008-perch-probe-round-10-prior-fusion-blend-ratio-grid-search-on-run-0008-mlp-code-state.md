# Research: run-0010 — Prior-Fusion Blend Ratio Grid Search Regression

## Verdict: Continue-Iterating (Override)

The report stage returned `submission-required`. This is **overridden to continue-iterating** because:

1. **Regression, not improvement**: val ROC-AUC = 0.6653 vs run-0008 leader 0.6801, a **-0.0148 regression**.
2. **Family name** `perch_cached_probe` triggers the probe exclusion per 04_submission_bar.md.
3. **Only 2 validated structural techniques** across 10 rounds: coverage expansion (run-0002) and MLP architecture (run-0008). No ensemble, post-processing, or domain adaptation has been applied.
4. **This is the 6th regression** in 10 rounds (rounds 3, 5, 6, 7, 9, 10). The pipeline needs forward steps, not more failures.
5. **Prior fusion holdout stable at 0.662**, confirming this is a technique failure, not infrastructure.

## Root Cause: Prior Does Not Carry Complementary Holdout Signal

Run-0010 grid-searched the blend weight between the MLP probe (val AUC 0.680) and the Bayesian prior (val AUC 0.662) on the run-0008 MLP code state. The best blended result was val_soundscape_macro_roc_auc = 0.6653, worse than the standalone MLP probe by 0.0148.

The mechanism: the Bayesian prior and MLP probe are not complementary on the holdout split. The prior produces weaker predictions (0.662) that are correlated with the probe's predictions (0.680) rather than correcting different error modes. Blending dilutes the MLP's discriminative signal with the prior's less accurate estimates. No blend weight outperformed the pure MLP probe.

Prior fusion holdout remained rock-stable at 0.662, and the train-side prior_fusion_macro_roc_auc held at 0.487, confirming pipeline health. The regression is entirely in the probe-prior fusion logic.

## Full Trajectory

| Round | Change | Val AUC | Delta vs Prev Best | Outcome |
|-------|--------|---------|-------------------|---------|
| 1 | Baseline | 0.665 | — | Coverage gap (52/75) |
| 2 | Coverage expansion | 0.678 | +0.013 | Leader |
| 3 | Coupled coverage + calibration | 0.676 | -0.002 | Pipeline destabilized |
| 4 | Revert + incremental coverage | 0.675 | -0.003 | No gain |
| 5 | Post-hoc temperature scaling | 0.666 | -0.012 | Regression |
| 6 | Focal loss | 0.644 | -0.034 | Catastrophic regression |
| 7 | Training temp reduction (7.39→1.0) | 0.665 | -0.013 | Regression |
| 8 | MLP head (1 hidden layer) | 0.680 | +0.002 | **Current leader** |
| 9 | Embedding-level MixUp | 0.665 | -0.015 | Regression |
| 10 | Prior-fusion blend ratio grid search | 0.665 | -0.015 | Regression |

## Key Metrics

| Metric | Run-0008 (Leader) | Run-0010 (Current) | Delta |
|--------|-------------------|---------------------|-------|
| val_soundscape_macro_roc_auc | 0.6801 | 0.6653 | -0.0148 |
| val_prior_fusion_macro_roc_auc | 0.662 | 0.662 | ~0 |
| soundscape_macro_roc_auc | 0.990 | 0.992 | +0.002 |
| prior_fusion_macro_roc_auc | 0.487 | 0.487 | ~0 |
| padded_cmap | 0.069 | 0.063 | -0.006 |

## Axis Exhaustion Summary

Three axes are now definitively exhausted across 10 rounds:

1. **Loss/calibration/temperature** (rounds 3, 5, 6, 7): Four distinct interventions all failed on the linear probe. Resolved by MLP architecture change in round 8.
2. **Embedding-level augmentation** (round 9): MixUp corrupts class structure in frozen Perch embedding space.
3. **Prior-fusion blend ratio** (round 10): The prior does not carry complementary holdout signal; blending always dilutes the MLP probe.

The only successful interventions were **coverage expansion** (run-0002, +0.013) and **MLP architecture** (run-0008, +0.002). The remaining productive axes are probe regularization, label-side techniques, and potentially a parallel SED lane.

## Progress Tracker

| Round | Focus | Status |
|-------|-------|--------|
| 1 | Baseline | Done — 0.665 val, 52/75 classes |
| 2 | Coverage expansion | Done — 0.678 val, 67/75 classes |
| 3 | Coupled coverage + calibration | Failed — pipeline destabilized |
| 4 | Revert + incremental coverage | No gain — 0.675, 71/75 |
| 5 | Temperature scaling calibration | Failed — 0.666 |
| 6 | Focal loss | Failed — 0.644 |
| 7 | Training temperature reduction | Failed — 0.665 |
| 8 | MLP head architecture | Done — 0.680, current leader |
| 9 | Embedding-level MixUp | Failed — 0.665 |
| 10 | Prior-fusion blend ratio search | Failed — 0.665 |
| 11 | MLP regularization or label smoothing | Next move |

## Adopt Now

1. Override to continue-iterating. Run-0010 regressed.
2. Declare the blend-ratio axis exhausted.
3. Run-0008 MLP code state remains the undisputed base.
4. Update experiment_conclusions.md with run-0010 results.

## Consider for Round 11

1. **MLP dropout regularization** (0.3-0.5): the MLP trains on ~700 windows; explicit regularization may improve generalization.
2. **Label smoothing** (ε=0.1): gentler than MixUp, operates on labels not embeddings.
3. **SED lane launch in parallel**: after 10 rounds and 3 exhausted axes, PCEN + secondary-label masking + waveform augmentation target the domain gap from a fundamentally different angle.
4. **Ensemble of run-0002 linear and run-0008 MLP probes**: if they make different errors, averaging may help without retraining.

## Reject

- Prior-fusion blend experiments (exhausted in run-0010).
- Embedding-level augmentation (exhausted in run-0009).
- Loss function changes (exhausted in run-0006).
- Temperature experiments (exhausted in rounds 5, 7).
- Post-hoc calibration (exhausted in run-0005).
- Any submission of perch_cached_probe results (fails submission bar).
- Resubstitution chasing, padded_cMAP optimization.
