# Research: run-0006 — Focal Loss Failed, Loss/Calibration Axis Exhausted

## Verdict: Continue-Iterating (Override)

The report stage returned `submission-required`. This is **overridden to continue-iterating** because:

1. **Regression, not improvement**: val ROC-AUC dropped from 0.6777 (leader) to 0.6436, a **-0.034 regression** — the worst of any run.
2. **Round count is 6**, meeting the minimum, but the trajectory shows 3 consecutive failures (rounds 3, 5, 6).
3. **Family name** `perch_cached_probe` triggers the probe exclusion.
4. **No validated pipeline**: only 1 structural technique (coverage expansion) has succeeded across 6 rounds.
5. **This run's regression is severe**: -0.034 is 2.6x worse than the temperature scaling failure in round 5.

## Root Cause: Focal Loss Over-Suppresses Easy-Class Gradients on Small Training Set

Focal loss with gamma=2.0 down-weights easy-example contributions by a factor of (1 - p_t)^2. On a 708-window / 52-fitted-class training set where most examples are already "easy" (the linear probe on frozen Perch embeddings produces relatively low-confidence predictions), focal loss aggressively suppresses the majority of the gradient signal. The result: the probe learns even less than with standard BCE.

This is the third consecutive failure targeting loss or calibration:
- Round 3: coupled coverage + calibration → destabilized pipeline
- Round 5: temperature scaling → -0.012 regression
- Round 6: focal loss → -0.034 regression

The pattern is clear: **the bottleneck is not how gradients are weighted or how outputs are scaled — it is the discriminative capacity of the frozen Perch embeddings for the target domain**.

## Key Metrics Comparison

| Metric | Run-0002 (Leader) | Run-0005 (Temp Scale) | Run-0006 (Focal Loss) | Delta vs Leader |
|--------|-------------------|-----------------------|----------------------|-----------------|
| val_soundscape_macro_roc_auc | 0.6777 | 0.6656 | 0.6436 | -0.0341 |
| val_prior_fusion_macro_roc_auc | 0.662 | 0.662 | 0.662 | ~0 |
| soundscape_macro_roc_auc | 0.998 | 0.989 | 0.944 | -0.054 |
| prior_fusion_macro_roc_auc | 0.487 | 0.487 | 0.487 | ~0 |
| padded_cmap | 0.057 | 0.067 | 0.069 | +0.012 |

Prior fusion val remains rock-stable at 0.662 across all runs, confirming pipeline health. The regression is entirely in the learned probe's discriminability.

## Progress Tracker

| Round | Focus | Status |
|-------|-------|--------|
| 1 | Baseline | Done — 0.665 val, 52/75 classes |
| 2 | Coverage expansion | Done — 0.678 val, 67/75 classes (**leader**) |
| 3 | Coverage + calibration (coupled) | **Failed** — 0.676, 63/75, prior fusion broken |
| 4 | Revert + incremental coverage | **No gain** — 0.675, 71/75, pipeline restored |
| 5 | Temperature scaling calibration | **Failed** — 0.666, -0.012 regression |
| 6 | Focal loss | **Failed** — 0.644, -0.034 regression |
| 7 | Audit training temp + embedding augmentation | **Next move** |
| 8+ | Ensemble / SED lane / submission probe | Pending |

## Diagnostic: The Loss/Calibration Axis Is Exhausted

Three consecutive failures on loss and calibration confirm:

1. **The linear probe on frozen embeddings has hit a discriminability ceiling.** BCE already produces the best achievable gradient signal for this architecture.
2. **Training temperature 7.39 is likely a compounding factor** — inflating logits toward uniformity, making the probe's job harder regardless of loss function.
3. **Post-hoc scaling cannot recover per-class discriminability that was never learned.**

## Actionable Guidance for Round 7

**Priority 1: Audit and fix the training temperature.** The training temperature of 7.39 is abnormally high and likely inflates logits toward near-uniform distributions. Reducing it to 1.0-2.0 on the run-0002 code state is a single-axis change that addresses a suspected root cause upstream of both loss and calibration.

**Priority 2: If temperature audit is inconclusive, try embedding-level augmentation** (MixUp or Gaussian noise) to simulate soundscape variability during probe training. This targets the domain gap at the representation level rather than the loss level.

**Priority 3: If the probe lane remains stuck, consider launching the SED lane in parallel** with PCEN frontend, secondary-label masking, and waveform augmentation. The frozen-embedding probe may simply lack the representational capacity to close the domain gap, and a full audio pipeline may be needed.

Run-0002 remains the undisputed leader. All future experiments branch from run-0002 code state.
