# Research: run-0011 — MLP Dropout Regularization Regression

## Verdict: Continue-Iterating (Override)

The report stage returned `submission-required`. This is **overridden to continue-iterating** because:

1. **Regression, not improvement**: val ROC-AUC = 0.6754 vs run-0008 leader 0.6801, a **-0.0047 regression**.
2. **Family name** `perch_cached_probe` triggers the probe exclusion per 04_submission_bar.md.
3. **Only 2 validated structural techniques** across 11 rounds: coverage expansion (run-0002) and MLP architecture (run-0008). No ensemble, post-processing, or domain adaptation has been applied.
4. **This is the 7th regression** in 11 rounds (rounds 3, 5, 6, 7, 9, 10, 11). The pipeline needs forward steps, not more failures.
5. **Prior fusion holdout stable at 0.662**, confirming this is a technique failure, not infrastructure.

## Root Cause: Dropout Suppresses Already-Scarce Gradient Signal

Run-0011 applied dropout (0.3-0.5) to the MLP hidden layer on the run-0008 leader code state. The hypothesis was that the MLP overfits on ~700 training windows. The result contradicts this: val AUC dropped to 0.6754 (-0.0047 from leader).

The mechanism: with only ~700 training windows, the data itself is already a strong regularizer. Dropout randomly zeros hidden units during training, further reducing the effective gradient signal per batch. On a small dataset where the model already struggles to learn discriminative boundaries in the frozen Perch embedding space, dropout makes the learning problem harder without any overfitting benefit to offset.

Prior fusion holdout remained rock-stable at 0.662, confirming pipeline health. The regression is entirely in the probe component.

## Full Trajectory

| Round | Change | Val AUC | Delta vs Leader | Outcome |
|-------|--------|---------|-----------------|---------|
| 1 | Baseline | 0.665 | — | Coverage gap (52/75) |
| 2 | Coverage expansion | 0.678 | +0.013 | Leader |
| 3 | Coupled coverage + calibration | 0.676 | -0.002 | Pipeline destabilized |
| 4 | Revert + incremental coverage | 0.675 | -0.003 | No gain |
| 5 | Post-hoc temperature scaling | 0.666 | -0.012 | Regression |
| 6 | Focal loss | 0.644 | -0.034 | Catastrophic regression |
| 7 | Training temp reduction | 0.665 | -0.013 | Regression |
| 8 | **MLP head** | **0.680** | **+0.002** | **Current leader** |
| 9 | Embedding-level MixUp | 0.665 | -0.015 | Regression |
| 10 | Prior-fusion blend search | 0.665 | -0.015 | Regression |
| 11 | MLP dropout regularization | 0.675 | -0.005 | Regression |

## Key Metrics

| Metric | Run-0008 (Leader) | Run-0011 (Current) | Delta |
|--------|-------------------|---------------------|-------|
| val_soundscape_macro_roc_auc | 0.6801 | 0.6754 | -0.0047 |
| val_prior_fusion_macro_roc_auc | 0.662 | 0.662 | ~0 |
| soundscape_macro_roc_auc | 0.990 | 0.992 | +0.002 |
| prior_fusion_macro_roc_auc | 0.487 | 0.487 | ~0 |
| padded_cmap | 0.069 | 0.070 | +0.001 |

## Axis Exhaustion Summary

Four axes are now definitively exhausted across 11 rounds:

1. **Loss/calibration/temperature** (rounds 3, 5, 6, 7): Four distinct interventions all failed. Resolved by MLP architecture change in round 8.
2. **Embedding-level augmentation** (round 9): MixUp corrupts class structure in frozen Perch space.
3. **Prior-fusion blend ratio** (round 10): The prior does not carry complementary holdout signal; blending dilutes the MLP probe.
4. **Dropout regularization** (round 11): The small training set already provides implicit regularization; explicit dropout only hurts.

The only successful interventions remain **coverage expansion** (run-0002, +0.013) and **MLP architecture** (run-0008, +0.002).

## Progress Tracker

| Round | Focus | Status |
|-------|-------|--------|
| 1 | Baseline | Done — 0.665, 52/75 classes |
| 2 | Coverage expansion | Done — 0.678, 67/75 classes |
| 3 | Coupled coverage + calibration | Failed — pipeline destabilized |
| 4 | Revert + incremental coverage | No gain — 0.675, 71/75 |
| 5 | Temperature scaling | Failed — 0.666 |
| 6 | Focal loss | Failed — 0.644 |
| 7 | Training temp reduction | Failed — 0.665 |
| 8 | MLP head architecture | Done — 0.680, current leader |
| 9 | Embedding-level MixUp | Failed — 0.665 |
| 10 | Prior-fusion blend search | Failed — 0.665 |
| 11 | MLP dropout regularization | Failed — 0.675 |
| 12 | Label smoothing or SED lane | **Next move** |

## Strategic Assessment

After 11 rounds in the perch_cached_probe lane with 4 exhausted axes and a 2-in-11 success rate, the lane is approaching diminishing returns. The MLP probe at 0.6801 represents a meaningful improvement over the linear probe ceiling (~0.678), but further improvements from this lane require either:

- A genuinely novel intervention that does not fall into any exhausted axis (label smoothing is the last untried option that fits).
- A fundamentally different lane (SED with audio-level processing) that bypasses the frozen-embedding limitation entirely.
