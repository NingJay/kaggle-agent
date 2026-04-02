# Research: run-0009 — Embedding-Level MixUp Regression

## Verdict: Continue-Iterating (Override)

The report stage returned `submission-required`. This is **overridden to continue-iterating** because:

1. **Regression, not improvement**: val ROC-AUC dropped from 0.6801 (run-0008 leader) to 0.6650, a **-0.015 regression**.
2. **Family name** `perch_cached_probe` triggers the probe exclusion per 04_submission_bar.md.
3. **Only 2 validated structural techniques** across 9 rounds: coverage expansion (run-0002) and MLP architecture (run-0008). No ensemble, post-processing, or domain adaptation has been applied.
4. **This is the 5th regression** in 9 rounds (rounds 3, 5, 6, 7, 9). The pipeline needs forward steps, not more failures.

## Root Cause: MixUp Corrupts Class Structure in Frozen Perch Embedding Space

Run-0009 applied embedding-level MixUp (interpolating Perch embedding vectors and their labels during training) on the run-0008 MLP head code state. The result: **val_soundscape_macro_roc_auc = 0.6650**, a -0.015 regression from the leader (0.6801).

The mechanism of failure: the frozen Perch embeddings are pre-trained class-discriminative vectors. Interpolating between them creates synthetic points that do not lie on meaningful class manifolds — the blended embeddings represent phantom classes that never appear at inference time. The MLP head learns to fit these synthetic points, which corrupts decision boundaries for the real classes.

Training resubstitution AUC remained very high (0.9918 vs 0.990 in run-0008), confirming the probe overfits to the blended representations. Prior fusion holdout stayed flat at 0.6622, isolating the regression entirely to the probe component.

## Full Trajectory

| Round | Change | Val AUC | Delta vs Prev Best | Outcome |
|-------|--------|---------|-------------------|---------|
| 1 | Baseline | 0.665 | — | Coverage gap (52/75) |
| 2 | Coverage expansion | 0.678 | +0.013 | **Leader** |
| 3 | Coupled coverage + calibration | 0.676 | -0.002 | Pipeline destabilized |
| 4 | Revert + incremental coverage | 0.675 | -0.003 | No gain |
| 5 | Post-hoc temperature scaling | 0.666 | -0.012 | Regression |
| 6 | Focal loss | 0.644 | -0.034 | Catastrophic regression |
| 7 | Training temp reduction (7.39→1.0) | 0.665 | -0.013 | Regression |
| 8 | **MLP head (1 hidden layer)** | **0.680** | **+0.002** | **New leader** |
| 9 | Embedding-level MixUp | 0.665 | -0.015 | Regression |

## Key Metrics

| Metric | Run-0008 (Leader) | Run-0009 (Current) | Delta |
|--------|-------------------|---------------------|-------|
| val_soundscape_macro_roc_auc | 0.6801 | 0.6650 | -0.0151 |
| val_prior_fusion_macro_roc_auc | 0.662 | 0.6622 | ~0 |
| soundscape_macro_roc_auc | 0.990 | 0.9918 | +0.002 |
| prior_fusion_macro_roc_auc | 0.487 | 0.4873 | ~0 |
| padded_cmap | 0.069 | 0.062 | -0.007 |

Prior fusion holdout is rock-stable. The regression is entirely in the learned probe.

## Axis Exhaustion Summary

Five rounds have regressed from the run-0002 baseline. Three distinct axes are now exhausted:

1. **Loss/calibration/temperature** (rounds 3, 5, 6, 7): Falsified. The linear probe capacity ceiling was the real bottleneck, resolved by the MLP head in round 8.
2. **Embedding-level augmentation** (round 9): Falsified. MixUp corrupts class structure in the frozen Perch space. The embeddings are not amenable to interpolation-based augmentation.

The remaining productive axes for the perch_cached_probe lane are: pipeline-level optimization (blend ratio), probe regularization (dropout/weight decay), and ensembling. Probe training-level changes (loss, augmentation, temperature) have a 1-in-5 success rate.

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
| 8 | MLP head architecture | **Done — 0.680, current leader** |
| 9 | Embedding-level MixUp | **Failed — 0.665, -0.015 regression** |
| 10 | Blend optimization or MLP regularization | **Next move** |

## Actionable Guidance

**Immediate**: Run-0008 MLP code state remains the base. Do not build on run-0009.

**Round 10 options** (ranked by expected value):

1. **Prior-fusion blend ratio search**: Grid search the blend weight between MLP probe predictions and Bayesian prior predictions on the holdout split. This extracts pipeline-level value without touching probe training. Single-axis, low risk.

2. **MLP dropout regularization**: Add dropout (0.3-0.5) to the hidden layer of the MLP head. The MLP has more parameters than the linear probe and trains on ~700 windows. Explicit regularization may improve generalization without corrupting representation structure like MixUp did.

3. **Label smoothing**: Replace hard labels with smoothed labels (ε=0.1) during training. Gentler than MixUp because it operates on the label side rather than the embedding side.

4. **SED lane launch**: If round 10 also fails, the perch_cached_probe lane has diminishing returns. Launching PCEN + secondary-label masking + waveform augmentation in parallel is the higher-value path.
