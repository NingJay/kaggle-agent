# Research: run-0007 — Training Temperature Reduction Failed, Axis Exhausted

## Verdict: Continue-Iterating (Override)

The report stage returned `submission-required`. This is **overridden to continue-iterating** because:

1. **Regression, not improvement**: val ROC-AUC dropped from 0.6777 (leader) to 0.6652, a -0.0125 regression.
2. **This is the 4th consecutive failure** (rounds 3, 5, 6, 7).
3. **Family name** `perch_cached_probe` triggers the probe exclusion.
4. **No validated pipeline**: only 1 structural technique (coverage expansion) has succeeded across 7 rounds.
5. **Temperature was not the bottleneck**: the hypothesis that 7.39 was abnormally high and inflating logits toward uniformity is falsified — reducing to 1.0 made things worse, not better.

## Root Cause: Temperature Was Not the Bottleneck — Probe Capacity Ceiling Confirmed

Round 7 reduced the training temperature from 7.39 to 1.0 on the run-0002 leader code state. The result: val_soundscape_macro_roc_auc dropped to 0.6652, worse than the leader (0.6777). The prior fusion holdout remained perfectly stable at 0.662, confirming pipeline health. The regression is entirely in the learned probe's discriminability.

This falsifies the round-6 hypothesis that "training temperature 7.39 is likely inflating logits toward near-uniform distributions." The temperature was not the problem. The frozen Perch embedding linear probe has hit a fundamental capacity ceiling.

## Four-Axis Exhaustion Summary

| Round | Axis Changed | Val AUC | Delta vs Leader | Outcome |
|-------|-------------|---------|-----------------|---------|
| 3 | Coupled coverage + calibration | 0.6758 | -0.0019 | Pipeline destabilized |
| 5 | Post-hoc temperature scaling | 0.6656 | -0.0121 | Regression |
| 6 | Focal loss (loss semantics) | 0.6436 | -0.0341 | Catastrophic regression |
| 7 | Training temperature reduction | 0.6652 | -0.0125 | Regression |

Four different interventions targeting logit scaling, loss weighting, and temperature all failed. The common factor: **the linear probe on frozen Perch embeddings lacks the representational capacity to improve further through output-level or gradient-level tweaks**.

## Key Metrics Comparison

| Metric | Run-0002 (Leader) | Run-0006 (Focal) | Run-0007 (Temp=1.0) | Delta vs Leader |
|--------|-------------------|-------------------|---------------------|-----------------|
| val_soundscape_macro_roc_auc | 0.6777 | 0.6436 | 0.6652 | -0.0125 |
| val_prior_fusion_macro_roc_auc | 0.662 | 0.662 | 0.662 | ~0 |
| soundscape_macro_roc_auc | 0.998 | 0.944 | 0.990 | -0.008 |
| prior_fusion_macro_roc_auc | 0.487 | 0.487 | 0.487 | ~0 |
| padded_cmap | 0.057 | 0.069 | 0.071 | +0.014 |

Prior fusion holdout is rock-stable at 0.662 across all 7 runs. The regression is always isolated to the learned probe. The probe adds only +0.016 over the prior on the same holdout split (0.678 vs 0.662) — a thin margin that evaporates with any perturbation.

## Progress Tracker

| Round | Focus | Status |
|-------|-------|--------|
| 1 | Baseline | Done — 0.665 val, 52/75 classes |
| 2 | Coverage expansion | Done — 0.678 val, 67/75 classes (**leader**) |
| 3 | Coverage + calibration (coupled) | **Failed** — 0.676, 63/75, prior fusion broken |
| 4 | Revert + incremental coverage | **No gain** — 0.675, 71/75, pipeline restored |
| 5 | Post-hoc temperature scaling | **Failed** — 0.666, -0.012 regression |
| 6 | Focal loss | **Failed** — 0.644, -0.034 regression |
| 7 | Training temperature reduction (7.39→1.0) | **Failed** — 0.665, -0.013 regression |
| 8 | Architecture or representation change | **Next move** |

## Strategic Diagnosis

The linear probe on frozen Perch embeddings has exhausted its improvement potential via output-level and gradient-level interventions. The remaining options are:

1. **Representation-level changes** (augmentation, non-linear head) — may squeeze more signal from the same embeddings.
2. **Pipeline-level changes** (prior-fusion blend optimization, ensemble) — may extract value from the existing 0.678/0.662 probe-prior pair.
3. **Lane change to SED** — a full audio pipeline bypasses the frozen-embedding limitation entirely.

## Adopt Now

1. **Override to continue-iterating.** Round 7 regressed.
2. **Declare the loss/calibration/temperature axis fully exhausted.** No further experiments on logit scaling, loss weighting, or temperature.
3. **Update experiment_conclusions.md** with run-0007 results and the axis exhaustion diagnosis.
4. **Run-0002 remains the undisputed leader at 0.6777.** All future experiments branch from run-0002 code state.

## Consider for Round 8

1. **Non-linear probe head**: Replace the linear layer with a 1-hidden-layer MLP (e.g., embedding_dim → 256 → num_classes with ReLU). Single-axis change on run-0002 code state. Targets representational capacity directly.
2. **Embedding-level augmentation**: MixUp or Gaussian noise at the embedding level during training to simulate soundscape variability. Targets the domain gap at the representation level.
3. **Prior-fusion blend ratio optimization**: The probe (0.678) and prior (0.662) are close on holdout. Explicitly tuning the blend ratio may find a sweet spot that outperforms either component alone.
4. **Launch SED lane in parallel**: PCEN frontend, secondary-label masking, waveform augmentation. The frozen-embedding probe may simply lack capacity for the domain gap.

## Reject

- Any further temperature experiments (rounds 5 and 7 exhausted this axis).
- Any further loss function changes (round 6 catastrophic regression).
- Any further post-hoc calibration (round 5 failed).
- Submitting any perch_cached_probe result (fails submission bar on all criteria).
- Resubstitution chasing (0.990 is sanity-check only).
- Cosmetic sweeps before a structural breakthrough.
- padded_cMAP optimization.
