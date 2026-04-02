# Research: run-0005 — Temperature Scaling Calibration Failed

## Verdict: Continue-Iterating (Override)

The report stage returned `submission-required`. This is **overridden to continue-iterating** because:

1. **Round count is 5**, meeting the minimum but still insufficient on other criteria.
2. **Family name** `perch_cached_probe` triggers the probe exclusion.
3. **Only 1 validated structural technique** (coverage expansion) across 5 rounds — temperature calibration failed, coupled changes in round 3 failed.
4. **No ensemble, PP, or domain adaptation** has been applied.
5. **This run regressed**: val ROC-AUC 0.6656 < run-0002's 0.6777 (-0.012).

## Root Cause: Global Temperature Scaling Is the Wrong Calibration Lever

The single global temperature parameter optimized to 0.88 on validation while training operates at temperature 7.39. This 8.4x mismatch between training and validation temperature scales means the probe's logits are already poorly scaled at emission time, and a single multiplicative factor cannot correct per-class miscalibration across 52 fitted classes. The pipeline is healthy (val_prior_fusion 0.662 unchanged from leader), confirming this is a technique failure, not infrastructure.

## Key Metrics Comparison

| Metric | Run-0002 (Leader) | Run-0004 (Revert) | Run-0005 (Current) | Delta vs Leader |
|--------|-------------------|-------------------|---------------------|-----------------|
| val_soundscape_macro_roc_auc | 0.6777 | 0.6753 | 0.6656 | -0.0121 |
| val_prior_fusion_macro_roc_auc | 0.662 | 0.662 | 0.662 | ~0 |
| soundscape_macro_roc_auc | 0.998 | 0.999 | 0.989 | -0.009 |
| prior_fusion_macro_roc_auc | 0.487 | 0.487 | 0.487 | ~0 |
| padded_cmap | 0.057 | 0.054 | 0.067 | +0.010 |

The prior fusion val being rock-stable at 0.662 across all three runs confirms the pipeline is healthy. The regression is isolated to the learned probe's calibration behavior.

## Progress Tracker

| Round | Focus | Status |
|-------|-------|--------|
| 1 | Baseline | Done — 0.665 val, 52/75 classes |
| 2 | Coverage expansion | Done — 0.678 val, 67/75 classes (**leader**) |
| 3 | Coverage + calibration (coupled) | **Failed** — 0.676 val, 63/75, prior fusion broken |
| 4 | Revert + incremental coverage | **No gain** — 0.675 val, 71/75, pipeline restored |
| 5 | Temperature scaling calibration | **Failed** — 0.666 val, -0.012 regression |
| 6 | Domain robustness or loss semantics | **Next move** |
| 7 | Ensemble complementarity | Pending |
| 8+ | Post-processing and submission probe | Pending |

## Diagnostic: Why Calibration Keeps Failing

Two consecutive calibration-adjacent experiments have failed (round 3 coupled, round 5 isolated). The pattern suggests:

- The probe head is too simple (linear on frozen Perch embeddings) for post-hoc calibration to help. The bottleneck may be embedding quality or representation alignment, not output scaling.
- Training temperature 7.39 may be inflating logits, making the probability distribution already near-uniform for many classes. A downstream temperature of 0.88 cannot recover per-class discriminability that was never learned.
- Per-class calibration (Platt, isotonic) might help but adds complexity and risks overfitting on the small validation set.

**Implication**: The higher-value next move is likely not calibration at all, but domain robustness or loss semantics changes that improve the probe's training-time discriminability.

## Actionable Guidance

**Round 6**: Pivot to domain robustness on the run-0002 code state. Concrete options (single-axis only):

1. **Loss semantics**: Replace BCE with asymmetric loss or focal loss to improve rare-class handling within the 67/75 fitted set.
2. **Domain-aware weighting**: Add frequency-dependent sample weighting or class-balanced loss to reduce the focal-vs-soundscape distribution gap.
3. **Embedding augmentation**: MixUp or noise injection at the embedding level to simulate soundscape variability during probe training.

Do not attempt calibration again until the probe produces more discriminative logits in the first place. The training temperature (7.39) should also be audited — if it is unintentionally high, reducing it may be a prerequisite for any calibration approach to work.

Run-0002 remains the undisputed leader. All future experiments branch from run-0002 code state.
