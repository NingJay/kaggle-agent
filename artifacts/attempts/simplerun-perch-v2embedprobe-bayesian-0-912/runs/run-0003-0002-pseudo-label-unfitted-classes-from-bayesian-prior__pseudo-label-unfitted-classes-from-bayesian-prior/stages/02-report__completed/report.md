## Report: run-0003-0002 — Pseudo-label unfitted classes from Bayesian prior

**Run status:** succeeded
**Primary metric (val_soundscape_macro_roc_auc):** 0.6651
**Leader:** run-0002-0001-class-coverage-fix-resolve-23-unfitted-classes at 0.6727
**Delta vs leader:** -0.0076 (down)
**Delta vs parent (run-0001-perch-baseline at 0.6649):** +0.0001 (essentially flat)

### What happened

The experiment tested whether pseudo-labeling the 23 unfitted classes using high-confidence Bayesian prior predictions would improve coverage and discriminative power. It ran successfully but the result is effectively neutral — a +0.00013 gain over the parent baseline, well within noise.

### Why it matters

The core bottleneck remains unchanged: the embedding probe massively overfits (train 0.992 vs val 0.665). The Bayesian prior itself is paradoxically inverted, scoring 0.662 on validation but only 0.487 on train. Pseudo-labeling from this prior did not move the needle because:

1. The prior signal on unfitted classes is weak and the probe cannot extract useful gradients from noisy pseudo-labels.
2. The 23 unfitted classes lack enough training support for the pseudo-labels to stick.
3. The class-coverage-fix approach (the current leader at 0.6727) addressed the same problem from a different angle and outperformed by +0.0076.

### Secondary metrics

| Metric | Value |
|---|---|
| soundscape_macro_roc_auc (train) | 0.9918 |
| padded_cmap | 0.0625 |
| prior_fusion_macro_roc_auc (train) | 0.4873 |
| val_prior_fusion_macro_roc_auc | 0.6622 |
| oof_probe_macro_roc_auc | 0.5197 |

### Recommendation

Retire this branch. The pseudo-labeling axis shows negligible information gain. The leader (class-coverage-fix) has already demonstrated a more effective path to closing the coverage gap. The next decision should focus on:

- Building a submission bundle from the leader (run-0002-0001) to get a Kaggle scoreboard signal.
- Addressing the probe overfitting bottleneck directly, possibly through regularization, larger probe architecture, or better train/val alignment.
- Exploring prior calibration as a hedge branch, since the prior component (val 0.662) is competitive with the probe component (val 0.665) and may offer orthogonal signal when properly calibrated.
