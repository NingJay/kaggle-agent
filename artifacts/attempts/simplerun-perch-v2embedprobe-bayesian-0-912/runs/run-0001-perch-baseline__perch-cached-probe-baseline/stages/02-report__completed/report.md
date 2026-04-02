## Report: run-0001-perch-baseline

**Status**: Succeeded | **Verdict**: submission-required | **Stage cursor**: report

### Primary metric

| Metric | Value |
|---|---|
| val_soundscape_macro_roc_auc | **0.6649** |

### Secondary metrics

| Metric | Value | Notes |
|---|---|---|
| soundscape_macro_roc_auc (train) | 0.9918 | Very high — overfitting signal |
| val_prior_fusion_macro_roc_auc | 0.6622 | Close to probe-only val, prior adds little |
| prior_fusion_macro_roc_auc (train) | 0.4873 | Below random — broken on train |
| padded_cmap | 0.0625 | Low, expected for baseline |
| oof_probe_macro_roc_auc | 0.5200 | Barely above random |

### Key observations

1. **Large train/val gap on soundscape ROC-AUC**: 0.992 vs 0.665. The probe is memorizing train soundscapes. This is expected given the small cache (708 windows, 59 files, 52 fitted classes) but limits generalization confidence.

2. **Prior fusion is inconsistent**: Train prior fusion (0.487) is below random while val prior fusion (0.662) is near the probe-only val score. This suggests the Bayesian prior component may not be correctly calibrated across splits, or the train-set prior is being dominated by label noise.

3. **OOF probe at 0.520**: The out-of-fold probe ROC-AUC is only slightly above 0.5, confirming the linear probe on Perch embeddings has limited discriminative power with current configuration. This is the main bottleneck.

4. **Dataset coverage**: 75 active classes but only 52 fitted — 23 classes lack sufficient training examples for the probe.

### Leaderboard position

This is the first completed run, so `run-0001-perch-baseline` is the default leader at 0.665.

### Root cause

The embedding probe itself is the weak link. The Perch embeddings carry signal (train ROC-AUC is high), but the linear probe fails to generalize (OOF 0.520, val 0.665). The Bayesian prior adds marginal value on validation.

### Recommended next steps

- **Immediate**: Package submission bundle for `run-0001-perch-baseline` to get a Kaggle leaderboard score and calibrate the local val metric.
- **Research**: Investigate probe architecture (deeper head, class-weighted loss, label smoothing), missing-class handling for the 23 unfitted classes, and whether the train soundscape overfit can be reduced via regularization or data augmentation.
- **Queue**: Decision stage should decide between (a) submitting this baseline as-is, (b) iterating on probe calibration before submission, or (c) both in parallel.
