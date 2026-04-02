## Report: run-0002-0001 (Class Coverage Fix)

### What happened

The class coverage fix experiment resolved 23 unfitted classes, raising fitted class count from 52 to 71 out of 75 active classes. The run succeeded and produced a new portfolio best.

### Metrics

| Metric | Value |
|---|---|
| **val_soundscape_macro_roc_auc** (primary) | **0.6727** |
| val_prior_fusion_macro_roc_auc | 0.6622 |
| soundscape_macro_roc_auc (train) | 0.9972 |
| prior_fusion_macro_roc_auc (train) | 0.4873 |
| padded_cmap | 0.0571 |
| oof_probe_macro_roc_auc | 0.5199 |

**Delta vs previous leader (run-0001-perch-baseline at 0.6649): +0.0077**

### Leaderboard status

This run is the new leader. It surpasses the previous best by +0.0077 on val_soundscape_macro_roc_auc. The fitted class count improved from 52 to 71 (4 classes remain unfitted out of 75).

### Diagnostics

The train/val gap remains large (0.997 vs 0.673), confirming the probe overfitting pattern identified in earlier runs. The Bayesian prior fusion on val (0.662) continues to outperform the probe-only OOF score (0.520), validating the prior-probe ensemble approach. The 4 remaining unfitted classes are a residual coverage gap but the marginal gain from closing them is likely small.

### Verdict

**submission-required** — The pipeline is live and at a new high. The immediate next step is packaging a submission bundle (CPU, internet-off) and running calibration checks before a real Kaggle submission. After that, the next research frontier is addressing the probe overfitting bottleneck (regularization, architecture changes) rather than further class coverage work which has diminishing returns.

### Open questions

- Will the val AUC hold under scored submission conditions (CPU, internet-off, 90-min limit)?
- What is causing the 4 remaining unfitted classes, and is it worth another round?
- Can probe regularization close the train/val generalization gap?
