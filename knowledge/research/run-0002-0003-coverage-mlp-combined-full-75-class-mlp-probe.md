## Research: run-0002-0003 Coverage + MLP Combined 75-Class Probe

### Current State

The combined coverage expansion + MLP regularization branch achieved **val_soundscape_macro_roc_auc = 0.672**, which is below:
- The standalone MLP probe leader at **0.6801** (card: `...md-strategic-assessment`)
- The combined aspiration target of **0.70+**
- The individual coverage expansion run at **0.678** (card: `...md-root-cause-coverage-gap-narrowed-but-not-closed`)

The prior fusion metric (0.662) trails soundscape (0.672) by 10 points, indicating the Bayesian prior fusion layer is poorly calibrated.

### Critical Anomaly

The training-time prior_fusion score (0.487) is catastrophically below the validation score (0.662) — a 175-point inversion. This strongly suggests a bug in the training fusion scoring path (possibly training priors applied incorrectly, or the train/val split leaks label information differently for the fusion metric). This must be investigated before any calibration tuning.

### Root Cause

The combined branch did not exceed its constituent parts. The coverage and MLP axes are not composing synergistically — the 0.672 result is below both the standalone MLP (0.6801) and standalone coverage expansion (0.678). The primary bottleneck is the miscalibrated prior fusion layer, compounded by a likely scoring bug in the training fusion path.

### Adopt Now

1. **Use MLP probe head from run-0011-0009** as the baseline configuration — it holds the lane leader at 0.6801.
2. **Ensure full 75-class coverage** in all future runs — validated as a structural gain (+0.013).
3. **Override to continue-iterating** — 0.672 fails the submission bar.
4. **Investigate the train prior_fusion anomaly** (0.487 vs 0.662) as a blocking bug.

### Consider

- Prepare submission bundle infrastructure in parallel so it's ready when a run clears 0.680+.
- Tune prior fusion temperature/weights to close the 10-point soundscape-vs-fusion gap.
- Run an ablation without the fusion layer to isolate whether fusion is the degrader.
- Simplify the combined branch — the two axes may be interfering.

### Reject

- No further isolated MLP regularization rounds (lane at diminishing returns, 2-in-11 success rate).
- No submission of the current 0.672 run.
- No further exploration of the 4 exhausted axes from the 11-round trajectory.
