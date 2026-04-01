## Research: Coverage + MLP Combined Branch (0.6727)

### Root Cause

The combined coverage + MLP regularization branch **regressed below both constituent parts** (0.6727 vs standalone MLP 0.6801 and standalone coverage 0.678). The two axes do not compose — combining them introduced interaction effects that hurt performance. A critical anomaly in the training prior_fusion path (0.487 train vs 0.662 val inversion) suggests a scoring bug that may be degrading optimization across all branches.

### Adopt Now

1. **Fix the prior_fusion training-time scoring bug.** The 175-point train/val inversion is the highest-priority root-cause repair. Resolving this may unlock gains across every branch, not just this one.
2. **Submit the standalone MLP leader (0.6801)** as the current best submission candidate — it remains the peak val ROC-AUC.
3. **Proceed to next iteration round** — the coverage expansion adopt-now confirms Round 2 is complete.

### Consider

- **Full 75/75 coverage closure as a standalone axis** (guard: only after fusion bug fix).
- **Calibration and loss tuning on the standalone MLP leader** rather than combining axes.
- **Standalone coverage expansion (0.678)** as a diversifying submission pair member.

### Reject

- **No more combined coverage + MLP experiments.** Proven non-composing.
- **No more isolated MLP regularization tuning.** Lane at diminishing returns (2/11 success rate).

### Structured Priors for Planning

| Type | Prior | Card ID |
|------|-------|---------|
| Positive | Fix prior_fusion training inversion before any further architecture work | `research-run-0002-0003-coverage-mlp-combined-full-75-class-mlp-probe-md-critical-anomaly` |
| Positive | MLP standalone leader at 0.6801 is current submission peak | `research-run-0011-0009-perch-probe-round-11-mlp-dropout-regularization-on-run-0008-leader-md-full-trajectory` |
| Conditional | Coverage gap 67→75/75 may yield +0.005–0.013 if fusion bug resolved | `research-run-0002-0001-perch-probe-full-class-coverage-expansion-md-root-cause-coverage-gap-narrowed-but-not-closed` |
| Conditional | Calibration + loss tuning on MLP leader, not combined axes | `research-run-0002-0001-perch-probe-full-class-coverage-expansion-md-consider` |
| Negative | Combined coverage + MLP does not compose — stop combining | `research-run-0002-0003-coverage-mlp-combined-full-75-class-mlp-probe-md-root-cause` |
| Negative | MLP regularization lane exhausted (2/11 success rate) | `research-run-0002-0003-coverage-mlp-combined-full-75-class-mlp-probe-md-reject` |
