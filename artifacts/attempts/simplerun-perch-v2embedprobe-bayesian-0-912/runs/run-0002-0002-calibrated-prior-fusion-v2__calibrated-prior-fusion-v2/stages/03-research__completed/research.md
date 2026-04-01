## Research Stage — Calibrated Prior Fusion v2

### Result Summary

Run `run-0002-0002-calibrated-prior-fusion-v2` completed successfully with **val_soundscape_macro_roc_auc = 0.6648**, a -0.0005 regression versus the current leader `run-0003-0001` (0.6652). The prior fusion component itself is near-random on train (0.487) and only tracks the raw probe on validation (0.662), confirming calibration has not extracted meaningful lift.

### Root Cause

The Bayesian prior calibration pathway has been tested twice now (run-0003-0002 and run-0002-0002) and both times the fusion component adds no positive signal. The gap between train soundscape ROC-AUC (0.992) and validation (0.665) indicates overfitting/domain shift that prior fusion does not address. The raw probe head remains the best available signal source.

### Adopt Now

1. **Ship the leader bundle** — `run-0003-0001` (class-coverage-expansion-probe) at 0.6652 is critic-approved and ready. Use raw probe output exclusively; no prior fusion blending. [card:research-run-0003-0002-prior-fusion-weight-calibration-md-adopt-now]
2. **Retain perch embedding probe as primary signal** — embedding-head models provide complementary signal at high confidence (require @ 0.88). [card:01-validated-findings-md-perch-and-embedding-findings]

### Consider

- **MLP regularization without prior fusion** — the coverage + MLP branch context (0.672 on a different split) suggests the MLP probe head may have independent value worth isolating. [card:research-run-0002-0003-coverage-mlp-combined-full-75-class-mlp-probe-md-current-state]
- **Revised class coverage strategy** — the 52→75 expansion was flat, so brute-force coverage is exhausted. A smarter class selection or weighting scheme may help. [card:03-next-experiment-priors-md-perch-probe-lane]

### Reject

- **Prior fusion calibration** — two iterations confirm regression. Veto further fusion work until a fundamentally different prior source is available. [card:research-run-0003-0002-prior-fusion-weight-calibration-md-root-cause]
- **Segment-level localization investment** — padded_cmap (0.062) is consistently low but ROC-AUC is the primary metric and has not plateaued enough to justify pivoting.

### Decision Guidance

The next submission slot should go to the `run-0003-0001` raw-probe leader bundle. The calibrated prior fusion branch (tertiary, rank 1) should be marked as exhausted. Future experiment capacity should target probe-head architecture changes (MLP regularization, class weighting) or a new prior source entirely.
