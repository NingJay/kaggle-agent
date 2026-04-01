## Research Stage — Perch Cached-Probe Baseline

### Root Cause

The Perch cached-probe baseline achieved **val macro ROC-AUC 0.665** and is locally validated, but has **no leaderboard anchor**. The critical blocker is submission bundle parity. Without a Kaggle score, all downstream calibration and fusion work is ungrounded speculation.

Key diagnostic: training soundscape ROC-AUC (0.992) vs validation (0.665) reveals a 0.327 gap, indicating domain shift or overfitting. The prior-fusion val score (0.662) tracks raw validation closely, confirming the Bayesian prior adds negligible lift at this stage.

### Adopt Now

1. **Ship the raw-probe submission bundle** — no prior fusion, no calibration. Card `research-run-0001-perch-baseline-md-adopt-now-guidance` establishes this as the top priority. The 0.665 score is the honest baseline.
2. **Anchor iteration on holdout validation**, not training metrics. Card `01-validated-findings-md-validation-and-metric-semantics` confirms holdout as the most trustworthy signal.
3. **Retain embedding-head models** for future ensembling. Card `01-validated-findings-md-perch-and-embedding-findings` notes they provide complementary signal.

### Consider (Post-Submission)

1. **Class coverage deficit** — Card `research-run-0002-0001-perch-probe-iteration-class-coverage-verification-and-calibration` identifies 23/75 classes (31%) uncovered by the probe head. This is the structural bottleneck but must wait for a leaderboard anchor before investing cycles.
2. **Prior-fusion calibration** — Card `research-run-0001-perch-baseline-md-root-cause` frames this as the next calibration frontier once we have a Kaggle baseline.

### Reject

1. **Any calibration or class-coverage work before the raw-probe leaderboard submission** — Card `research-run-0001-perch-baseline-md-reject` vetoes this explicitly.
2. **Architecture changes to close the training-validation gap** before confirming leaderboard parity.

### Knowledge Cards Consulted

| Card ID | Stance | Policy | Confidence |
|---------|--------|--------|------------|
| `research-run-0001-perch-baseline-md-adopt-now-guidance` | positive | prefer | 0.68 |
| `research-run-0001-perch-baseline-md-root-cause` | positive | prefer | 0.68 |
| `research-run-0001-perch-baseline-md-reject` | negative | avoid | 0.58 |
| `research-run-0001-perch-baseline-md-current-state` | general | context | 0.42 |
| `01-validated-findings-md-perch-and-embedding-findings` | positive | require | 0.88 |
| `01-validated-findings-md-validation-and-metric-semantics` | positive | require | 0.88 |
| `research-run-0002-*class-coverage*` | conditional | conditional | 0.50 |
| `04-submission-bar-md-example` | general | context | 0.60 |

### Policy Trace

1. `require` @ 0.88: Holdout validation is primary iteration signal. Embedding heads retain future value.
2. `prefer` @ 0.68: Ship raw-probe bundle immediately. Submission parity over calibration.
3. `avoid` @ 0.58: No calibration or class-coverage work pre-submission.
4. `conditional` @ 0.50: Class-coverage expansion post-leaderboard-anchor only.
