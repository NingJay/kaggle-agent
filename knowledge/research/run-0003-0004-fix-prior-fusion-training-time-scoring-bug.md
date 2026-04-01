## Research: Prior Fusion Bugfix Assessment — run-0003-0004

### Verdict

The prior fusion scoring bugfix **did not work**. Val AUC regressed from 0.6723 (run-0002 leader) to 0.6646 (run-0003-0004). The prior_fusion train/val inversion persists at identical levels (train 0.487, val 0.662), confirming the fix did not alter the fusion computation path.

### Root Cause

The prior fusion component is structurally non-functional — this is not a scoring bug but a fundamental signal deficit. The linear probe on frozen Perch embeddings has hit a capacity ceiling. Multiple rounds of output-level interventions (temperature tuning, fusion bugfix) have failed to produce gains.

### Adopt Now

1. **Anchor on run-0002 leader** (0.6723 val AUC) as the baseline. Revert any code changes from the failed bugfix attempt.
2. **Disregard training-time prior_fusion metrics.** The persistent 175-point train/val inversion (0.487 vs 0.662) proves this metric path is unreliable for iteration decisions.
3. **Accept the linear probe capacity ceiling.** Further gradient-level or output-level tuning on the current architecture will not yield meaningful improvement.

### Consider

1. **MLP probe head** — Replace the linear layer with `embedding_dim → 256 → num_classes` (ReLU). This is the highest-priority single-axis experiment, directly targeting the confirmed capacity ceiling on run-0002 code state.
2. **Submit run-0002 leader** to Kaggle for a baseline leaderboard score. The pipeline verdict is submission-required; anchoring to the leaderboard enables informed iteration.
3. **Disable prior fusion** and evaluate raw probe predictions alone. If prior fusion adds no signal, removing it simplifies the submission bundle.
4. **Class coverage audit** — verify whether the 75-class cap excludes scored species that matter for macro ROC-AUC.

### Reject

1. Further temperature tuning experiments (proven dead in Round 7).
2. Further attempts to fix prior_fusion scoring (the fix attempt changed nothing structurally).
3. Gradient-level interventions on the frozen backbone (exhausted per strategic diagnosis).
