## Research: run-0003-0002 Prior-Fusion Weight Calibration

### Root Cause

Prior-fusion weight calibration (run-0003-0002) scored val_soundscape_macro_roc_auc=0.6648, a -0.003 regression vs the leader run-0002-0001 (0.6651). The fusion component itself (val_prior_fusion_macro_roc_auc=0.6622) underperforms the raw probe, confirming the Bayesian prior weighting adds noise at current calibration settings.

### Adopt Now

1. **Ship run-0002-0001 raw-probe submission bundle** — it holds the local leader at 0.6651. Use raw probe output exclusively; prior fusion degrades it (0.6622). The bundle is already validated, critic-approved, and has submission_status=candidate_created. `[card:research-run-0002-0001-data-augmentation-and-backbone-fine-tuning-md-adopt-now]` `[memory:memory-0002]`
2. **No calibration before anchor** — the honest baseline must reach the Kaggle leaderboard before any calibrated variant is considered. `[card:research-run-0001-perch-baseline-md-adopt-now-guidance]`

### Consider (Post-Anchor Only)

- Prior-fusion weight calibration may be revisited after a leaderboard anchor is obtained. Any future calibration experiment must demonstrate positive delta vs the raw probe on both local validation and public LB. `[card:research-run-0001-perch-baseline-md-consider-conditional]`
- The -0.003 regression is small enough to be a validation split artifact, but disambiguating this is lower priority than shipping the leader bundle.

### Reject

- **Do not submit run-0003-0002 or any calibrated variant to Kaggle** — override the submission-required verdict. Temperature scaling and calibration experiments have produced consistent regression across runs-0004, 0005, and now 0003-0002. `[card:research-run-0005-0004-perch-probe-round-5-isolated-temperature-scaling-calibration-on-run-0002-leader-code-state-md-verdict-continue-iterating-override]`
- **Deprioritize the entire prior_calibration idea class** until post-anchor re-evaluation. `[card:research-run-0001-perch-baseline-md-reject]` `[card:research-run-0005-0004-perch-probe-round-5-isolated-temperature-scaling-calibration-on-run-0002-leader-code-state-md-key-metrics-comparison]`

### Branch Portfolio Summary

| Branch | Outcome | Val ROC-AUC | Delta | Status |
|--------|---------|-------------|-------|--------|
| run-0002-0001 (training_axis) | submission_candidate | 0.6651 | +0.0002 | validated, approved |
| run-0001 (baseline) | flat | 0.6650 | -0.0002 | validated, approved |
| run-0003-0002 (calibration) | regression | 0.6648 | -0.0030 | succeeded |

### Recommendation for Decision Stage

Verdict: **advance run-0002-0001 raw-probe bundle to submission**. Deprioritize prior_calibration branch. No further calibration experiments should be queued until the leaderboard anchor is confirmed.
