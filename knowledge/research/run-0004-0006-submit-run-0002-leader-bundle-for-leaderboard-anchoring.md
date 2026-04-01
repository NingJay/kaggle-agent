## Research: run-0004 Submission Bundle — Leaderboard Anchoring

### Current State

Run-0004 (submission bundle for workitem-0006) succeeded at **0.6709 val ROC-AUC**, a minor regression from the leader run-0002 at **0.6723**. Prior fusion remains non-functional (0.487 raw vs 0.662 val), confirming it must be bypassed. The critical blocker is the absence of a real Kaggle leaderboard score.

### Adopt-Now Priors

1. **Ship raw-probe bundle to Kaggle** `[card:research-run-0001-perch-baseline-md-adopt-now-guidance]`. No prior fusion. The 0.6709–0.6723 local range is the best available. A leaderboard score anchors all future calibration.
2. **Use holdout val as sole iteration signal** `[card:01-validated-findings-md-validation-and-metric-semantics]`. Policy: require (0.88). Ignore training metrics and prior_fusion metrics for decisions.
3. **Audit training temperature (7.39 → 1.0–2.0)** on run-0002 leader code state before further architecture changes `[card:research-run-0007-0006-perch-probe-round-7-audit-training-temperature-reduce-from-7-39-to-1-0-on-run-0002-leader-code-state-md-consider-for-round-8]`.
4. **Run-0002 is the leader base** `[branch:memory-0002]` at 0.6723 val, idea:combined, role:aspiration. All iterations should fork from this code state.

### Consider (Conditional)

- **Non-linear probe head** (embedding_dim → 256 → num_classes with ReLU) — test only after temperature audit `[card:research-run-0007-0006-perch-probe-round-7-audit-training-temperature-reduce-from-7-39-to-1-0-on-run-0002-leader-code-state-md-consider-for-round-8]`.
- **Prior calibration investigation** after temperature fix, contingent on resolving the train/val inversion `[card:research-run-0006-0005-perch-probe-round-6-focal-loss-on-run-0002-leader-code-state-md-actionable-guidance-for-round-7]`.
- **Bundle materialization regression diagnosis** — the 0.6723→0.6709 drop from packaging must be understood `[card:research-run-0002-0003-coverage-mlp-combined-full-75-class-mlp-probe-md-consider]`.

### Reject

- **Blocking submission for more iteration** — overrides are vetoed (0.78); the leaderboard data point is overdue.
- **Isolated temperature-scaling calibration** — did not close gaps in rounds 4–5; must pair with architecture changes.
- **Prior fusion blend tuning** — the 0.487/0.662 inversion confirms fusion is broken at the code level; tuning ratios on a broken component wastes budget.
- **Re-applying run-0003 bugfix** `[branch:memory-0003]` — it regressed to 0.6646 and did not alter the fusion computation path.

### Branch Memory Summary

| Memory | Run | Outcome | Val AUC | Delta | Signal |
|--------|-----|---------|---------|-------|--------|
| memory-0002 | run-0002 | leader | 0.6723 | +0.0071 | +3.0 |
| memory-0001 | run-0001 | regressed | 0.6652 | -0.0071 | -1.8 |
| memory-0003 | run-0003 | regressed | 0.6646 | -0.0077 | -1.8 |

### Next Action

Advance workitem-0006 to actual Kaggle submission using the run-0004 bundle (bypass prior fusion). Then iterate on temperature audit + MLP probe head as a combined change on run-0002 leader code state.
