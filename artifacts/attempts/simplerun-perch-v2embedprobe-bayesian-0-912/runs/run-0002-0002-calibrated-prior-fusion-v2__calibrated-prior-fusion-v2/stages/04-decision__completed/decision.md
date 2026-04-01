## Decision: Exhaust prior fusion branch, ship leader, plan probe-head improvements

### Current Run Verdict
Calibrated prior fusion v2 (run-0002-0002) is a **regression** at 0.6648 vs the leader 0.6652. The prior fusion component itself is near-random on train (0.487) and below raw probe on validation (0.662). This branch is now **exhausted** after two failed iterations.

### Submission Recommendation
**Ship the run-0003-0001 raw-probe leader bundle** (0.6652, critic-approved). No prior fusion blending. This uses one of the 5 daily submission slots on the strongest validated candidate.

### Branch Search Priority
1. **Close prior fusion lane** — two negative results with consistent signal (prior adds negative value). Veto any further iterations without a fundamentally different prior source.
2. **Deprioritize brute-force coverage expansion** — 52→75 class expansion was flat; future coverage work needs a revised class selection strategy.
3. **Open probe-head architecture axis** — MLP regularization (without prior fusion) showed a different metric context at 0.672. Investigate whether probe-head architectural changes can lift val_soundscape_macro_roc_auc standalone.
4. **Preserve submission parity** — the baseline fallback (run-0001, 0.6651) remains a second submission candidate.

### Next Plan Scope
The plan stage should:
- Mark prior fusion branch as exhausted in portfolio policy
- Generate a probe-head architecture experiment (e.g., deeper MLP, dropout regularization, layer norm) targeting the raw probe output
- Consider revised class selection (frequency-weighted, geography-aware) rather than brute-force expansion
- Bundle the leader for submission in parallel

### Portfolio Intent
Primary data_coverage branch holds the leader. Probe-head architecture is the next high-value axis. Prior fusion is closed.
