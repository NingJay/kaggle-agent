## Decision: Submit Raw Probe Baseline to Leaderboard

**Verdict:** submission-required → plan submission bundle

**Current state:** Val ROC-AUC 0.665, training ROC-AUC 0.992, no leaderboard anchor exists.

**Decision rationale:**

1. **Policy convergence on "ship now":** All prefer-tier cards (adopt-now guidance, root-cause analysis) point to submission bundle parity as the single critical blocker. The 0.665 local score is the honest baseline — it must be grounded against Kaggle before any downstream work can be meaningfully prioritized.

2. **Veto compliance:** The avoid-tier card explicitly blocks calibration, class-coverage expansion, and prior-fusion tuning before a leaderboard submission. Complying with this veto means the next action cannot be a model iteration — it must be a submission packaging action.

3. **Conditional leads deferred correctly:** The class-coverage deficit (23/75 classes, 31% uncovered) identified in run-0002 is a genuine structural bottleneck, but it is gated on a post-leaderboard-anchor condition. It remains deprioritized until we have a Kaggle score.

4. **Multi-branch search preserved:** After the leaderboard anchor is established, the next plan stage should fan out across the highest-value axes: (a) class-coverage expansion for the 23 uncovered classes, (b) domain-shift mitigation between training and validation soundscapes, and (c) calibrated prior fusion. The leaderboard delta will determine the relative priority of these branches.

**Submission stance:** Submit the raw probe output with no prior fusion, no calibration, and no post-processing. The goal is a clean leaderboard anchor, not an optimized score.

**Deprioritized axes (post-submission only):**
- Calibration / temperature scaling
- Prior fusion tuning
- Model architecture changes for the 0.327 training-validation gap

**Active axes (for post-submission plan):**
- Class-coverage expansion (52→75 classes)
- Domain shift diagnosis (training vs validation soundscape)
- Ensemble signal from embedding heads

**Human required:** Yes — this is a real Kaggle submission and falls under ask-first boundaries.
