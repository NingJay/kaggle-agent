## Research: run-0003-0001-class-coverage-expansion-probe

### Verdict: submission-required (flat)

Run run-0003-0001 achieved val ROC-AUC **0.6652**, essentially flat against the prior run (0.6651, memory-0001). The coverage expansion from 52→75 active classes with 708 cache rows did not materially move the primary metric.

### Key observations

- **Train-val gap remains severe**: 0.992 train vs 0.665 val ROC-AUC — overfitting or domain shift dominates.
- **Prior fusion adds negligible lift**: 0.662 vs 0.665 raw val, indicating miscalibrated Bayesian prior weighting.
- **Padded_cmap stays low** at 0.062 — segment-level localization is weak.
- **Lane leader** is still the MLP probe from run-0002-0003 at 0.6801.

### Adopt now

1. **Package submission bundle** from current best configuration; verdict is submission-required.
2. **Calibrate prior fusion** — temperature scaling or isotonic calibration on probe head logits.
3. **Adopt MLP probe head** from run-0002-0003 as baseline (lane leader 0.6801) — card:research-run-0002-0003-coverage-mlp-combined-full-75-class-mlp-probe-md-adopt-now.
4. **Require embedding-head models** as complementary signal — card:01-validated-findings-md-perch-and-embedding-findings.

### Consider (post-submission)

- Revisit the remaining 8-class coverage gap (67/75 fitted) — card:research-run-0002-0001-perch-probe-full-class-coverage-expansion-md-root-cause-coverage-gap-narrowed-but-not-closed.
- Investigate padded_cmap improvement as a secondary axis.
- Domain-shift mitigation via frequency augmentation on training soundscapes.
- Post-submission class coverage verification — card:research-run-0002-0001-perch-probe-iteration-class-coverage-verification-and-calibration-md-run-0002-0001-perch-probe-iteration-class-coverage-verification-and-calibration.

### Reject

- Further micro-tuning on exhausted axes (mixup, augmentation variants) — card:research-run-0009-0008-perch-probe-round-9-embedding-level-mixup-augmentation-on-run-0008-mlp-head-code-state-md-axis-exhaustion-summary.
- Re-running class coverage expansion in isolation (proven flat).
- MLP architecture variants before submission parity and calibration are established.
