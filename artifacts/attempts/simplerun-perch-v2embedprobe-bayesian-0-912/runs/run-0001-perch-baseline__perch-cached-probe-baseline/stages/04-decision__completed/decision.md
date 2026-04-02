## Decision: Iterate — 3-Branch Portfolio Before Submission

### Current State
- **Run**: run-0001-perch-baseline
- **Metric**: val_soundscape_macro_roc_auc = 0.665
- **Verdict**: submission-required (pipeline gap, not model gap)
- **Round**: 1 of expected multi-round cycle

### Why Not Submit Now
1. **Bundle parity unvalidated** — no CPU-only, internet-off dry-run exists yet. Submitting without confirming the bundle reproduces local predictions wastes daily quota.
2. **Submission bar compliance** — the baseline is round 1. The bar requires iteration through calibration → ensemble → soundscape prior → post-processing before submitting.
3. **Unresolved diagnostics** — fitted_class_count vs active_class_count is unknown; if coverage is low, post-hoc tuning is premature.

### Branch Portfolio (3 slots)

| # | Role | Component | Title | Rationale |
|---|------|-----------|-------|-----------|
| 1 | primary | backbone | Submission bundle parity + dry-run | Package Perch cache + linear probe + Bayesian prior into CPU-only bundle. Dry-run locally. **Blocking task.** |
| 2 | hedge | prior_calibration | Anchor-based prior calibration | The 0.662 val_prior_fusion vs 0.665 probe score gap suggests the Bayesian prior is functional but suboptimal. Test anchor-based calibration against LB history. Policy: conditional → prefer with evidence. |
| 3 | explore | probe_head | Class-count diagnostic + probe output inspection | Resolve the open question: what is fitted_class_count vs active_class_count? If coverage is adequate, this unblocks future threshold tuning. If not, it informs the next iteration. Minimal config change. |

### Veto Compliance
- **class_coverage**: Blocked per policy-class-coverage (avoid, conf=0.36). No branch targets class coverage expansion directly. The diagnostic branch (slot 3) inspects coverage but does not attempt to close the gap — it only resolves the informational uncertainty that makes the axis conditional.
- **Low-information patterns**: calibration_only and threshold sweeps are avoided as sole branches. Slot 2 tests a structured anchor-based approach, not a temperature grid.

### Deprioritized Axes
- **class_coverage**: Vetoed. Will re-evaluate after slot 3 diagnostic results.
- **data_filtering**, **preprocessing_aug**, **pseudo_label**: No supporting evidence for this family yet. Ledger miner shows no strong components.
- **optimization**: Premature until bundle parity and class-count questions are resolved.

### What This Unlocks
- Slot 1 confirms submission pipeline readiness → enables quota consumption in round 3+.
- Slot 2 either improves the prior fusion score or rules out calibration as the bottleneck.
- Slot 3 resolves the conditional probe_head policy by providing the missing fitted vs active class evidence.

### Policy Trace
- backbone: prefer (conf=0.36) → slot 1 justified by submission-findings card
- prior_calibration: conditional (conf=0.32) → slot 2 justified by post-parity consideration card, testing for promotion
- probe_head: conditional (conf=0.36) → slot 3 justified by open-questions card, providing evidence for future promotion/relegation
- class_coverage: avoid (conf=0.36) → respected, no branch targets expansion
