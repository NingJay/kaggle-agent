## Research: run-0001-perch-baseline

### Current State

The perch cached-probe baseline completed with **val_soundscape_macro_roc_auc = 0.665**. The run succeeded but the system correctly identifies that submission bundle parity is the blocking gap — not model iteration.

### Key Metric Breakdown

| Metric | Value | Trustworthiness |
|---|---|---|
| val_soundscape_macro_roc_auc | 0.665 | **Primary — holdout-aware** |
| val_prior_fusion_macro_roc_auc | 0.662 | Secondary reference |
| soundscape_macro_roc_auc (train) | 0.992 | Not trustworthy — train-side |
| prior_fusion_macro_roc_auc (unseen) | 0.487 | Weak reference only |

### Adopt Now

1. **Validate submission bundle parity.** Package the Perch embedding cache + linear probe + Bayesian prior into a CPU-only, internet-off notebook. Dry-run locally before any Kaggle submission. (submission-findings)
2. **Anchor on val_soundscape_macro_roc_auc.** This is the holdout-aware primary signal. Ignore the inflated train-side 0.992. (validation-and-metric-semantics, experiment-rules)

### Consider (Post-Parity)

- **Prior calibration refinement**: The 0.662 prior-fusion score vs 0.665 probe score suggests the Bayesian prior is close but not optimal. Anchor-based calibration against LB history could close this gap.
- **Class coverage check**: If fitted classes lag active classes, expand coverage before threshold/temperature tuning. Conditional until probe output dimensions are inspected.
- **Embedding ensemble**: A second complementary probe head could add signal, but only after the single-probe baseline submission is validated.
- **Submission bar compliance**: The baseline is round 1. The submission bar guidance recommends iterating through calibration, ensemble, soundscape prior, and post-processing before submitting.

### Reject

- **Do not submit without dry-run.** Consume no daily quota until the CPU bundle reproduces predictions locally.
- **Do not micro-tune the probe head** before bundle parity and class coverage are resolved.
- **Do not trust train-side metrics** for submission decisions.

### Policy Summary

| Component | Policy | Action |
|---|---|---|
| backbone | prefer | Stable — no changes needed |
| prior_calibration | prefer | Pursue after bundle parity |
| preprocessing_aug | prefer | Not blocking — defer |
| probe_head | conditional | Validate single-probe first |
| class_coverage | conditional | Inspect dimensions before tuning |

### Open Questions for Decision Stage

1. What is the fitted_class_count vs active_class_count for this probe head?
2. Does the current code already produce a submission-ready CSV, or does the bundle need new inference glue?
3. What is the expected LB score range given the 0.665 local validation?
