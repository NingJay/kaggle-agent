## Decision: Branch Portfolio for run-0001-perch-baseline

**Verdict:** Hold submission, launch multi-branch portfolio.

### Current State

- Val soundscape macro ROC-AUC: **0.665**
- Train soundscape macro ROC-AUC: 0.992 (severe overfitting)
- Classes fitted: **52/75** (31% coverage deficit)
- Prior fusion val AUC: 0.662 (Bayesian prior outperforms probe on validation)

### Root Cause

The probe model is the bottleneck. Train/val AUC gap (0.327) indicates overfitting on ~700 training windows. The Bayesian prior is stronger on validation than train, confirming the probe — not the prior — is the weak link. The 23 unfitted classes are a structural blocker per 04_submission_bar.md.

### Rationale

Three policy rules are active:
1. **Prefer** class_coverage — positive evidence from exp-perch-baseline.
2. **Avoid** probe_head — negative evidence from exp-0010 (blend regression) and exp-0011 (dropout failure). Veto checker enforces this.
3. **Conditional** prior_calibration — mixed evidence; override required.

The branch_typing_compiler flags calibration_only, blend_only, and temperature as low-information patterns. Naive blend-ratio tuning is definitively dead (exp-0010: -0.0148 regression). Heavy dropout is dead (exp-0011: gradient suppression with no benefit).

Given the 0.665 val AUC is well below the submission bar threshold for a round-1 baseline, and multiple high-value axes remain open, the correct decision is a multi-branch portfolio rather than a single low-information calibration sweep.

### Portfolio

| # | Branch Role | Component | Hypothesis | Rationale |
|---|---|---|---|---|
| 1 | **Primary** | class_coverage | Fixing the 23 unfitted classes will raise val AUC above 0.68 by closing the class coverage gap | Strongest policy support (prefer), addresses structural blocker, 31% coverage deficit is the single blocking issue |
| 2 | **Hedge** | prior_calibration | Label smoothing on probe logits will reduce train/val overfitting without suppressing gradient signal | Conditional policy; must pair with coverage fix; avoids forbidden calibration_only pattern by combining with class coverage work |
| 3 | **Novel** | pseudo_label | Pseudo-labeling from high-confidence Bayesian prior predictions on unfitted classes may improve coverage and discriminative power simultaneously | Novel hypothesis generator recommendation; unsupported but plausible; directly addresses the coverage gap from a data perspective rather than architecture |

### Deprioritized Axes

- **probe_head**: Vetoed. Naive blend-ratio tuning and heavy dropout are both definitively negative. Any future probe_head work requires explicit override justification.
- **calibration_only, blend_only, temperature**: Flagged as low-information by branch_typing_compiler. Only viable if they demonstrate higher information gain than baseline.

### Submission Stance

**Hold.** This is a round-1 Perch cached-embeddings + linear probe baseline. Per 04_submission_bar.md, round-1 baselines are explicitly premature regardless of score. The pipeline needs at minimum class coverage fixes and one successful regularization intervention before submission is warranted.
