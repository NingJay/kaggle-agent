# Session Memory

- current_objective: Submission bundle assembly and calibration for run-0001-perch-baseline
- current_leader: `run-0001-perch-baseline`
- leader_metric: `val_soundscape_macro_roc_auc=0.6649679056916058`

## Top Positive Priors
- backbone: backbone has stronger supporting evidence than contradictory evidence.

## Top Negative Vetoes
- prior_calibration: prior_calibration has stronger contradictory evidence than supporting evidence.

## Unresolved Questions
- Resolve contradiction on class_coverage: class_coverage remains conditional because the evidence is mixed or incomplete. Counterevidence: Experiment: `exp-perch-baseline`
- Resolve contradiction on probe_head: probe_head remains conditional because the evidence is mixed or incomplete. Counterevidence: Embedding-head style models can add complementary signal even when their standalone score is not the best.
- What removes the current bottleneck: The notebook-derived Bayesian prior + embedding probe pipeline completed its first end-to-end run. The training+validation pipeline is functional but the submission bundle has not been built yet. The gap between train soundscape ROC-AUC (0.992) and validation soundscape ROC-AUC (0.665) signals meaningful overfitting or distribution shift, and prior fusion is not yet contributing (val prior fusion 0.662 vs val soundscape 0.665 — nearly identical). Calibration and bundle parity are the immediate blockers.?

## Current Bottlenecks
- The notebook-derived Bayesian prior + embedding probe pipeline completed its first end-to-end run. The training+validation pipeline is functional but the submission bundle has not been built yet. The gap between train soundscape ROC-AUC (0.992) and validation soundscape ROC-AUC (0.665) signals meaningful overfitting or distribution shift, and prior fusion is not yet contributing (val prior fusion 0.662 vs val soundscape 0.665 — nearly identical). Calibration and bundle parity are the immediate blockers.
- Submission bundle assembly and calibration for run-0001-perch-baseline

## Pending Decisions
- Choose the next perch_cached_probe branch portfolio.
