# Session Memory

- current_objective: Probe overfitting mitigation and submission bundle parity
- current_leader: `run-0004-0003-class-coverage-fix-resolve-23-unfitted-classes`
- leader_metric: `val_soundscape_macro_roc_auc=0.6726994510451058`

## Top Positive Priors
- grounded: grounded has stronger supporting evidence than contradictory evidence.

## Unresolved Questions
- Resolve contradiction on class_coverage: class_coverage remains conditional because the evidence is mixed or incomplete. Counterevidence: Experiment: `exp-0001-perch-probe-full-class-coverage-expansion`
- Resolve contradiction on prior_calibration: prior_calibration remains conditional because the evidence is mixed or incomplete. Counterevidence: 1. **Fix class coverage (23/75 unfitted).** Card `imports-kaggle-agent-experiment-conclusions-md-run-0001-perch-basel...
- Resolve contradiction on probe_head: probe_head remains conditional because the evidence is mixed or incomplete. Counterevidence: Experiment: `exp-0008-perch-probe-round-9-embedding-level-mixup-augmentation-on-run-0008-mlp-head-code-state`
- What removes the current bottleneck: The notebook-derived Bayesian prior + embedding probe pipeline is live. Resolving 23 of the previously unfitted classes raised fitted count from 52 to 71/75, yielding a +0.0078 improvement. However, the severe train/val gap (train soundscape AUC 0.997 vs val 0.673) confirms that probe overfitting or distribution shift remains the structural bottleneck. Four classes remain unfitted. Next steps are submission bundle parity and calibration.?

## Current Bottlenecks
- The notebook-derived Bayesian prior + embedding probe pipeline is live. Resolving 23 of the previously unfitted classes raised fitted count from 52 to 71/75, yielding a +0.0078 improvement. However, the severe train/val gap (train soundscape AUC 0.997 vs val 0.673) confirms that probe overfitting or distribution shift remains the structural bottleneck. Four classes remain unfitted. Next steps are submission bundle parity and calibration.
- Probe overfitting mitigation and submission bundle parity

## Pending Decisions
- Choose the next perch_cached_probe branch portfolio.
