# Session Memory

- current_objective: Three factors block immediate submission: (1) the CPU-only, internet-off bundle has not been dry-run locally, (2) the submission bar requires at least calibration and ensemble iteration beyond the raw baseline, and (3) open questions about fitted_class_count vs active_class_count remain unresolved. The veto checker blocks class_coverage axes, and the branch diversifier recommends backbone/probe_head/prior_calibration as the viable mix. Collapsing to a single low-information calibration sweep would waste branch budget when the bundle pipeline itself is unvalidated.
- current_leader: `run-0001-perch-baseline`
- leader_metric: `val_soundscape_macro_roc_auc=0.6649754967376823`

## Top Positive Priors
- backbone: backbone has stronger supporting evidence than contradictory evidence.
- preprocessing_aug: preprocessing_aug has stronger supporting evidence than contradictory evidence.

## Unresolved Questions
- Resolve contradiction on probe_head: probe_head remains conditional because the evidence is mixed or incomplete. Counterevidence: 1. **Validate submission bundle parity.** Package the Perch embedding cache + linear probe + Bayesian prior into a CP...
- What removes the current bottleneck: The perch cached-probe baseline (val_soundscape_macro_roc_auc=0.665) is live but submission bundle parity is not yet validated. The immediate gap is not model performance but submission pipeline readiness: the notebook-derived Bayesian prior + embedding probe pipeline needs a CPU-only, internet-off bundle that reproduces the local score before any daily submission quota is consumed. The val_prior_fusion_macro_roc_auc (0.662) is close to the primary metric (0.665), confirming the probe head is working but calibration refinement remains conditional.?

## Current Bottlenecks
- The perch cached-probe baseline (val_soundscape_macro_roc_auc=0.665) is live but submission bundle parity is not yet validated. The immediate gap is not model performance but submission pipeline readiness: the notebook-derived Bayesian prior + embedding probe pipeline needs a CPU-only, internet-off bundle that reproduces the local score before any daily submission quota is consumed. The val_prior_fusion_macro_roc_auc (0.662) is close to the primary metric (0.665), confirming the probe head is working but calibration refinement remains conditional.

## Pending Decisions
- Choose the next perch_cached_probe branch portfolio.
- Compile grounded and novel branch slots without violating veto rules.
