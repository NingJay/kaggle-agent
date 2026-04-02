# Session Memory

- current_objective: Continue frontier search, but enforce grounded-vs-novel slot allocation and minimum information gain.
- current_leader: `run-0001-perch-baseline`
- leader_metric: `val_soundscape_macro_roc_auc=0.6651564148629492`

## Top Positive Priors
- prior_calibration: repair the bottleneck with the strongest grounded prior first
- general: # Experiment Conclusions

## Unresolved Questions
- Resolve contradiction on prior_calibration: prior_calibration remains conditional because the evidence is mixed or incomplete. Counterevidence: repair the bottleneck with the strongest grounded prior first
- What removes the current bottleneck: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.?

## Current Bottlenecks
- The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.

## Pending Decisions
- Choose the next perch_cached_probe branch portfolio.
- Compile grounded and novel branch slots without violating veto rules.
