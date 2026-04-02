# Run Report run-0001-perch-baseline

- Headline: Keep iterating on the probe stack
- Current focus: root-cause repair and next config selection
- Run status: `succeeded`
- Primary metric: `val_soundscape_macro_roc_auc=0.6651564148629492`
- Root cause: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.
- Current ledger leader: `run-0001-perch-baseline` with val_soundscape_macro_roc_auc=0.665156

## Findings
- Perch cached-probe baseline completed: val_soundscape_macro_roc_auc=0.6651564148629492 and verdict submission-required.

## Promotion Candidates
- Promote run-0001-perch-baseline if validation remains trustworthy against the current leader.

## Next Questions
- What exact change caused the current root cause: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.?
