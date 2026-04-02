# Experiment Conclusions

## run-0001-perch-baseline
- Experiment: `exp-perch-baseline`
- Best AUC: 0.664975
- Root cause: The perch cached-probe baseline achieved val_soundscape_macro_roc_auc=0.665 but submission bundle parity is not yet validated. The submission bar guidance explicitly requires multi-round iteration before consuming daily quota. The current state is round 1 of the expected iteration cycle.
- Verdict: submission-required
