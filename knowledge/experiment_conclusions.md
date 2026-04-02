# Experiment Conclusions

## run-0001-perch-baseline
- Experiment: `exp-perch-baseline`
- Best AUC: 0.664922
- Root cause: The probe model is the primary bottleneck: train soundscape ROC-AUC (0.992) vs val (0.665) reveals severe overfitting or train/val distribution gap. Only 52/75 classes are fitted (31% coverage deficit), which structurally blocks submission bar criteria. The Bayesian prior is paradoxically stronger on validation (0.662) than train (0.487), confirming the probe is the weak link, not the prior.
- Verdict: submission-required

## run-0002-0001-class-coverage-fix-resolve-23-unfitted-classes
- Experiment: `exp-0001-class-coverage-fix-resolve-23-unfitted-classes`
- Best AUC: 0.672667
- Root cause: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.
- Verdict: submission-required

## run-0003-0002-pseudo-label-unfitted-classes-from-bayesian-prior
- Experiment: `exp-0002-pseudo-label-unfitted-classes-from-bayesian-prior`
- Best AUC: 0.665053
- Root cause: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.
- Verdict: submission-required
