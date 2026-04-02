## Research Stage — run-0001-perch-baseline

### Root Cause Diagnosis

The probe model is the primary bottleneck. Train soundscape ROC-AUC (0.992) vs validation (0.665) reveals severe overfitting or a train/val distribution gap. The Bayesian prior is paradoxically stronger on validation (0.662) than train (0.487), confirming the probe itself is the weak link rather than the prior. Additionally, only 52 of 75 classes are fitted (a 31% coverage deficit), which structurally blocks submission bar criteria.

### Adopt Now

1. **Fix class coverage (23/75 unfitted).** Card `imports-kaggle-agent-experiment-conclusions-md-run-0001-perch-baseline` identifies this as the single blocking issue. No downstream work — calibration, ensemble, post-processing — is meaningful until all active classes receive fitted probes.

2. **Build submission bundle parity.** Package the current run as a CPU-only, internet-off bundle. Per `04_submission_bar`, this baseline does not clear the submission bar alone (round 1, simple linear probe), but establishing bundle parity unblocks the iteration loop.

3. **Use val_soundscape_macro_roc_auc as primary signal.** Per experiment rules (`imports-kaggle-agent-00-experiment-rules-md-current-runtime-interpretation`), holdout-aware soundscape validation is the intended primary metric; prior_fusion_macro_roc_auc is a weaker fallback only.

### Consider (Conditional)

- **Probe head regularization** — lighter than dropout (exp-0011 showed 0.3-0.5 dropout suppresses gradient signal on ~700-window training set). The branch_typing_compiler flags `calibration_only`, `blend_only`, and `temperature` as low-information patterns, so any intervention must carry novel signal.

- **Calibration on probe logits** — the Bayesian prior already provides useful holdout signal, but naive blend-ratio tuning regressed in exp-0010. Only pursue after class coverage is resolved.

### Reject

- **Naive prior-fusion blend ratio tuning** — exp-0010 showed MLP probe and Bayesian prior predictions are correlated, not complementary; blending dilutes discriminative power (-0.0148 regression).
- **Submitting this baseline without iteration** — per 04_submission_bar, round-1 Perch baselines are explicitly premature.
- **Heavy dropout regularization (0.3-0.5)** — exp-0011 confirmed it suppresses gradient signal without overfitting benefit on the small training set.
- **Random-guess configurations** — debug runs (run-0001, run-0003) showed frozen heads / mismatched labels produce ROC-AUC=0.5. Any new probe variant must verify gradient flow early.

### Contradictions

- **prior_calibration**: Mixed evidence — exp-0011 (dropout) is positive on the component but showed regression. Exp-0010 (blend ratio) is definitively negative. The lane has 2 of 11 interventions succeeding, indicating diminishing returns on probe-level work within the frozen Perch embedding space.
- **probe_head**: Mixed evidence — exp-0008 (embedding-level mixup) is positive but only marginally improved AUC. Earlier debug smokes are negative. Resolution requires testing whether the coverage deficit alone explains the train/val gap.

### Unresolved Questions

- Will fixing the 23 unfitted classes alone close the train/val AUC gap, or does the probe architecture itself need fundamental change?
- Is the frozen Perch embedding space a hard ceiling for this lane, as suggested by the 2/11 success rate in exp-0011?
