# Experiment Conclusions

## run-0001-perch-baseline
- Experiment: `exp-perch-baseline`
- Best AUC: 0.665078
- Root cause: The Perch cached-probe baseline is locally validated at val ROC-AUC 0.665 with no leaderboard anchor. Submission bundle parity is the critical blocker — every downstream calibration, class-coverage expansion, and prior-fusion tuning decision is ungrounded without a Kaggle score. The 0.327 gap between training (0.992) and validation (0.665) soundscape ROC-AUC confirms domain shift or overfitting, but diagnosing it further before establishing a leaderboard baseline is premature.
- Verdict: submission-required

## run-0002-0002-calibrated-prior-fusion-v2
- Experiment: `exp-0002-calibrated-prior-fusion-v2`
- Best AUC: 0.664766
- Root cause: Calibrated prior fusion v2 (run-0002-0002) scored 0.6648 val ROC-AUC, a -0.0005 regression vs the leader run-0003-0001 (0.6652). The prior fusion component is near-random on train (0.487) and below raw probe on validation (0.662), confirming Bayesian prior calibration extracts no meaningful signal. The prior fusion branch is exhausted after two failed attempts (run-0003-0002 and run-0002-0002).
- Verdict: submission-required

## run-0003-0001-class-coverage-expansion-probe
- Experiment: `exp-0001-class-coverage-expansion-probe`
- Best AUC: 0.665216
- Root cause: The class-coverage-expansion-probe run (run-0003-0001) achieved val ROC-AUC 0.6652, essentially flat versus the prior baseline (0.6651). The lane leader MLP probe from run-0002-0003 holds 0.6801, which is the best score in branch. The primary bottleneck is the 0.327 train-val gap (0.992 train vs 0.665 val), indicating severe overfitting or domain shift. Padded_cmap at 0.062 confirms weak segment-level localization. Class coverage expansion alone produced no material gain. Five rounds of micro-tuning have been exhausted per the axis-exhaustion negative prior. Before packaging a submission, the higher-value move is to close the gap between the current best (0.6652) and the lane leader (0.6801) by adopting the MLP probe head configuration and then pursuing calibration and domain-shift mitigation.
- Verdict: submission-required
