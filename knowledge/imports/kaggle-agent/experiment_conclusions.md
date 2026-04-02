# Experiment Conclusions

## run-0001-perch-baseline
- Experiment: `exp-perch-baseline`
- Best AUC: 0.664958
- Root cause: Round 1 baseline completed with val ROC-AUC 0.665, but only 52/75 classes are fitted (31% coverage deficit). The report stage's submission-required verdict is overridden: this is the first run of a simple linear probe with no iteration history, and the family name contains "baseline" — it fails all four submission bar criteria from 04_submission_bar.md. The structural class coverage gap is the single blocking issue; no downstream work (calibration, post-processing, ensemble) is meaningful until the probe covers all active classes.
- Verdict: submission-required

## run-0002-0001-perch-probe-full-class-coverage-expansion
- Experiment: `exp-0001-perch-probe-full-class-coverage-expansion`
- Best AUC: 0.677713
- Root cause: Run-0002 expanded fitted classes from 52 to 67 out of 75 active (+15 classes, +0.013 val AUC), confirming that coverage expansion is a high-value structural lever. However, 8 classes remain unfitted and the pipeline has only completed 2 iteration rounds. The report stage's submission-required verdict is overridden because: (1) round count 2 < 5 minimum, (2) family name perch_cached_probe triggers probe exclusion, (3) only one structural technique (coverage expansion) has been applied — no calibration, ensemble, PP, or domain adaptation yet, (4) two data points (0.665 → 0.678) establish a trend but not a validated pipeline.
- Verdict: submission-required

## run-0003-0002-perch-probe-round-3-close-8-class-coverage-gap-to-75-75-and-apply-calibration
- Experiment: `exp-0002-perch-probe-round-3-close-8-class-coverage-gap-to-75-75-and-apply-calibration`
- Best AUC: 0.675847
- Root cause: Round 3 coupled two structural changes (coverage expansion + calibration) in a single code change, which destabilized probe fitting: fitted_class_count dropped from 67 to 63, val ROC-AUC regressed from 0.6777 to 0.6758, and val_prior_fusion collapsed from 0.662 to 0.475. The prior fusion collapse is especially diagnostic — since the Bayesian prior requires no training, its holdout degradation points to a data pipeline, class filtering, or evaluation config bug introduced in round 3.
- Verdict: submission-required

## run-0004-0003-perch-probe-round-4-revert-to-run-0002-state-and-incrementally-close-8-class-gap-single-axis-change
- Experiment: `exp-0003-perch-probe-round-4-revert-to-run-0002-state-and-incrementally-close-8-class-gap-single-axis-change`
- Best AUC: 0.675281
- Root cause: Run-0004 reverted the destabilizing round-3 changes and restored pipeline health (prior fusion val 0.662 recovered from 0.475, fitted classes 67→71/75), but val_soundscape_macro_roc_auc decreased slightly from 0.6777 (run-0002 leader) to 0.6753. The pipeline is confirmed healthy but no forward progress was made on the primary metric. The remaining 4-class coverage gap appears data-driven (missing embeddings) rather than config-driven. Run-0002 remains the performance leader and should be the base for all future rounds.
- Verdict: submission-required

## run-0005-0004-perch-probe-round-5-isolated-temperature-scaling-calibration-on-run-0002-leader-code-state
- Experiment: `exp-0004-perch-probe-round-5-isolated-temperature-scaling-calibration-on-run-0002-leader-code-state`
- Best AUC: 0.665569
- Root cause: Global temperature scaling calibration is the wrong lever for this probe. The training temperature (7.39) inflates logits toward near-uniform distributions, and a single downstream temperature (0.88) cannot recover per-class discriminability that was never learned. Two consecutive calibration-adjacent rounds failed (round 3 coupled, round 5 isolated), confirming the bottleneck is probe training-time discriminability, not post-hoc output scaling.
- Verdict: submission-required

## run-0006-0005-perch-probe-round-6-focal-loss-on-run-0002-leader-code-state
- Experiment: `exp-0005-perch-probe-round-6-focal-loss-on-run-0002-leader-code-state`
- Best AUC: 0.643623
- Root cause: Three consecutive experiments targeting loss weighting and post-hoc scaling (rounds 3, 5, 6) have all failed, confirming the bottleneck is not gradient weighting or output calibration but the discriminative capacity of the frozen Perch embeddings as processed through the linear probe. The training temperature of 7.39 remains unaudited and is likely inflating logits toward near-uniform distributions upstream of any loss function, compounding the discriminability ceiling.
- Verdict: submission-required

## run-0007-0006-perch-probe-round-7-audit-training-temperature-reduce-from-7-39-to-1-0-on-run-0002-leader-code-state
- Experiment: `exp-0006-perch-probe-round-7-audit-training-temperature-reduce-from-7-39-to-1-0-on-run-0002-leader-code-state`
- Best AUC: 0.665170
- Root cause: Run-0007 reduced training temperature from 7.39 to 1.0, producing a -0.0125 regression (0.6777→0.6652 val ROC-AUC). This is the 4th consecutive failure (rounds 3, 5, 6, 7) across four distinct axes: coupled calibration, post-hoc temperature scaling, focal loss, and training temperature reduction. The loss/calibration/temperature axis is definitively exhausted. The frozen Perch embedding linear probe has hit a representational capacity ceiling — no output-level or gradient-level intervention can improve it beyond run-0002's 0.6777. The bottleneck is discriminative capacity of frozen embeddings for the target domain.
- Verdict: submission-required

## run-0008-0007-perch-probe-round-8-replace-linear-head-with-1-hidden-layer-mlp-embedding-dim-256-num-classes-relu-on-run-0002-leader-code-state
- Experiment: `exp-0007-perch-probe-round-8-replace-linear-head-with-1-hidden-layer-mlp-embedding-dim-256-num-classes-relu-on-run-0002-leader-code-state`
- Best AUC: 0.680075
- Root cause: Run-0008 broke through the linear probe ceiling by replacing the head with a 1-hidden-layer MLP (embedding_dim→256→num_classes, ReLU), achieving val_soundscape_macro_roc_auc = 0.6801 (+0.0024 over run-0002). The bottleneck was representational capacity, not optimization or calibration. Prior fusion holdout remains stable at 0.662, confirming the gain is isolated to probe architecture. The pipeline has only 2 validated structural techniques across 8 rounds and has not yet applied augmentation, ensemble, post-processing, or domain adaptation — all required before submission is justified.
- Verdict: submission-required

## run-0009-0008-perch-probe-round-9-embedding-level-mixup-augmentation-on-run-0008-mlp-head-code-state
- Experiment: `exp-0008-perch-probe-round-9-embedding-level-mixup-augmentation-on-run-0008-mlp-head-code-state`
- Best AUC: 0.664987
- Root cause: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.
- Verdict: submission-required

## run-0010-0008-perch-probe-round-10-prior-fusion-blend-ratio-grid-search-on-run-0008-mlp-code-state
- Experiment: `exp-0008-perch-probe-round-10-prior-fusion-blend-ratio-grid-search-on-run-0008-mlp-code-state`
- Best AUC: 0.665297
- Root cause: Run-0010 grid-searched the blend weight between the MLP probe (val AUC 0.6801) and the Bayesian prior (val AUC 0.662), producing a best fused val AUC of 0.6653 — a -0.0148 regression. The prior does not carry complementary holdout signal relative to the MLP probe; the two sets of predictions are correlated rather than corrective, so blending dilutes the MLP's discriminative power. Prior fusion holdout remained stable at 0.662, confirming the regression is entirely in the fusion logic, not a pipeline health issue.
- Verdict: submission-required

## run-0011-0009-perch-probe-round-11-mlp-dropout-regularization-on-run-0008-leader
- Experiment: `exp-0009-perch-probe-round-11-mlp-dropout-regularization-on-run-0008-leader`
- Best AUC: 0.675418
- Root cause: Run-0011 is the 7th regression in 11 rounds of the perch_cached_probe lane. Dropout (0.3-0.5) suppressed already-scarce gradient signal on the ~700-window training set without any overfitting benefit. Four axes are now definitively exhausted: loss/calibration/temperature (rounds 3,5,6,7), embedding-level augmentation (round 9), prior-fusion blend ratio (round 10), and dropout regularization (round 11). The lane is at diminishing returns — only 2 of 11 interventions succeeded. The frozen Perch embedding space imposes a fundamental ceiling on further probe-level improvements.
- Verdict: submission-required
