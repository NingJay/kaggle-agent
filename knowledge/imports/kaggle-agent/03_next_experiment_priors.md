# Next Experiment Priors

## Priority Order

When choosing the next experiment, prefer this order:

1. Improve clean holdout signal.
2. Improve class coverage and fitted-class count.
3. Improve domain robustness.
4. Improve ensemble complementarity.
5. Improve submission calibration.
6. Only then spend time on cosmetic post-process sweeps.

## Perch Probe Lane

- If fitted classes lag active classes, expand coverage before micro-tuning.
- Compare learned probe against prior fusion on the same holdout split.
- Prefer experiments that change data coverage, class weighting, or domain robustness over tiny probe hyperparameter changes.

## SED Lane

Highest-value adoption candidates from the public SED comparison:

1. PCEN or equivalent adaptive frontend behavior.
2. Better secondary-label handling with explicit masking semantics.
3. Waveform-level background mixing and waveform-level MixUp.
4. Time shift augmentation.
5. LLRD, AMP, and gradient clipping.

Treat these as more important than superficial backbone churn.

## Domain Generalization Lane

Most promising DG-style moves:

1. Freq-MixStyle in SED training.
2. Stronger soundscape simulation or multi-clip mixing.
3. Tune the zero-shot plus fine-tuned blend ratio explicitly.
4. Consider GroupDRO only when site counts are sufficient.

## Submission Probe Lane

- Use submission slots to answer specific questions, not just to “submit the current best.”
- Good probe questions:
  which lane calibrates better to LB,
  whether PP gain transfers,
  whether an ensemble member carries unique LB signal,
  whether a candidate is stronger than its local ranking suggests.

## Anti-Priors

Do not prioritize these by default:

- padded cMAP optimization,
- resubstitution score chasing,
- source-only metric wins without holdout confirmation,
- leaderboard speculation without anchors,
- re-reading raw historical reports when the distilled rule already exists here.
