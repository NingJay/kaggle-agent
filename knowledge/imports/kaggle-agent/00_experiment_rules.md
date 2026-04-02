# Experiment Rules

## Metric Trust Order

Use these in descending trust for local experiment decisions:

1. `val_soundscape_macro_roc_auc`
   Meaning: leak-aware site holdout on labeled soundscapes.
   Use: main keep or discard signal when present.
2. `val_prior_fusion_macro_roc_auc`
   Meaning: same holdout split, simpler prior baseline.
   Use: check whether the learned probe is adding value over the prior path.
3. `prior_fusion_macro_roc_auc`
   Meaning: non-primary proxy reference.
   Use: comparison and calibration support only.
4. `soundscape_macro_roc_auc`
   Meaning: full-pipeline resubstitution or backward-compatibility score.
   Use: sanity-check only.
5. `padded_cmap`
   Meaning: historical carry-over metric.
   Use: archival comparison only.

## Comparison Rules

- Never compare holdout validation and leaderboard as if they were on the same scale.
- Never let resubstitution metrics outrank holdout metrics in local experiment promotion.
- Never compare proxy-target validation across models trained under clearly different regimes without labeling that caveat.
- Always label whether a number is raw, post-process, ensemble, predicted LB, or actual LB.

## Current Runtime Interpretation

- For the current cached-probe runtime, treat holdout-aware soundscape validation as the intended local primary signal whenever it is produced.
- If holdout validation is absent, fall back to `prior_fusion_macro_roc_auc` as a weaker reference, not as an equivalent substitute.

## Research Framing Rules

- The core domain gap is focal recordings to Pantanal soundscapes.
- This is a domain generalization problem, not source-free domain adaptation.
- `train_soundscapes_labels` is a labeled proxy-target bridge, not the true test distribution.

## Decision Rules

- Prefer structural improvements over cosmetic tuning.
- Structural means class coverage, domain robustness, loss semantics, augmentation design, and submission calibration.
- Cosmetic means small threshold or post-process tweaks without validation or anchor evidence.
