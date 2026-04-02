# Evaluation Contract Notes

This note captures the current operating interpretation for BirdCLEF evaluation inside this repo, using the local harness references as context.

## Official Competition Contract

- Official scored metric: macro ROC-AUC over non-empty classes.
- Scored submission path: CPU notebook, internet off, 90 minute runtime cap.

## Current Local Evaluation Layers

### 1. `val_soundscape_macro_roc_auc`

- Meaning: leak-aware site-level holdout validation on labeled soundscape data.
- Use: default keep or discard signal for local iteration when available.
- Trust: highest local trust among the current runtime metrics.

### 2. `val_prior_fusion_macro_roc_auc`

- Meaning: holdout score for the prior-fusion baseline on the same validation split.
- Use: compare whether the learned probe is actually beating the simpler prior path.
- Trust: clean local comparison, but still a source-domain validation number.

### 3. `prior_fusion_macro_roc_auc`

- Meaning: prior-fusion reference score outside the strict holdout keep/discard path.
- Use: proxy or calibration reference, not a standalone promotion rule.
- Trust: lower than holdout because it is not the main leak-free operating metric.

### 4. `soundscape_macro_roc_auc`

- Meaning: full-pipeline resubstitution style score kept for continuity and historical comparison.
- Use: sanity-check and backward compatibility only.
- Trust: do not use as the main promotion metric when holdout metrics are present.

### 5. `padded_cmap`

- Meaning: historical compatibility metric carried over from older code and reports.
- Use: archival comparison only.
- Trust: lowest relevance for current BirdCLEF 2026 decision making.

## Reading Historical Reports Correctly

- Older harness material often mixes holdout, proxy validation, post-processed local scores, predicted LB, and actual LB in the same page.
- Historical README material still mentions padded cMAP as the competition metric.
- Historical submission examples may assume GPU kernels; current BirdCLEF code competition scoring should remain CPU-first.

## Operating Rules

- Do not compare holdout validation and public LB as if they were the same scale.
- Do not let resubstitution metrics outrank holdout metrics in experiment triage.
- Always record which split a metric came from: holdout, proxy-target, resubstitution, post-process, or leaderboard.
- Always record trust semantics: clean, estimated, anchor-calibrated, or historical.
- Use submission anchors and calibration reports to connect local validation to LB strategy.
