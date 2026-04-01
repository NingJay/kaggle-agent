## Research: Data Augmentation and Backbone Fine-Tuning (run-0002-0001)

### Outcome

Run-0002-0001 completed with **val_soundscape_macro_roc_auc = 0.6651**, a marginal +0.0002 over the baseline run-0001 (0.6650). This is functionally flat. The data augmentation and backbone fine-tuning experiment did not produce a meaningful improvement on the cached-probe pipeline.

Key secondary metrics:
- `prior_fusion_macro_roc_auc`: 0.4873 (degraded)
- `val_prior_fusion_macro_roc_auc`: 0.6622 (below raw probe, confirming prior fusion hurts at current weights)
- `soundscape_macro_roc_auc`: 0.9918 (train-side, not trustworthy for iteration)

### Root Cause

The blocking gap is **submission bundle parity**. No leaderboard anchor exists. All future calibration and fusion work is ungrounded without a real Kaggle score. The augmentation/fine-tuning path is yielding diminishing returns locally.

### Adopt Now

1. **Package run-0002-0001 as a submission bundle** — it holds the current local leader. Use raw probe output, not prior-fusion output (0.6622 < 0.6651). An anchor on the leaderboard is prerequisite for all downstream work.

2. **Treat holdout soundscape validation as the primary iteration signal** per validated findings (require@0.88) and runtime interpretation rules (conditional@0.62).

### Consider (Blocked Until Anchor)

- Prior-fusion weight recalibration after leaderboard score is obtained (conditional@0.50).
- The run-0002-0003 coverage-MLP branch achieved 0.672 in a different config — low-confidence lead worth investigating post-anchor.
- Leaderboard vs holdout scale calibration per comparison rules.

### Reject

- Further augmentation/fine-tuning iterations pre-anchor. The signal is flat.
- Embedding-level mixup on MLP probe head — degraded metrics in run-0009 (avoid@0.68).
- Any direct holdout-to-leaderboard score comparison without scale calibration.

### Branch Memory

Branch `perch_cached_probe` memory-0001 records baseline at 0.6650 with flat outcome and approved critic. Current run confirms the family is stable but not improving through augmentation alone. The next high-value move is a submission, not another training experiment.

### Knowledge Cards Consulted

8 cards retrieved from 6 knowledge files. Key cited cards:
- `research-run-0001-perch-baseline-md-root-cause` (prefer@0.68)
- `01-validated-findings-md-validation-and-metric-semantics` (require@0.88)
- `research-run-0009-0008-perch-probe-round-9-embedding-level-mixup-augmentation-on-run-0008-mlp-head-code-state-md-key-metrics` (avoid@0.68)
- `research-run-0001-perch-baseline-md-consider-conditional` (conditional@0.50)
