## Report: run-0001-perch-baseline

### Status
The perch cached-probe baseline completed successfully in ~5 seconds. This is the first run in the experiment portfolio and establishes the performance floor.

### Metrics
| Metric | Value |
|---|---|
| val_soundscape_macro_roc_auc (primary) | **0.6650** |
| soundscape_macro_roc_auc | 0.9918 |
| padded_cmap | 0.0626 |
| prior_fusion_macro_roc_auc | 0.4873 |
| val_prior_fusion_macro_roc_auc | 0.6622 |
| oof_probe_macro_roc_auc | 0.5195 |

### Key Observations
- **Train/validation gap is large.** Training soundscape ROC-AUC is 0.992 vs validation 0.665, indicating the probe may be overfitting to the training soundscape windows.
- **OOF probe ROC-AUC is near random (0.519).** The out-of-fold probe barely beats chance, suggesting the linear probe on Perch embeddings alone is weak without the Bayesian prior.
- **Prior fusion hurts on train (0.487) but matches validation (0.662).** The Bayesian prior fusion pulls training predictions down significantly while barely affecting validation, reinforcing the overfitting diagnosis.
- **Class coverage is incomplete.** 75 active classes but only 52 fitted, meaning 23 classes have no probe model and rely entirely on the prior.
- **708 labeled windows across 59 files** — a small dataset, which makes overfitting expected.

### Dataset Context
- Cache rows: 708
- Fully labeled files: 59
- Active classes: 75, fitted: 52 (69%)

### Root Cause
The notebook-derived Bayesian prior + embedding probe pipeline is functional end-to-end. The immediate bottleneck is not metric performance but submission bundle readiness — the pipeline must be packaged into a CPU-first, internet-off notebook and validated before consuming a daily submission slot.

### Verdict: submission-required
The pipeline ran cleanly and produced reproducible artifacts. The next high-value action is building the submission bundle to confirm scoring parity on the Kaggle infrastructure.

### Queue Implications
- **This run is the current leader** by default (only run). Metric delta vs leader is 0.0 (flat).
- No anomalies detected.
- No open issues.
- The work item should advance to the `research` stage to scope submission bundle construction, or directly to `submission` if the bundle already exists from a prior iteration.

### Risks
- The 0.665 validation ROC-AUC is modest. Calibration improvements (temperature scaling, threshold tuning) and richer probe architectures (multi-layer, class-specific) are likely needed for competitive scores.
- The 23 unfitted classes will default to prior-only predictions, which may drag down macro-averaged metrics.
