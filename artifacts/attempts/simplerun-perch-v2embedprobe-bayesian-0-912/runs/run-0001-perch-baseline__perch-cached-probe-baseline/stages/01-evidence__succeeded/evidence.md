# Evidence Bundle run-0001-perch-baseline

- Work item: `workitem-perch-baseline`
- Experiment: `exp-perch-baseline`
- Run status: `succeeded`
- Primary metric: `val_soundscape_macro_roc_auc=0.6649754967376823`
- Verdict: `submission-required`
- Root cause: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.
- Delta vs leader: +0.000000

## Dataset Summary
- cache_row_count: 708
- fully_labeled_windows: 708
- fully_labeled_files: 59
- active_class_count: 75
- fitted_class_count: 52

## Runtime Summary
## Runtime Summary
- Backend: sklearn_cached_probe (reference_bayesian_pipeline)
- Cache root: /home/staff/jiayining/kaggle/kaggle_comp_seed/BirdCLEF-2026/input/perch-meta
- Cached rows: 708
- Fully labeled windows: 708
- Active classes: 75
- Fitted classes: 52
- Probe PCA dim: 32
- Probe min positives: 8
- Probe C: 0.25
- Use raw Perch scores: True
- Probe alpha: 0.4
- Prior lambda event: 0.4
- Prior lambda texture: 1.0
- Prior lambda proxy texture: 0.8
- Smooth texture alpha: 0.35
- prior_fusion_macro_roc_auc=0.487292
- soundscape_macro_roc_auc=0.991790 (full-pipeline resubstitution)
- oof_probe_macro_roc_auc=0.519454
- padded_cmap=0.062622
### Holdout Validation (sites: S03, S08, S13, S19)
- Train: 47 files (564 windows)
- Val: 12 files (144 windows)
- val_soundscape_macro_roc_auc=0.664975
- val_prior_fusion_macro_roc_auc=0.662212
- Val fitted classes: 34

## Observation Atoms
- `primary_metric` | `primary_metric` | val_soundscape_macro_roc_auc recorded for run-0001-perch-baseline
- `root_cause` | `runtime` | The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.

## Open Questions
- What exact change caused the current root cause: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.?
