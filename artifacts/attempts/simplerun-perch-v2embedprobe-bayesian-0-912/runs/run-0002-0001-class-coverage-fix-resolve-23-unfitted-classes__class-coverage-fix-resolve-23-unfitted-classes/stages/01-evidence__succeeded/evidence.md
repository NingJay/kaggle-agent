# Evidence Bundle run-0002-0001-class-coverage-fix-resolve-23-unfitted-classes

- Work item: `workitem-0001-class-coverage-fix-resolve-23-unfitted-classes`
- Experiment: `exp-0001-class-coverage-fix-resolve-23-unfitted-classes`
- Run status: `succeeded`
- Primary metric: `val_soundscape_macro_roc_auc=0.6726671893669534`
- Verdict: `submission-required`
- Root cause: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.
- Delta vs leader: +0.000000
- Delta vs parent: +0.007745

## Dataset Summary
- cache_row_count: 708
- fully_labeled_windows: 708
- fully_labeled_files: 59
- active_class_count: 75
- fitted_class_count: 71

## Runtime Summary
## Runtime Summary
- Backend: sklearn_cached_probe (reference_bayesian_pipeline)
- Cache root: /home/staff/jiayining/kaggle/kaggle_comp_seed/BirdCLEF-2026/input/perch-meta
- Cached rows: 708
- Fully labeled windows: 708
- Active classes: 75
- Fitted classes: 71
- Probe PCA dim: 32
- Probe min positives: 1
- Probe C: 0.25
- Use raw Perch scores: True
- Probe alpha: 0.4
- Prior lambda event: 0.4
- Prior lambda texture: 1.0
- Prior lambda proxy texture: 0.8
- Smooth texture alpha: 0.35
- prior_fusion_macro_roc_auc=0.487292
- soundscape_macro_roc_auc=0.997217 (full-pipeline resubstitution)
- oof_probe_macro_roc_auc=0.519901
- padded_cmap=0.057084
### Holdout Validation (sites: S03, S08, S13, S19)
- Train: 47 files (564 windows)
- Val: 12 files (144 windows)
- val_soundscape_macro_roc_auc=0.672667
- val_prior_fusion_macro_roc_auc=0.662212
- Val fitted classes: 51

## Observation Atoms
- `primary_metric` | `grounded` | val_soundscape_macro_roc_auc recorded for run-0002-0001-class-coverage-fix-resolve-23-unfitted-classes
- `root_cause` | `grounded` | The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.

## Open Questions
- What exact change caused the current root cause: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.?
