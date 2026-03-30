# Plan Program

Produce an executable experiment or submission plan with a config path, dedupe key, and launch mode.

Rules:
- Return `plan_status="planned"` for normal experiment or code-fix follow-up work.
- Use `plan_status="submission_candidate"` only when the input explicitly points to a submission-candidate packaging step.
- Do not use `submission_candidate` for training-mechanics fixes, debug reruns, or ordinary experiment iteration.
- Read and use the provided `# Knowledge Context` section explicitly instead of relying only on the input manifest.
- When holdout validation exists, treat `val_soundscape_macro_roc_auc` as the primary keep/discard metric. Treat resubstitution `soundscape_macro_roc_auc` as diagnostic only.
- If class imbalance or class coverage is the blocking issue, plan to fix coverage first and defer calibration-only tuning until coverage is acceptable.
