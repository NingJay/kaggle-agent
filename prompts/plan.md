# Plan Program

Produce an executable experiment or submission plan with a config path, dedupe key, and launch mode.

Rules:
- Return `plan_status="planned"` for normal experiment or code-fix follow-up work.
- Use `plan_status="submission_candidate"` only when the input explicitly points to a submission-candidate packaging step.
- Do not use `submission_candidate` for training-mechanics fixes, debug reruns, or ordinary experiment iteration.
