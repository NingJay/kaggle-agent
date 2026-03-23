# Kaggle Agent Contract

## Mission

Operate `kaggle_agent` as a three-plane autonomous research system for BirdCLEF 2026:

- deterministic control plane
- model-driven decision plane
- BirdCLEF runtime

## Hard Rules

- Do not reintroduce a monolithic `workspace.py` style control layer.
- Treat `state/` as the source of truth for experiments, runs, decisions, and submission candidates.
- Treat `artifacts/decision_briefs`, `artifacts/research`, `artifacts/decisions`, and `artifacts/plans` as the decision-plane API.
- Keep `train_sed.py` as the stable runtime launch bridge.
- Keep external model integrations behind adapter commands or optional runtime backends.

## Ask-First Boundaries

- Pushing Kaggle datasets or kernels
- Submitting notebooks online
- Installing or upgrading heavy ML dependencies globally
- Deleting archived legacy artifacts
- Killing non-agent processes

## Runtime Rules

- `configs/debug.yaml` must stay runnable without TensorFlow/Perch.
- `configs/default.yaml` should describe the real Perch-head baseline path.
- Runs must emit structured artifacts into `artifacts/runs/<run_id>/`.
- Decision processing must not be coupled into the runtime itself.

## Decision Rules

- Build `decision_brief` before research.
- Build research summary before decision record.
- Build decision record before planning or auto-launch.
- Auto-launch is allowed only when decision output and planner output are both valid.
