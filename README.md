# Kaggle Agent

`kaggle-agent` is now a BirdCLEF-focused Kaggle Research OS:

- agent-primary
- artifact-driven
- spec-enforced
- ledger-backed

The public surface is a working research room, not a thin orchestration shell. The machine truth lives in `state/ledger.db`. Every autonomous stage emits both `*.json` and `*.md`. Reports, checklists, findings, issues, and submission intelligence are regenerated from the ledger.

## Stage Graph

```text
work_item
-> execute
-> evidence
-> report
-> research
-> decision
-> plan
-> codegen
-> critic
-> validate
-> submission
```

The runtime stays BirdCLEF-specific by default through `train_sed.py` and `BirdCLEF-2026-Codebase/`, but the orchestration kernel remains lightly generic.

## Headless Adapters

The default adapter routing is now:

- `report`, `research`, `decision` -> Claude Code headless
- `plan`, `codegen` -> Claude Code on isolated stage worktrees by default
- `plan`, `codegen` can also be switched back to Codex `exec`
- `critic` -> Claude Code headless with optional Amp sidecar
- `evidence`, `validate`, and scored submission bundling stay deterministic

The wrappers live at `kaggle_agent.adapters.stage_wrapper` and consume the existing stage contract:

- `KAGGLE_AGENT_STAGE`
- `KAGGLE_AGENT_WORKSPACE_ROOT`
- `KAGGLE_AGENT_INPUT_MANIFEST`
- `KAGGLE_AGENT_OUTPUT_DIR`
- `KAGGLE_AGENT_PROMPT_FILE`

Each wrapper writes:

- `<stage>.json`
- `<stage>.md`
- `provider_meta.json`
- `raw_stdout.txt`
- `raw_stderr.txt`
- `events.jsonl` when the provider supports event streaming

Optional environment variables when you are not already logged into the local CLIs:

- `ANTHROPIC_API_KEY` for Claude-backed stages
- `CODEX_API_KEY` or `OPENAI_API_KEY` for Codex-backed stages
- `AMP_API_KEY` for the optional Amp critic sidecar

If a provider binary is missing, the wrapper exits with a soft-skip code and the existing deterministic fallback takes over. Claude/Codex can also run off local CLI login state without explicit env vars. If the provider returns malformed JSON or violates the stage schema, the stage fails hard.

## Isolated Stage Worktrees

`plan` and `codegen` no longer run directly against the main repo root. The wrapper now prepares an isolated source workspace under `state/worktrees/<stage>/<stage-run>/workspace` before invoking the provider.

- `plan` gets an isolated source snapshot so repo context stays stable across runs.
- `codegen` edits only its isolated workspace and writes deterministic verify artifacts under `state/worktrees/codegen/<stage-run>/verify_runtime`.
- The control plane still records canonical artifacts into the normal stage output directory, but the AI-facing code context is no longer the live main workspace.

## Core Surface

```text
.
├── AGENTS.md
├── COMPETITION.md
├── PLAYBOOK.md
├── CHECKLIST.md
├── JOURNAL.md
├── FINDINGS.md
├── ISSUES.md
├── SUBMISSIONS.md
├── prompts/
├── reports/
├── state/
├── artifacts/
├── knowledge/
└── BirdCLEF-2026-Codebase/
```

## Key Commands

```bash
python -m kaggle_agent.cli init
python -m kaggle_agent.cli doctor
python -m kaggle_agent.cli status
python -m kaggle_agent.cli list-ready
python -m kaggle_agent.cli start-next --sync
python -m kaggle_agent.cli tick
python -m kaggle_agent.cli watch --interval-seconds 600
python -m kaggle_agent.cli build-submission --run-id <run_id>
python -m kaggle_agent.cli dry-run-submission <candidate_id>
python -m kaggle_agent.cli plan-submission
```

## Runtime Notes

- `configs/debug.yaml` is the smoke path and must stay runnable without TensorFlow or scikit-learn.
- `configs/default.yaml` is the cached Perch probe baseline.
- `BirdCLEF-2026-Codebase/src/birdclef_runtime/training.py` lazily imports the cached-probe backend so debug smoke does not require heavy probe dependencies at module import time.

## Submission Contract

BirdCLEF scored bundles are treated as:

- CPU-only
- internet off
- 90 minute maximum runtime
- 5 daily submission slots
- 2 final selection slots

`build-submission` now writes a usable bundle:

- `candidate_manifest.json`
- `candidate.md`
- `bundle_runner.py`
- `notebook.ipynb`
- `kernel-metadata.json`
- `dry_run.json`

The local dry-run writes a `submission.csv` from the competition sample format so the bundle is contract-checked before any human-gated online push.
