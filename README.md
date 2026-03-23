# Kaggle Agent

`kaggle_agent` is now organized as three explicit planes:

- `Deterministic Control Plane`
- `Model-Driven Decision Plane`
- `BirdCLEF Runtime`

The old monolithic `workspace.py` flow has been replaced by:

1. experiments and runs in `state/`
2. run artifacts in `artifacts/`
3. decision briefs, research summaries, decision records, and plans as first-class outputs
4. a rewritten BirdCLEF runtime under `BirdCLEF-2026-Codebase/`

## Core Directories

```text
kaggle_agent/
├── workspace.toml
├── artifacts/
│   ├── runs/
│   ├── decision_briefs/
│   ├── research/
│   ├── decisions/
│   ├── plans/
│   ├── reports/
│   └── submissions/
├── legacy/
├── state/
├── kaggle_agent/
│   ├── control/
│   ├── decision/
│   ├── adapters/
│   └── service.py
└── BirdCLEF-2026-Codebase/
```

## New CLI

```bash
python -m kaggle_agent.cli init
python -m kaggle_agent.cli doctor
python -m kaggle_agent.cli status
python -m kaggle_agent.cli enqueue-config /abs/path/to/config.yaml
python -m kaggle_agent.cli start-next
python -m kaggle_agent.cli tick
python -m kaggle_agent.cli watch --interval-seconds 600
python -m kaggle_agent.cli build-submission
```

## Runtime Notes

- `configs/debug.yaml` uses a mock hash backbone and a pure-Python trainer for smoke tests.
- `configs/default.yaml` is the real Perch-head path. It expects the saved model plus optional dependencies like TensorFlow and SoundFile.
- Every completed run writes:
  - `result.json`
  - `metrics.json`
  - `artifacts.json`
  - `summary.md`

## Adapter Contract

The decision plane can call external tools through command adapters configured in `workspace.toml`.

Each adapter receives:

- `KAGGLE_AGENT_STAGE`
- `KAGGLE_AGENT_WORKSPACE_ROOT`
- `KAGGLE_AGENT_INPUT_FILE`
- `KAGGLE_AGENT_OUTPUT_FILE`

Additional stage-specific environment variables may be provided.

If no adapter command is configured, the system falls back to internal heuristic research/decision/planning logic.
