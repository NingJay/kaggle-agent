# Attempt Reset And Human-Readable Surface Design

## Summary

`kaggle_agent` needs a clean restart flow for a failed attempt and a filesystem-first artifact layout that a human can inspect without reading ledger internals. The current layout mixes machine buckets such as `artifacts/decision/` with ambiguous stage directories such as `stage-0004-decision`, and the default `workitem-perch-debug-smoke` forces a smoke-style bootstrap into the main experiment narrative.

This design introduces:

- A hard reset path that clears `artifacts/`, `state/`, `reports/`, and the root surface files
- A current-attempt manifest rooted at the notebook-derived slug `simplerun-perch-v2embedprobe-bayesian-0-912`
- Human-readable canonical artifact paths under `artifacts/attempts/...`
- An explicit preflight entrypoint that is separate from the default experiment queue

## Goals

- Let the user restart from a failed attempt without retaining stale machine outputs
- Make the filesystem itself legible enough for manual intervention and review
- Preserve `ledger.db` as machine truth while aligning artifact paths and surface files with human mental models
- Remove smoke/debug runs from the default experiment boot sequence

## Non-Goals

- Replacing the strict stage graph
- Removing structured stage JSON or markdown outputs
- Building a new web UI
- Archiving the current failed attempt before reset

## Design

### 1. Hard Reset

Add a reset-capable initialization path that:

- Deletes `artifacts/`, `state/`, and `reports/`
- Recreates those directories and a fresh ledger
- Rewrites `CHECKLIST.md`, `JOURNAL.md`, `FINDINGS.md`, `ISSUES.md`, and `SUBMISSIONS.md` from templates
- Seeds runtime state with a single current-attempt slug: `simplerun-perch-v2embedprobe-bayesian-0-912`

This is intentionally destructive and should only happen through an explicit command path.

### 2. Attempt-Centric Artifact Layout

Canonical stage artifacts move from:

- `artifacts/decision/stage-0004-decision`

to:

- `artifacts/attempts/<attempt_slug>/runs/<run_slug>/stages/04-decision__<status>/`

The directory name itself must convey:

- stage ordering
- stage name
- current stage outcome

Each stage directory still contains the same machine files: `input_manifest.json`, stage JSON, stage markdown, and provider metadata when available.

### 3. Human-Readable Run Naming

Runs and stage folders should expose both machine ids and readable labels:

- run slug: `<run_id>__<experiment_slug>`
- stage slug: `<index:02d>-<stage_name>__<status_slug>`

The stable ids remain in the ledger; the readable slug is for artifact paths and status reporting.

### 4. Explicit Preflight

Replace the default `workitem-perch-debug-smoke` seed with:

- the real notebook-derived baseline work item as the default queue root
- a dedicated preflight command or work item factory that can be invoked manually

Preflight runs should be clearly labeled as preflight in both work item titles and artifact paths, and should not gate the main experiment queue unless the human explicitly wants that.

### 5. Surface Files Follow Attempt Semantics

`CHECKLIST.md`, `JOURNAL.md`, `FINDINGS.md`, `ISSUES.md`, and `SUBMISSIONS.md` should render:

- current attempt slug
- readable run labels
- readable stage status summaries

The CLI `status` output should also pivot away from raw `stage_run_id` and show the readable artifact label/path.

## Files Expected To Change

- `kaggle_agent/schema.py`
- `kaggle_agent/control/store.py`
- `kaggle_agent/control/reporting.py`
- `kaggle_agent/decision/helpers.py`
- `kaggle_agent/cli.py`
- `kaggle_agent/service.py`
- `tests/test_workspace.py`

## Acceptance Criteria

- Running the destructive reset path leaves no old files under `artifacts/`, `state/`, `reports/`, or the root surface files
- The new workspace starts with the notebook-derived attempt slug
- New stage artifacts land under `artifacts/attempts/<attempt_slug>/runs/.../stages/...`
- Default initialization no longer seeds `workitem-perch-debug-smoke`
- Preflight can still be invoked explicitly
- CLI and surface files expose readable run and stage labels
