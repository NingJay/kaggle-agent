# Attempt Reset Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a destructive reset path, attempt-centric artifact naming, and explicit preflight flow for human-readable restartability.

**Architecture:** Keep the ledger and stage graph intact, but move artifact path generation and root-surface state around an explicit current attempt record. Canonical artifact directories become attempt-centric and human-readable, while preflight is separated from default seed work items.

**Tech Stack:** Python 3, dataclasses, sqlite-backed state, filesystem artifact layout, pytest

---

### Task 1: Lock The Reset And Seed Expectations In Tests

**Files:**
- Modify: `tests/test_workspace.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for hard reset and explicit preflight seeding**

Add assertions covering:
- force reset clears stale `artifacts/`, `state/exports`, `reports`, and root surface docs
- default work items no longer include `workitem-perch-debug-smoke`
- default queue still includes the real baseline work item
- explicit preflight creation remains possible

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `PYTHONPATH=. pytest -q tests/test_workspace.py -k "reset or preflight or init"`
Expected: FAIL because the current initializer preserves the old seeding/layout behavior.

- [ ] **Step 3: Implement minimal store/service changes**

Update workspace initialization and seed creation to support destructive reset and explicit preflight.

- [ ] **Step 4: Re-run targeted tests**

Run: `PYTHONPATH=. pytest -q tests/test_workspace.py -k "reset or preflight or init"`
Expected: PASS

### Task 2: Lock The Human-Readable Artifact Layout In Tests

**Files:**
- Modify: `tests/test_workspace.py`
- Modify: `kaggle_agent/decision/helpers.py`
- Modify: `kaggle_agent/schema.py`

- [ ] **Step 1: Write failing tests for attempt-centric artifact paths**

Add assertions that stage outputs are written under:
- `artifacts/attempts/<attempt_slug>/runs/<run_slug>/stages/<ordered-stage-status>/`

Also assert that the attempt slug is `simplerun-perch-v2embedprobe-bayesian-0-912`.

- [ ] **Step 2: Run targeted test to verify failure**

Run: `PYTHONPATH=. pytest -q tests/test_workspace.py -k artifact`
Expected: FAIL because outputs still use `artifacts/<stage>/stage-0004-...`.

- [ ] **Step 3: Implement path helpers and readable labels**

Update state/config/path helpers so stage output directories and related metadata use the new canonical layout while preserving stable ids in JSON payloads.

- [ ] **Step 4: Re-run targeted test**

Run: `PYTHONPATH=. pytest -q tests/test_workspace.py -k artifact`
Expected: PASS

### Task 3: Update Surface Files And CLI Status

**Files:**
- Modify: `kaggle_agent/control/reporting.py`
- Modify: `kaggle_agent/cli.py`
- Test: `tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for surface readability**

Add assertions that the rendered surfaces/status output include:
- current attempt slug
- readable run labels
- readable stage labels instead of only raw `stage_run_id`

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `PYTHONPATH=. pytest -q tests/test_workspace.py -k "status or journal or checklist"`
Expected: FAIL because current output is machine-centric.

- [ ] **Step 3: Implement surface rendering changes**

Update the markdown surfaces and CLI status printer to foreground attempt/run/stage labels and paths.

- [ ] **Step 4: Re-run targeted tests**

Run: `PYTHONPATH=. pytest -q tests/test_workspace.py -k "status or journal or checklist"`
Expected: PASS

### Task 4: Run Full Regression And Push

**Files:**
- Modify: `docs/superpowers/specs/2026-03-27-attempt-reset-surface-design.md`
- Modify: `docs/superpowers/plans/2026-03-27-attempt-reset-surface.md`

- [ ] **Step 1: Run full test suite**

Run: `PYTHONPATH=. pytest -q`
Expected: PASS with current skipped live-adapter tests unchanged.

- [ ] **Step 2: Inspect git diff**

Run: `git status --short && git diff --stat`
Expected: only intended reset/layout/surface changes.

- [ ] **Step 3: Commit**

Run: `git add docs/superpowers/specs/2026-03-27-attempt-reset-surface-design.md docs/superpowers/plans/2026-03-27-attempt-reset-surface.md kaggle_agent tests && git commit -m "feat: reset attempts and humanize artifact surfaces"`
Expected: commit created.

- [ ] **Step 4: Push**

Run: `git push -u origin feat/attempt-reset-surface`
Expected: remote branch created.
