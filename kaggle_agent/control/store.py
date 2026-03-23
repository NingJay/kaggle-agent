from __future__ import annotations

import shutil
from pathlib import Path

from kaggle_agent.schema import (
    DecisionRecord,
    ExperimentSpec,
    RunRecord,
    RuntimeState,
    SubmissionCandidate,
    WorkspaceConfig,
    WorkspaceState,
)
from kaggle_agent.utils import atomic_write_json, ensure_directory, now_utc_iso, read_json

EXPERIMENTS_FILE = "experiments.json"
RUNS_FILE = "runs.json"
DECISIONS_FILE = "decisions.json"
SUBMISSIONS_FILE = "submission_candidates.json"
RUNTIME_FILE = "runtime.json"
FINAL_RUN_STATUSES = {"succeeded", "failed"}


def _default_experiments(config: WorkspaceConfig) -> list[ExperimentSpec]:
    timestamp = now_utc_iso()
    debug_config = config.runtime_root() / "configs" / "debug.yaml"
    default_config = config.runtime_root() / "configs" / "default.yaml"
    return [
        ExperimentSpec(
            id="exp-perch-debug-smoke",
            title="Perch-head debug smoke",
            hypothesis="Verify the new runtime, decision loop, and dual-metric reporting on a tiny debug run.",
            family="perch_head_debug",
            config_path=str(debug_config.relative_to(config.root)),
            priority=10,
            tags=["debug", "perch-head", "smoke"],
            created_at=timestamp,
            updated_at=timestamp,
        ),
        ExperimentSpec(
            id="exp-perch-baseline",
            title="Perch cached-probe baseline",
            hypothesis="Run the first real cached Perch embedding probe after the debug smoke passes.",
            family="perch_cached_probe",
            config_path=str(default_config.relative_to(config.root)),
            priority=20,
            depends_on=["exp-perch-debug-smoke"],
            tags=["baseline", "perch", "cached-probe"],
            created_at=timestamp,
            updated_at=timestamp,
        ),
    ]


def _default_runtime_state() -> RuntimeState:
    timestamp = now_utc_iso()
    return RuntimeState(initialized_at=timestamp, last_tick_at=timestamp, last_report_at=timestamp)


def ensure_layout(config: WorkspaceConfig) -> None:
    ensure_directory(config.root / config.paths.state_dir)
    ensure_directory(config.artifact_root())
    ensure_directory(config.legacy_root())
    ensure_directory(config.knowledge_root())
    for category in [
        "runs",
        "reports",
        "decision_briefs",
        "research",
        "decisions",
        "plans",
        "submissions",
        "logs",
    ]:
        ensure_directory(config.artifact_path(category))
    for category in ["research", "papers"]:
        ensure_directory(config.knowledge_path(category))
    ensure_directory(config.generated_config_root())


def archive_legacy_artifacts(config: WorkspaceConfig) -> Path | None:
    legacy_candidates = [
        "knowledge",
        "prompts",
        "reports",
        "logs",
        "generated_configs",
        "runtime",
        "submissions",
        "state",
        "artifacts",
    ]
    existing = [config.root / name for name in legacy_candidates if (config.root / name).exists()]
    if not existing:
        return None
    timestamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    archive_root = ensure_directory(config.legacy_root() / timestamp)
    for path in existing:
        shutil.move(str(path), str(archive_root / path.name))
    return archive_root


def _submission_dedupe_key(run_id: str) -> str:
    return f"submission_candidate:{run_id}"


def _infer_post_run_stage(
    run: RunRecord,
    decisions_by_run_id: dict[str, DecisionRecord],
    submission_run_ids: set[str],
) -> str:
    if run.status not in FINAL_RUN_STATUSES:
        return ""
    if run.post_run_stage:
        return run.post_run_stage
    if not run.decision_record_path:
        return "pending"
    if not run.plan_path:
        return "decision_done"
    decision = decisions_by_run_id.get(run.run_id)
    if decision and decision.submission_recommendation == "candidate":
        return "complete" if run.run_id in submission_run_ids else "apply_done"
    return "plan_done"


def _normalize_state(state: WorkspaceState) -> WorkspaceState:
    for submission in state.submissions:
        if not submission.dedupe_key:
            submission.dedupe_key = _submission_dedupe_key(submission.source_run_id)
    for experiment in state.experiments:
        if experiment.source_decision_id and not experiment.dedupe_key:
            experiment.dedupe_key = f"decision:{experiment.source_decision_id}:experiment"
    decisions_by_run_id = {item.source_run_id: item for item in state.decisions}
    submission_run_ids = {item.source_run_id for item in state.submissions}
    timestamp = now_utc_iso()
    for run in state.runs:
        inferred_stage = _infer_post_run_stage(run, decisions_by_run_id, submission_run_ids)
        if inferred_stage and not run.post_run_stage:
            run.post_run_stage = inferred_stage
        if run.post_run_stage and not run.post_run_updated_at:
            run.post_run_updated_at = run.completed_at or timestamp
    return state


def load_state(config: WorkspaceConfig) -> WorkspaceState:
    experiments_raw = read_json(config.state_path(EXPERIMENTS_FILE), [])
    runs_raw = read_json(config.state_path(RUNS_FILE), [])
    decisions_raw = read_json(config.state_path(DECISIONS_FILE), [])
    submissions_raw = read_json(config.state_path(SUBMISSIONS_FILE), [])
    runtime_raw = read_json(config.state_path(RUNTIME_FILE), None)
    experiments = [ExperimentSpec.from_dict(item) for item in experiments_raw]
    runs = [RunRecord.from_dict(item) for item in runs_raw]
    decisions = [DecisionRecord.from_dict(item) for item in decisions_raw]
    submissions = [SubmissionCandidate.from_dict(item) for item in submissions_raw]
    runtime = RuntimeState.from_dict(runtime_raw) if runtime_raw else _default_runtime_state()
    state = WorkspaceState(
        experiments=experiments,
        runs=runs,
        decisions=decisions,
        submissions=submissions,
        runtime=runtime,
    )
    return _normalize_state(state)


def save_state(config: WorkspaceConfig, state: WorkspaceState) -> None:
    atomic_write_json(config.state_path(EXPERIMENTS_FILE), [item.to_dict() for item in state.experiments])
    atomic_write_json(config.state_path(RUNS_FILE), [item.to_dict() for item in state.runs])
    atomic_write_json(config.state_path(DECISIONS_FILE), [item.to_dict() for item in state.decisions])
    atomic_write_json(config.state_path(SUBMISSIONS_FILE), [item.to_dict() for item in state.submissions])
    atomic_write_json(config.state_path(RUNTIME_FILE), state.runtime.to_dict())


def initialize_workspace(config: WorkspaceConfig, archive_legacy: bool = True, force: bool = False) -> WorkspaceState:
    if archive_legacy:
        archive_legacy_artifacts(config)
    ensure_layout(config)
    has_existing_state = all(config.state_path(name).exists() for name in [EXPERIMENTS_FILE, RUNS_FILE, DECISIONS_FILE, SUBMISSIONS_FILE, RUNTIME_FILE])
    if has_existing_state and not force:
        return load_state(config)
    state = WorkspaceState(
        experiments=_default_experiments(config),
        runs=[],
        decisions=[],
        submissions=[],
        runtime=_default_runtime_state(),
    )
    save_state(config, state)
    return state


def find_experiment(state: WorkspaceState, experiment_id: str) -> ExperimentSpec:
    for item in state.experiments:
        if item.id == experiment_id:
            return item
    raise KeyError(f"Unknown experiment: {experiment_id}")


def find_run(state: WorkspaceState, run_id: str) -> RunRecord:
    for item in state.runs:
        if item.run_id == run_id:
            return item
    raise KeyError(f"Unknown run: {run_id}")
