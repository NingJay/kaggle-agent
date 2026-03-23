from __future__ import annotations

import time
from pathlib import Path

from kaggle_agent.control.executor import collect_finished_runs, start_run
from kaggle_agent.control.reporting import write_reports
from kaggle_agent.control.scheduler import choose_next_experiment, register_experiment
from kaggle_agent.control.store import load_state, save_state
from kaggle_agent.control.submission import build_submission_candidate
from kaggle_agent.decision.decider import make_decision
from kaggle_agent.decision.evidence import build_decision_brief
from kaggle_agent.decision.planner import build_plan
from kaggle_agent.decision.research import build_research_summary
from kaggle_agent.schema import DecisionRecord, ExperimentSpec, RunRecord, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import now_utc_iso, read_json, truncate, workspace_lock

FINAL_RUN_STATUSES = {"succeeded", "failed"}


def _mark_post_run_stage(run: RunRecord, stage: str, *, error: str = "") -> None:
    run.post_run_stage = stage
    run.post_run_error = error
    run.post_run_updated_at = now_utc_iso()


def _record_post_run_error(config: WorkspaceConfig, state: WorkspaceState, run: RunRecord, error: Exception) -> None:
    run.post_run_error = truncate(str(error), limit=500)
    run.post_run_updated_at = now_utc_iso()
    save_state(config, state)


def _decision_for_run(state: WorkspaceState, run_id: str) -> DecisionRecord | None:
    for decision in reversed(state.decisions):
        if decision.source_run_id == run_id:
            return decision
    return None


def _require_decision_for_run(state: WorkspaceState, run_id: str) -> DecisionRecord:
    decision = _decision_for_run(state, run_id)
    if decision is None:
        raise ValueError(f"Missing decision record for {run_id}")
    return decision


def _load_plan_payload(run: RunRecord) -> dict[str, object]:
    if not run.plan_path:
        raise ValueError(f"Missing plan path for {run.run_id}")
    payload = read_json(Path(run.plan_path), None)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid plan payload for {run.run_id}")
    return payload


def _planned_experiment_dedupe_key(decision: DecisionRecord) -> str:
    return f"decision:{decision.decision_id}:experiment"


def _find_existing_planned_experiment(
    state: WorkspaceState,
    decision: DecisionRecord,
    payload: dict[str, object],
) -> ExperimentSpec | None:
    dedupe_key = str(payload.get("dedupe_key") or _planned_experiment_dedupe_key(decision))
    for experiment in state.experiments:
        if experiment.dedupe_key == dedupe_key:
            return experiment
    depends_on = [str(item) for item in payload.get("depends_on", [decision.experiment_id])]
    title = str(payload.get("title", ""))
    family = str(payload.get("family", ""))
    config_path = str(payload.get("config_path", ""))
    for experiment in state.experiments:
        if (
            experiment.title == title
            and experiment.family == family
            and experiment.config_path == config_path
            and experiment.depends_on == depends_on
        ):
            experiment.dedupe_key = dedupe_key
            if not experiment.source_decision_id:
                experiment.source_decision_id = decision.decision_id
            experiment.updated_at = now_utc_iso()
            return experiment
    return None


def _apply_plan_payload(config: WorkspaceConfig, state: WorkspaceState, run: RunRecord) -> None:
    decision = _require_decision_for_run(state, run.run_id)
    payload = _load_plan_payload(run)
    status = str(payload.get("status", ""))
    if status in {"hold", "submission_candidate"}:
        return
    if status != "planned":
        raise ValueError(f"Unsupported plan status for {run.run_id}: {status}")
    existing = _find_existing_planned_experiment(state, decision, payload)
    if existing is not None:
        return
    register_experiment(
        state,
        title=str(payload["title"]),
        hypothesis=str(payload.get("hypothesis") or f"Follow-up planned from {decision.decision_id}: {decision.why}"),
        family=str(payload["family"]),
        config_path=str(payload["config_path"]),
        priority=int(payload.get("priority", 50)),
        depends_on=[str(item) for item in payload.get("depends_on", [decision.experiment_id])],
        tags=[str(item) for item in payload.get("tags", [])],
        launch_mode=str(payload.get("launch_mode", "background")),
        dedupe_key=str(payload.get("dedupe_key") or _planned_experiment_dedupe_key(decision)),
        source_decision_id=decision.decision_id,
        requeue_existing=False,
    )


def _process_run_post_completion(config: WorkspaceConfig, state: WorkspaceState, run: RunRecord) -> None:
    while run.status in FINAL_RUN_STATUSES and run.post_run_stage != "complete":
        try:
            if not run.post_run_stage:
                _mark_post_run_stage(run, "pending")
            if run.post_run_stage == "pending":
                build_decision_brief(config, state, run.run_id)
                _mark_post_run_stage(run, "evidence_done")
            elif run.post_run_stage == "evidence_done":
                build_research_summary(config, state, run.run_id)
                _mark_post_run_stage(run, "research_done")
            elif run.post_run_stage == "research_done":
                make_decision(config, state, run.run_id)
                _mark_post_run_stage(run, "decision_done")
            elif run.post_run_stage == "decision_done":
                decision = _require_decision_for_run(state, run.run_id)
                build_plan(config, state, decision)
                _mark_post_run_stage(run, "plan_done")
            elif run.post_run_stage == "plan_done":
                _apply_plan_payload(config, state, run)
                _mark_post_run_stage(run, "apply_done")
            elif run.post_run_stage == "apply_done":
                decision = _require_decision_for_run(state, run.run_id)
                if decision.submission_recommendation == "candidate":
                    build_submission_candidate(config, state, run.run_id)
                    _mark_post_run_stage(run, "submission_done")
                else:
                    _mark_post_run_stage(run, "complete")
            elif run.post_run_stage == "submission_done":
                _mark_post_run_stage(run, "complete")
            else:
                raise ValueError(f"Unknown post-run stage for {run.run_id}: {run.post_run_stage}")
            save_state(config, state)
        except Exception as error:
            _record_post_run_error(config, state, run, error)
            break


def process_completed_runs(config: WorkspaceConfig, state: WorkspaceState) -> None:
    finished_runs = collect_finished_runs(config, state)
    if finished_runs:
        save_state(config, state)
    for run in state.runs:
        if run.status not in FINAL_RUN_STATUSES:
            continue
        if run.post_run_stage == "complete":
            continue
        _process_run_post_completion(config, state, run)


def maybe_start_next_run(
    config: WorkspaceConfig,
    state: WorkspaceState,
    *,
    background: bool | None = None,
) -> tuple[RunRecord | None, bool]:
    next_experiment = choose_next_experiment(config, state)
    if next_experiment is None:
        return None, False
    run_in_background = background if background is not None else next_experiment.launch_mode != "sync"
    run = start_run(config, state, next_experiment.id, background=run_in_background)
    return run, run_in_background


def _tick_workspace_once(config: WorkspaceConfig, *, auto_start: bool = True) -> WorkspaceState:
    state = load_state(config)
    state.runtime.last_tick_at = now_utc_iso()
    process_completed_runs(config, state)
    if auto_start and not state.runtime.active_run_ids:
        run, run_in_background = maybe_start_next_run(config, state)
        if run is not None and not run_in_background:
            process_completed_runs(config, state)
    write_reports(config, state)
    save_state(config, state)
    return state


def tick_workspace(config: WorkspaceConfig, *, auto_start: bool = True) -> WorkspaceState:
    with workspace_lock(config.lock_path()):
        return _tick_workspace_once(config, auto_start=auto_start)


def watch_workspace(config: WorkspaceConfig, *, interval_seconds: int, iterations: int, auto_start: bool = True) -> None:
    total = iterations if iterations > 0 else 1_000_000_000
    for index in range(total):
        with workspace_lock(config.lock_path()):
            _tick_workspace_once(config, auto_start=auto_start)
        if index == total - 1:
            break
        time.sleep(interval_seconds)
