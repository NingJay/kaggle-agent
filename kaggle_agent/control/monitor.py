from __future__ import annotations

import time
from pathlib import Path

from kaggle_agent.control.executor import collect_finished_runs, start_run
from kaggle_agent.control.reporting import write_reports
from kaggle_agent.control.scheduler import choose_next_work_item, mark_work_item_status, register_work_item
from kaggle_agent.control.store import find_work_item, load_state, save_state
from kaggle_agent.control.submission import build_submission_candidate, plan_submission_slots
from kaggle_agent.decision.codegen import build_codegen
from kaggle_agent.decision.critic import build_critic
from kaggle_agent.decision.decider import build_decision
from kaggle_agent.decision.evidence import build_evidence
from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    fail_stage_run,
    latest_stage_payload,
    latest_stage_run,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.decision.planner import build_plan
from kaggle_agent.decision.reporter import build_report
from kaggle_agent.decision.research import build_research
from kaggle_agent.schema import SpecRecord, ValidationRecord, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import now_utc_iso, truncate, workspace_lock


FINAL_RUN_STATUSES = {"succeeded", "failed"}


def _validated_spec_for_dedupe(state: WorkspaceState, dedupe_key: str) -> SpecRecord | None:
    for spec in reversed(state.specs):
        if spec.dedupe_key == dedupe_key and spec.status == "validated":
            return spec
    return None


def _create_or_update_spec(
    config: WorkspaceConfig,
    state: WorkspaceState,
    run_id: str,
    stage_run_id: str,
    payload: dict[str, object],
) -> SpecRecord:
    existing = _validated_spec_for_dedupe(state, str(payload.get("dedupe_key", "")))
    if existing is not None:
        existing.config_path = str(payload["config_path"])
        existing.payload_path = str(Path(config.stage_dir("validate", stage_run_id)) / "validate.json")
        existing.updated_at = now_utc_iso()
        return existing
    spec = SpecRecord(
        spec_id=f"spec-{state.runtime.next_spec_number:04d}",
        work_item_id=f"derived:{run_id}",
        source_stage_run_id=stage_run_id,
        spec_type="experiment",
        title=str(payload["title"]),
        family=str(payload["family"]),
        config_path=str(payload["config_path"]),
        payload_path=str(Path(config.stage_dir("validate", stage_run_id)) / "validate.json"),
        launch_mode=str(payload.get("launch_mode", "background")),
        status="validated",
        dedupe_key=str(payload.get("dedupe_key", "")),
        created_at=now_utc_iso(),
        updated_at=now_utc_iso(),
    )
    state.runtime.next_spec_number += 1
    state.specs.append(spec)
    return spec


def _run_validate_stage(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    plan = latest_stage_payload(state, run_id, "plan")
    codegen = latest_stage_payload(state, run_id, "codegen")
    critic = latest_stage_payload(state, run_id, "critic")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="validate",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    write_input_manifest(
        input_manifest_path,
        {
            "run": run.to_dict(),
            "plan": plan,
            "codegen": codegen,
            "critic": critic,
        },
    )
    plan_status = str(plan.get("plan_status", "hold"))
    queued_work_item_id = ""
    spec_id = ""
    status = "not_required"
    summary = "No follow-up experiment required."
    if plan_status == "planned":
        config_path = str(codegen.get("generated_config_path") or plan.get("config_path") or "")
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = config.root / config_file
        if critic.get("status") != "approved":
            status = "failed"
            summary = "Critic rejected the generated bundle."
        elif not config_file.exists():
            status = "failed"
            summary = f"Missing config for validation: {config_file}"
        else:
            status = "validated"
            summary = f"Validated follow-up config at {config_file}"
            plan_payload = dict(plan)
            plan_payload["config_path"] = str(config_file.relative_to(config.root)) if config_file.is_relative_to(config.root) else str(config_file)
            spec = _create_or_update_spec(config, state, run_id, stage_run.stage_run_id, plan_payload)
            spec_id = spec.spec_id
            if config.automation.auto_execute_plans:
                queued = register_work_item(
                    state,
                    title=str(plan["title"]),
                    work_type=str(plan.get("work_type", "experiment_iteration")),
                    family=str(plan["family"]),
                    config_path=str(plan_payload["config_path"]),
                    priority=int(plan.get("priority", 50)),
                    pipeline=["execute", "evidence", "report", "research", "decision", "plan", "codegen", "critic", "validate", "submission"],
                    depends_on=[str(item) for item in plan.get("depends_on", [run.work_item_id])],
                    dedupe_key=str(plan.get("dedupe_key", "")),
                    source_run_id=run.run_id,
                    source_stage_run_id=stage_run.stage_run_id,
                    notes=[str(plan.get("hypothesis", ""))],
                )
                queued.latest_spec_id = spec.spec_id
                queued.updated_at = now_utc_iso()
                queued_work_item_id = queued.id
    payload = {
        "stage": "validate",
        "status": status,
        "summary": summary,
        "plan_status": plan_status,
        "spec_id": spec_id,
        "queued_work_item_id": queued_work_item_id,
    }
    markdown = stage_markdown(
        f"Validation {run_id}",
        [
            f"- Status: `{status}`",
            f"- Summary: {summary}",
            f"- Spec id: `{spec_id or 'n/a'}`",
            f"- Queued work item: `{queued_work_item_id or 'n/a'}`",
        ],
    )
    complete_stage_run(stage_run, payload=payload, markdown=markdown, validator_status=status)
    validation = ValidationRecord(
        validation_id=f"validation-{state.runtime.next_validation_number:04d}",
        work_item_id=run.work_item_id,
        source_stage_run_id=stage_run.stage_run_id,
        spec_id=spec_id,
        status=status,
        summary=summary,
        output_json_path=stage_run.output_json_path,
        output_md_path=stage_run.output_md_path,
        created_at=now_utc_iso(),
    )
    state.runtime.next_validation_number += 1
    state.validations.append(validation)
    return stage_run


def _run_submission_stage(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    decision = latest_stage_payload(state, run_id, "decision")
    plan = latest_stage_payload(state, run_id, "plan")
    validation = latest_stage_payload(state, run_id, "validate")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="submission",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    write_input_manifest(
        input_manifest_path,
        {
            "run": run.to_dict(),
            "decision": decision,
            "plan": plan,
            "validation": validation,
        },
    )
    should_build = (
        str(decision.get("submission_recommendation", "no")) == "candidate"
        or str(plan.get("plan_status", "")) == "submission_candidate"
    )
    if should_build:
        candidate = build_submission_candidate(config, state, run_id)
        slot_plan = plan_submission_slots(config, state)
        payload = {
            "stage": "submission",
            "status": "candidate_created",
            "candidate_id": candidate.id,
            "cpu_ready": candidate.cpu_ready,
            "slot_plan": slot_plan,
        }
        markdown = stage_markdown(
            f"Submission {run_id}",
            [
                "- Status: `candidate_created`",
                f"- Candidate id: `{candidate.id}`",
                f"- CPU ready: `{candidate.cpu_ready}`",
                f"- Remaining daily slots: {slot_plan['remaining_daily_slots']}",
                f"- Remaining final slots: {slot_plan['remaining_final_slots']}",
            ],
        )
    else:
        payload = {"stage": "submission", "status": "skipped", "reason": "No submission recommendation for this run."}
        markdown = stage_markdown(
            f"Submission {run_id}",
            ["- Status: `skipped`", f"- Reason: {payload['reason']}"],
        )
    complete_stage_run(stage_run, payload=payload, markdown=markdown)
    return stage_run


def _finalize_work_item_status(state: WorkspaceState, run_id: str) -> None:
    run = next(item for item in state.runs if item.run_id == run_id)
    work_item = find_work_item(state, run.work_item_id)
    decision = latest_stage_payload(state, run_id, "decision")
    if run.stage_error:
        work_item.status = "failed"
    elif run.status == "failed" and str(decision.get("decision_type", "")) == "blocked":
        work_item.status = "blocked"
    elif latest_stage_payload(state, run_id, "submission").get("status") == "candidate_created":
        work_item.status = "submitted"
    else:
        work_item.status = "complete" if run.status == "succeeded" else "failed"
    work_item.updated_at = now_utc_iso()


def _process_run_stage_chain(config: WorkspaceConfig, state: WorkspaceState, run_id: str) -> None:
    run = next(item for item in state.runs if item.run_id == run_id)
    while run.status in FINAL_RUN_STATUSES and run.stage_cursor and run.stage_cursor != "complete" and not run.stage_error:
        try:
            if run.stage_cursor == "evidence":
                build_evidence(config, state, run_id)
                run.stage_cursor = "report"
            elif run.stage_cursor == "report":
                build_report(config, state, run_id)
                run.stage_cursor = "research"
            elif run.stage_cursor == "research":
                build_research(config, state, run_id)
                run.stage_cursor = "decision"
            elif run.stage_cursor == "decision":
                build_decision(config, state, run_id)
                run.stage_cursor = "plan"
            elif run.stage_cursor == "plan":
                build_plan(config, state, run_id)
                run.stage_cursor = "codegen"
            elif run.stage_cursor == "codegen":
                build_codegen(config, state, run_id)
                run.stage_cursor = "critic"
            elif run.stage_cursor == "critic":
                build_critic(config, state, run_id)
                run.stage_cursor = "validate"
            elif run.stage_cursor == "validate":
                _run_validate_stage(config, state, run_id)
                run.stage_cursor = "submission"
            elif run.stage_cursor == "submission":
                _run_submission_stage(config, state, run_id)
                run.stage_cursor = "complete"
                _finalize_work_item_status(state, run_id)
            else:
                raise ValueError(f"Unknown stage cursor: {run.stage_cursor}")
            run.stage_updated_at = now_utc_iso()
        except Exception as error:  # noqa: BLE001
            latest = latest_stage_run(state, run_id)
            if latest is not None and latest.status == "running":
                fail_stage_run(latest, error)
            run.stage_error = truncate(str(error), limit=800)
            run.stage_updated_at = now_utc_iso()
            break


def process_completed_runs(config: WorkspaceConfig, state: WorkspaceState) -> None:
    finished_runs = collect_finished_runs(config, state)
    if finished_runs:
        for run in finished_runs:
            mark_work_item_status(state, run.work_item_id, "reviewing")
    for run in state.runs:
        if run.status not in FINAL_RUN_STATUSES:
            continue
        if run.stage_cursor == "complete":
            continue
        _process_run_stage_chain(config, state, run.run_id)


def maybe_start_next_run(
    config: WorkspaceConfig,
    state: WorkspaceState,
    *,
    background: bool | None = None,
) -> tuple[RunRecord | None, bool]:
    next_work_item = choose_next_work_item(config, state)
    if next_work_item is None:
        return None, False
    run_in_background = background if background is not None else next_work_item.work_type != "deep_dive_analysis"
    run = start_run(config, state, next_work_item.id, background=run_in_background)
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
