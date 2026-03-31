from __future__ import annotations

import json
import time
from pathlib import Path

from kaggle_agent.adapters.command import CommandAdapterError
from kaggle_agent.control.executor import collect_finished_runs, reconcile_active_run_ids, start_run
from kaggle_agent.control.reporting import write_reports
from kaggle_agent.control.scheduler import choose_next_work_item, choose_next_work_items, mark_work_item_status, register_work_item
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
from kaggle_agent.knowledge import synchronize_branch_memory
from kaggle_agent.schema import SpecRecord, ValidationRecord, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import now_utc_iso, truncate, workspace_lock


FINAL_RUN_STATUSES = {"succeeded", "failed"}
CODEGEN_CRITIC_MAX_REPAIR_CYCLES = 4
AUTO_RETRY_STAGE_NAMES = {"report", "research", "decision", "plan", "codegen", "critic"}
AUTO_RETRY_ERROR_MARKERS = (
    "timed out",
    "timeout",
    "temporarily unavailable",
    "connection reset",
    "broken pipe",
)


def _submission_branch_origin(work_item) -> str:
    if work_item is None:
        return "frontier"
    if work_item.branch_role == "support":
        return "support"
    if not work_item.source_run_id and not work_item.parent_work_item_id:
        return "baseline"
    return "frontier"


def _validated_spec_for_dedupe(state: WorkspaceState, dedupe_key: str) -> SpecRecord | None:
    for spec in reversed(state.specs):
        if spec.dedupe_key == dedupe_key and spec.status == "validated":
            return spec
    return None


def _completed_stage_runs_count(state: WorkspaceState, run_id: str, stage_name: str) -> int:
    return sum(
        1
        for stage_run in state.stage_runs
        if stage_run.run_id == run_id and stage_run.stage_name == stage_name and stage_run.status in {"completed", "failed"}
    )


def _should_retry_codegen_after_critic_reject(state: WorkspaceState, run_id: str) -> bool:
    plan = latest_stage_payload(state, run_id, "plan")
    critic = latest_stage_payload(state, run_id, "critic")
    if str(plan.get("plan_status", "")) != "planned":
        return False
    if str(critic.get("status", "")) != "rejected":
        return False
    return _completed_stage_runs_count(state, run_id, "codegen") < CODEGEN_CRITIC_MAX_REPAIR_CYCLES


def _should_auto_retry_stage_error(run) -> bool:
    if not run.stage_cursor or run.stage_cursor not in AUTO_RETRY_STAGE_NAMES:
        return False
    message = str(run.stage_error or "").lower()
    return any(marker in message for marker in AUTO_RETRY_ERROR_MARKERS)


def _should_auto_retry_stage_exception(stage_run, error: Exception) -> bool:
    if stage_run is None:
        return False
    if stage_run.stage_name not in AUTO_RETRY_STAGE_NAMES:
        return False
    if not isinstance(error, CommandAdapterError):
        return False
    message = str(error).lower()
    return any(marker in message for marker in AUTO_RETRY_ERROR_MARKERS)


def _create_or_update_spec(
    state: WorkspaceState,
    stage_run,
    payload: dict[str, object],
    *,
    work_item_id: str,
) -> SpecRecord:
    existing = _validated_spec_for_dedupe(state, str(payload.get("dedupe_key", "")))
    if existing is not None:
        existing.work_item_id = work_item_id
        existing.config_path = str(payload["config_path"])
        existing.payload_path = stage_run.output_json_path
        existing.code_state_ref = str(payload.get("code_state_ref", ""))
        existing.portfolio_id = str(payload.get("portfolio_id", "") or existing.portfolio_id)
        existing.idea_class = str(payload.get("idea_class", "") or existing.idea_class)
        existing.branch_role = str(payload.get("branch_role", "") or existing.branch_role)
        existing.branch_rank = int(payload.get("branch_rank", existing.branch_rank))
        existing.knowledge_card_ids = [str(item) for item in payload.get("knowledge_card_ids", [])] or existing.knowledge_card_ids
        existing.updated_at = now_utc_iso()
        return existing
    spec = SpecRecord(
        spec_id=f"spec-{state.runtime.next_spec_number:04d}",
        work_item_id=work_item_id,
        source_stage_run_id=stage_run.stage_run_id,
        spec_type="experiment",
        title=str(payload["title"]),
        family=str(payload["family"]),
        config_path=str(payload["config_path"]),
        payload_path=stage_run.output_json_path,
        launch_mode=str(payload.get("launch_mode", "background")),
        code_state_ref=str(payload.get("code_state_ref", "")),
        status="validated",
        dedupe_key=str(payload.get("dedupe_key", "")),
        portfolio_id=str(payload.get("portfolio_id", "")),
        idea_class=str(payload.get("idea_class", "")),
        branch_role=str(payload.get("branch_role", "")),
        branch_rank=int(payload.get("branch_rank", 0) or 0),
        knowledge_card_ids=[str(item) for item in payload.get("knowledge_card_ids", [])],
        created_at=now_utc_iso(),
        updated_at=now_utc_iso(),
    )
    state.runtime.next_spec_number += 1
    state.specs.append(spec)
    return spec


def _plan_branches(plan: dict[str, object], run) -> list[dict[str, object]]:
    branches = plan.get("branch_plans")
    if isinstance(branches, list):
        normalized = [dict(item) for item in branches if isinstance(item, dict)]
        if normalized:
            return normalized
    if str(plan.get("plan_status", "")) != "planned":
        return []
    return [
        {
            "title": str(plan.get("title", "")),
            "family": str(plan.get("family", "")),
            "hypothesis": str(plan.get("hypothesis", "")),
            "reason": str(plan.get("reason", "")),
            "config_path": str(plan.get("config_path", "")),
            "priority": int(plan.get("priority", 50) or 50),
            "depends_on": [str(item) for item in plan.get("depends_on", [run.work_item_id])],
            "tags": [str(item) for item in plan.get("tags", [])],
            "launch_mode": str(plan.get("launch_mode", "background")),
            "dedupe_key": str(plan.get("dedupe_key", "")),
            "work_type": str(plan.get("work_type", "experiment_iteration")),
            "portfolio_id": str(plan.get("portfolio_id", "")),
            "idea_class": str(plan.get("idea_class", "")),
            "branch_role": str(plan.get("branch_role", "primary")),
            "branch_rank": int(plan.get("branch_rank", 0) or 0),
            "knowledge_card_ids": [str(item) for item in plan.get("knowledge_card_ids", [])],
            "policy_trace": [str(item) for item in plan.get("policy_trace", [])],
            "branch_memory_ids": [str(item) for item in plan.get("branch_memory_ids", [])],
            "scheduler_hints": dict(plan.get("scheduler_hints", {})) if isinstance(plan.get("scheduler_hints"), dict) else {},
        }
    ]


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
    dispatch_summary: list[dict[str, object]] = []
    portfolio_id = str(plan.get("portfolio_id", "") or "")
    if plan_status == "planned":
        branches = _plan_branches(plan, run)
        code_state_ref = str(codegen.get("code_state_ref", "") or "")
        code_state_path = Path(code_state_ref) if code_state_ref else None
        verify_status = str(codegen.get("verify_status") or codegen.get("smoke_status") or "skipped")
        verify_summary = str(codegen.get("verify_summary") or codegen.get("smoke_summary") or "")
        run_bundle_value = str(codegen.get("run_bundle_path", "") or "")
        run_bundle_path = Path(run_bundle_value) if run_bundle_value else None
        if run_bundle_path is not None and run_bundle_path.exists():
            run_bundle = json.loads(run_bundle_path.read_text(encoding="utf-8"))
            run_bundle_config_path = str(run_bundle.get("config_path") or "")
            if run_bundle_config_path:
                for branch in branches[:1]:
                    branch["config_path"] = run_bundle_config_path
        if critic.get("status") != "approved":
            status = "failed"
            summary = "Critic rejected the generated bundle."
        elif code_state_path is not None and not code_state_path.exists():
            status = "failed"
            summary = f"Missing code state for validation: {code_state_path}"
        elif code_state_path is not None and verify_status != "passed":
            status = "failed"
            summary = f"Codegen verify failed: {verify_summary or verify_status}"
        elif not branches:
            status = "failed"
            summary = "Plan did not materialize any executable follow-up branches."
        else:
            resolved_branches: list[dict[str, object]] = []
            missing_config: Path | None = None
            for branch in branches:
                config_path_value = str(branch.get("config_path", "") or "")
                config_file = Path(config_path_value)
                if not config_file.is_absolute():
                    config_file = config.root / config_file
                if not config_file.exists():
                    missing_config = config_file
                    break
                branch_payload = dict(branch)
                branch_payload["config_path"] = (
                    str(config_file.relative_to(config.root)) if config_file.is_relative_to(config.root) else str(config_file)
                )
                branch_payload["code_state_ref"] = code_state_ref
                resolved_branches.append(branch_payload)
            if missing_config is not None:
                status = "failed"
                summary = f"Missing config for validation: {missing_config}"
            else:
                status = "validated"
                spec_ids: list[str] = []
                queued_work_item_ids: list[str] = []
                for index, branch_payload in enumerate(resolved_branches):
                    if config.automation.auto_execute_plans:
                        queued = register_work_item(
                            state,
                            title=str(branch_payload["title"]),
                            work_type=str(branch_payload.get("work_type", "experiment_iteration")),
                            family=str(branch_payload["family"]),
                            config_path=str(branch_payload["config_path"]),
                            priority=int(branch_payload.get("priority", 50) or 50),
                            pipeline=["execute", "evidence", "report", "research", "decision", "plan", "codegen", "critic", "validate", "submission"],
                            depends_on=[str(item) for item in branch_payload.get("depends_on", [run.work_item_id])],
                            dedupe_key=str(branch_payload.get("dedupe_key", "")),
                            source_run_id=run.run_id,
                            source_stage_run_id=stage_run.stage_run_id,
                            portfolio_id=str(branch_payload.get("portfolio_id", "") or plan.get("portfolio_id", "")),
                            parent_work_item_id=run.work_item_id,
                            idea_class=str(branch_payload.get("idea_class", "")),
                            branch_role=str(branch_payload.get("branch_role", "primary")),
                            branch_rank=int(branch_payload.get("branch_rank", index) or index),
                            knowledge_card_ids=[str(item) for item in branch_payload.get("knowledge_card_ids", [])],
                            notes=[str(branch_payload.get("hypothesis", "")), str(branch_payload.get("reason", ""))],
                            policy_trace=[str(item) for item in branch_payload.get("policy_trace", [])],
                            branch_memory_ids=[str(item) for item in branch_payload.get("branch_memory_ids", [])],
                            scheduler_hints=dict(branch_payload.get("scheduler_hints", {})) if isinstance(branch_payload.get("scheduler_hints"), dict) else {},
                        )
                        queued.updated_at = now_utc_iso()
                        work_item_id = queued.id
                    else:
                        queued = None
                        work_item_id = f"derived:{stage_run.stage_run_id}:{index:02d}"
                    spec = _create_or_update_spec(state, stage_run, branch_payload, work_item_id=work_item_id)
                    spec_ids.append(spec.spec_id)
                    if queued is not None:
                        queued.latest_spec_id = spec.spec_id
                        queued.updated_at = now_utc_iso()
                        queued_work_item_ids.append(queued.id)
                    dispatch_summary.append(
                        {
                            "branch_rank": int(branch_payload.get("branch_rank", index) or index),
                            "branch_role": str(branch_payload.get("branch_role", "primary")),
                            "idea_class": str(branch_payload.get("idea_class", "")),
                            "portfolio_id": str(branch_payload.get("portfolio_id", "") or portfolio_id),
                            "work_item_id": work_item_id,
                            "spec_id": spec.spec_id,
                            "config_path": str(branch_payload.get("config_path", "")),
                            "knowledge_card_ids": [str(item) for item in branch_payload.get("knowledge_card_ids", [])],
                            "policy_trace": [str(item) for item in branch_payload.get("policy_trace", [])],
                            "branch_memory_ids": [str(item) for item in branch_payload.get("branch_memory_ids", [])],
                            "scheduler_hints": dict(branch_payload.get("scheduler_hints", {})) if isinstance(branch_payload.get("scheduler_hints"), dict) else {},
                        }
                    )
                    if index == 0:
                        spec_id = spec.spec_id
                        queued_work_item_id = queued.id if queued is not None else ""
                summary = (
                    f"Validated {len(resolved_branches)} follow-up branches and queued {len(queued_work_item_ids)} work items."
                    if config.automation.auto_execute_plans
                    else f"Validated {len(resolved_branches)} follow-up branches."
                )
    payload = {
        "stage": "validate",
        "status": status,
        "summary": summary,
        "plan_status": plan_status,
        "spec_id": spec_id,
        "queued_work_item_id": queued_work_item_id,
        "branch_count": len(_plan_branches(plan, run)) if plan_status == "planned" else 0,
        "portfolio_id": portfolio_id,
        "dispatch_summary": dispatch_summary,
    }
    markdown = stage_markdown(
        f"Validation {run_id}",
        [
            f"- Status: `{status}`",
            f"- Summary: {summary}",
            f"- Spec id: `{spec_id or 'n/a'}`",
            f"- Queued work item: `{queued_work_item_id or 'n/a'}`",
            f"- Branch count: {payload['branch_count']}",
            *(["", "## Dispatch Summary"] if dispatch_summary else []),
            *(
                f"- rank={item.get('branch_rank', 0)} | role={item.get('branch_role', '')} | idea={item.get('idea_class', '')} | work_item=`{item.get('work_item_id', '')}` | spec=`{item.get('spec_id', '')}`"
                for item in dispatch_summary
            ),
        ],
    )
    complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown, validator_status=status)
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
    work_item = next((item for item in state.work_items if item.id == run.work_item_id), None)
    branch_origin = _submission_branch_origin(work_item)
    if should_build:
        candidate = build_submission_candidate(config, state, run_id)
        slot_plan = plan_submission_slots(config, state)
        payload = {
            "stage": "submission",
            "status": "candidate_created",
            "candidate_id": candidate.id,
            "cpu_ready": candidate.cpu_ready,
            "slot_plan": slot_plan,
            "branch_origin": branch_origin,
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
        payload["branch_origin"] = branch_origin
        markdown = stage_markdown(
            f"Submission {run_id}",
            ["- Status: `skipped`", f"- Reason: {payload['reason']}", f"- Branch origin: `{branch_origin}`"],
        )
    complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
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
    while run.status in FINAL_RUN_STATUSES and run.stage_cursor and run.stage_cursor != "complete":
        if run.stage_error:
            if _should_auto_retry_stage_error(run):
                run.stage_error = ""
            else:
                break
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
                if _should_retry_codegen_after_critic_reject(state, run_id):
                    run.stage_cursor = "codegen"
                    run.stage_updated_at = now_utc_iso()
                    break
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
                fail_stage_run(latest, error, state=state)
            if _should_auto_retry_stage_exception(latest, error):
                run.stage_error = ""
                run.stage_updated_at = now_utc_iso()
                break
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
        if run.stage_cursor != "complete":
            _process_run_stage_chain(config, state, run.run_id)
        if run.stage_cursor == "complete" or run.stage_error:
            synchronize_branch_memory(state, run.run_id)


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


def _auto_start_ready_runs(config: WorkspaceConfig, state: WorkspaceState) -> list[str]:
    started: list[str] = []
    for work_item in choose_next_work_items(config, state):
        run = start_run(config, state, work_item.id, background=True)
        started.append(run.run_id)
    return started


def _tick_workspace_once(config: WorkspaceConfig, *, auto_start: bool = True) -> WorkspaceState:
    state = load_state(config)
    reconcile_active_run_ids(state)
    state.runtime.last_tick_at = now_utc_iso()
    process_completed_runs(config, state)
    reconcile_active_run_ids(state)
    if auto_start and config.automation.auto_start_planned_runs:
        _auto_start_ready_runs(config, state)
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
