from __future__ import annotations

from pathlib import Path

from kaggle_agent.branch_typing import compare_typings, envelope_violations
from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    latest_stage_payload,
    run_configured_stage_adapter,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.knowledge_reducer import active_search_envelope
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState


def _primary_branch(plan: dict[str, object]) -> dict[str, object]:
    branches = plan.get("branch_plans")
    if isinstance(branches, list):
        for item in branches:
            if isinstance(item, dict):
                return dict(item)
    return dict(plan)


def _apply_typing_contract(
    state: WorkspaceState,
    run_id: str,
    plan: dict[str, object],
    codegen: dict[str, object],
    payload: dict[str, object],
) -> dict[str, object]:
    branch = _primary_branch(plan)
    proposal_typing = (
        dict(codegen.get("proposal_typing", {}))
        if isinstance(codegen.get("proposal_typing"), dict)
        else (dict(branch.get("proposal_typing", {})) if isinstance(branch.get("proposal_typing"), dict) else {})
    )
    realized_typing = dict(codegen.get("realized_typing", {})) if isinstance(codegen.get("realized_typing"), dict) else {}
    info_gain_estimate = (
        dict(codegen.get("info_gain_estimate", {}))
        if isinstance(codegen.get("info_gain_estimate"), dict)
        else (dict(branch.get("info_gain_estimate", {})) if isinstance(branch.get("info_gain_estimate"), dict) else {})
    )
    family = str(branch.get("family") or plan.get("family") or "")
    search_envelope = dict(plan.get("search_envelope", {})) if isinstance(plan.get("search_envelope"), dict) else {}
    if not search_envelope:
        envelope_record = active_search_envelope(state, run_id=run_id, family=family)
        if envelope_record is not None and isinstance(envelope_record.envelope, dict):
            search_envelope = dict(envelope_record.envelope)
    drift = compare_typings(proposal_typing, realized_typing)
    violations = envelope_violations(
        search_envelope,
        proposal_typing=proposal_typing,
        realized_typing=realized_typing,
        info_gain_estimate=info_gain_estimate,
        override_reason=str(branch.get("override_reason", "") or ""),
    )
    concerns = [str(item) for item in payload.get("concerns", [])]
    warnings = [str(item) for item in payload.get("warnings", [])]
    required_fixes = [str(item) for item in payload.get("required_fixes", [])]
    reusable_judgments = [str(item) for item in payload.get("reusable_judgments", [])]
    status = str(payload.get("status", "approved") or "approved")
    if drift.get("severe"):
        status = "rejected"
        concerns.extend(str(item) for item in drift.get("summary", []))
        required_fixes.append("Align the realized branch with the proposed typing before validate-stage fan-out.")
    elif drift.get("drifted"):
        warnings.extend(str(item) for item in drift.get("summary", []))
    if violations:
        status = "rejected"
        concerns.extend(f"Envelope violation: {item}" for item in violations)
        required_fixes.append("Repair the branch so the realized typing satisfies the active search envelope.")
    if drift.get("drifted"):
        reusable_judgments.append(
            "Typing drift was detected between the planned branch intent and the realized code/config surface."
        )
    if violations:
        reusable_judgments.append("Validate-stage fan-out is blocked until envelope violations are resolved.")
    branch_quality = dict(payload.get("branch_quality", {})) if isinstance(payload.get("branch_quality"), dict) else {}
    branch_quality.update(
        {
            "branch_role": str(codegen.get("branch_role", "") or plan.get("branch_role", "")),
            "idea_class": str(codegen.get("idea_class", "") or plan.get("idea_class", "")),
            "verify_status": str(codegen.get("verify_status", "")),
            "proposal_typing_id": str(proposal_typing.get("proposal_typing_id", "") or codegen.get("proposal_typing_id", "")),
            "realized_typing_id": str(realized_typing.get("realized_typing_id", "") or codegen.get("realized_typing_id", "")),
            "grounding_mode": str(
                realized_typing.get("typing_payload", {}).get(
                    "grounding_mode",
                    proposal_typing.get("typing_payload", {}).get("grounding_mode", ""),
                )
            ),
            "cost_tier": str(
                realized_typing.get("typing_payload", {}).get(
                    "cost_tier",
                    info_gain_estimate.get("cost_tier", ""),
                )
            ),
        }
    )
    payload["status"] = status
    payload["concerns"] = list(dict.fromkeys(concerns))
    payload["warnings"] = list(dict.fromkeys(warnings))
    payload["required_fixes"] = list(dict.fromkeys(required_fixes))
    payload["reusable_judgments"] = list(dict.fromkeys(reusable_judgments))
    payload["branch_quality"] = branch_quality
    payload["proposal_typing"] = proposal_typing
    payload["realized_typing"] = realized_typing
    payload["info_gain_estimate"] = info_gain_estimate
    payload["typing_drift"] = drift
    payload["envelope_violations"] = violations
    payload["search_envelope"] = search_envelope
    return payload


def build_critic(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    plan = latest_stage_payload(state, run_id, "plan")
    codegen = latest_stage_payload(state, run_id, "codegen")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="critic",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    plan_status = str(plan.get("plan_status", "hold"))
    write_input_manifest(input_manifest_path, {"run": run.to_dict(), "plan": plan, "codegen": codegen})
    if plan_status != "planned":
        payload = {
            "stage": "critic",
            "status": "approved",
            "concerns": [],
            "warnings": [f"Critic provider was not invoked because the plan is `{plan_status}`."],
            "required_fixes": [],
            "branch_quality": {
                "branch_role": str(codegen.get("branch_role", "") or plan.get("branch_role", "")),
                "idea_class": str(codegen.get("idea_class", "") or plan.get("idea_class", "")),
                "verify_status": str(codegen.get("verify_status", "")),
                "plan_status": plan_status,
            },
            "branch_memory_ids": [str(item) for item in codegen.get("branch_memory_ids", [])],
            "reusable_judgments": [
                f"Critic short-circuited because this run is following the `{plan_status}` route rather than an experiment materialization route."
            ],
        }
        markdown = stage_markdown(
            f"Critic Review {run_id}",
            [
                "- Status: `approved`",
                f"- Reason: plan_status=`{plan_status}` short-circuits critic review.",
                "- Provider invocation: skipped",
            ],
        )
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
        return stage_run

    adapted = run_configured_stage_adapter(
        config,
        state,
        stage_run,
        input_manifest_path=input_manifest_path,
        extra_env={"KAGGLE_AGENT_RUN_ID": run_id},
    )
    if adapted is not None:
        payload, markdown = adapted
        payload.setdefault("warnings", [])
        payload.setdefault("required_fixes", [])
        payload.setdefault("reusable_judgments", [])
        payload.setdefault(
            "branch_quality",
            {
                "branch_role": str(codegen.get("branch_role", "") or plan.get("branch_role", "")),
                "idea_class": str(codegen.get("idea_class", "") or plan.get("idea_class", "")),
                "verify_status": str(codegen.get("verify_status", "")),
            },
        )
        payload.setdefault("branch_memory_ids", [str(item) for item in codegen.get("branch_memory_ids", [])])
        payload = _apply_typing_contract(state, run_id, plan, codegen, payload)
        if payload.get("typing_drift") or payload.get("envelope_violations"):
            markdown = markdown.rstrip() + "\n\n## Contract Enforcement\n"
            typing_drift = payload.get("typing_drift", {})
            if isinstance(typing_drift, dict):
                for item in typing_drift.get("summary", []):
                    markdown += f"\n- typing: {item}"
            for item in payload.get("envelope_violations", []):
                markdown += f"\n- envelope: {item}"
            markdown += "\n"
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
        return stage_run

    status = "approved"
    concerns: list[str] = []
    warnings: list[str] = []
    required_fixes: list[str] = []
    config_path = str(codegen.get("generated_config_path", ""))
    if codegen.get("status") == "noop":
        warnings.append("No new code was generated for this plan.")
    if config_path and not Path(config_path).exists():
        status = "rejected"
        concerns.append(f"Missing generated config: {config_path}")
        required_fixes.append("Materialize the generated config before validation.")
    verify_status = str(codegen.get("verify_status") or codegen.get("smoke_status") or "")
    if verify_status == "failed":
        status = "rejected"
        concerns.append(str(codegen.get("verify_summary") or "Codegen verify failed."))
        required_fixes.append("Repair the source edits or generated config until the deterministic verify passes.")
    if not codegen.get("changed_files"):
        warnings.append("Codegen reported no changed files; confirm whether the branch is intentionally config-only.")
    reusable_judgments = []
    branch_role = str(codegen.get("branch_role", "") or plan.get("branch_role", ""))
    idea_class = str(codegen.get("idea_class", "") or plan.get("idea_class", ""))
    if branch_role or idea_class:
        reusable_judgments.append(f"Branch `{branch_role or 'n/a'}` on `{idea_class or 'n/a'}` returned verify_status={verify_status or 'n/a'}.")
    if concerns:
        reusable_judgments.append("Do not dispatch this branch portfolio until the required fixes are addressed.")
    else:
        reusable_judgments.append("This bundle is structurally acceptable for validate-stage fan-out.")
    payload = {
        "stage": "critic",
        "status": status,
        "concerns": concerns,
        "warnings": warnings,
        "required_fixes": required_fixes,
        "branch_quality": {
            "branch_role": branch_role,
            "idea_class": idea_class,
            "verify_status": verify_status,
            "policy_trace": [str(item) for item in codegen.get("policy_trace", [])],
            "motivation_summary": str(codegen.get("motivation_summary", "")),
        },
        "branch_memory_ids": [str(item) for item in codegen.get("branch_memory_ids", [])],
        "reusable_judgments": reusable_judgments,
    }
    payload = _apply_typing_contract(state, run_id, plan, codegen, payload)
    markdown = stage_markdown(
        f"Critic Review {run_id}",
        [
            f"- Status: `{status}`",
            *(f"- Concern: {item}" for item in concerns),
            *(f"- Warning: {item}" for item in warnings),
            *(f"- Fix: {item}" for item in required_fixes),
            "",
            "## Reusable Judgments",
            *(f"- {item}" for item in reusable_judgments),
        ],
    )
    complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
    return stage_run
