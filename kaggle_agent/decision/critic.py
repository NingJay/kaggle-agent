from __future__ import annotations

from pathlib import Path

from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    latest_stage_payload,
    run_configured_stage_adapter,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState


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
