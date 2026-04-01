from __future__ import annotations

import json
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
from kaggle_agent.utils import atomic_write_text


FALLBACK_ALLOWED_EDIT_ROOTS = [
    "train_sed.py",
    "BirdCLEF-2026-Codebase/configs",
    "BirdCLEF-2026-Codebase/src",
    "BirdCLEF-2026-Codebase/train.py",
    "BirdCLEF-2026-Codebase/inference.py",
    "BirdCLEF-2026-Codebase/scripts",
]
CODEGEN_MAX_REPAIR_ATTEMPTS = 3


def _codegen_retry_manifest(
    run,
    plan: dict[str, object],
    *,
    previous_payload: dict[str, object] | None = None,
    previous_critic: dict[str, object] | None = None,
    attempt_number: int = 1,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "run": run.to_dict(),
        "plan": plan,
        "codegen_attempt_number": attempt_number,
    }
    if previous_critic is not None and str(previous_critic.get("status", "")) == "rejected":
        manifest["previous_critic_attempt"] = {
            "status": str(previous_critic.get("status", "")),
            "concerns": [str(item) for item in previous_critic.get("concerns", [])],
            "warnings": [str(item) for item in previous_critic.get("warnings", [])],
            "required_fixes": [str(item) for item in previous_critic.get("required_fixes", [])],
            "amp_probe_summary": str(previous_critic.get("amp_probe_summary", "")),
        }
    if previous_payload is None:
        return manifest
    manifest["previous_codegen_attempt"] = {
        "attempt_number": attempt_number - 1,
        "status": str(previous_payload.get("status", "")),
        "reason": str(previous_payload.get("reason", "")),
        "verify_status": str(previous_payload.get("verify_status", "")),
        "verify_summary": str(previous_payload.get("verify_summary", "")),
        "changed_files": [str(item) for item in previous_payload.get("changed_files", [])],
    }
    return manifest


def _should_retry_codegen_payload(payload: dict[str, object], *, plan_status: str, attempt_number: int) -> bool:
    if plan_status != "planned":
        return False
    if attempt_number >= CODEGEN_MAX_REPAIR_ATTEMPTS:
        return False
    status = str(payload.get("status", "") or "")
    verify_status = str(payload.get("verify_status") or payload.get("smoke_status") or "")
    return status != "generated" or verify_status == "failed"


def _primary_branch(plan: dict[str, object]) -> dict[str, object]:
    branches = plan.get("branch_plans")
    if isinstance(branches, list):
        for item in branches:
            if isinstance(item, dict):
                return dict(item)
    return dict(plan)


def _decorate_codegen_payload(payload: dict[str, object], plan: dict[str, object]) -> dict[str, object]:
    branch = _primary_branch(plan)
    payload = dict(payload)
    payload.setdefault("portfolio_id", str(branch.get("portfolio_id") or plan.get("portfolio_id") or ""))
    payload.setdefault("branch_role", str(branch.get("branch_role") or ""))
    payload.setdefault("idea_class", str(branch.get("idea_class") or ""))
    payload.setdefault("knowledge_card_ids", [str(item) for item in branch.get("knowledge_card_ids", [])] or [str(item) for item in plan.get("knowledge_card_ids", [])])
    payload.setdefault("policy_trace", [str(item) for item in branch.get("policy_trace", [])] or [str(item) for item in plan.get("policy_trace", [])])
    payload.setdefault("branch_memory_ids", [str(item) for item in branch.get("branch_memory_ids", [])])
    payload.setdefault("scheduler_hints", dict(branch.get("scheduler_hints", {})) if isinstance(branch.get("scheduler_hints"), dict) else dict(plan.get("scheduler_hints", {})))
    payload.setdefault(
        "motivation_summary",
        str(branch.get("reason") or branch.get("hypothesis") or plan.get("reason") or ""),
    )
    return payload


def _append_codegen_context_markdown(markdown: str, plan: dict[str, object], payload: dict[str, object]) -> str:
    has_meaningful_context = any(
        [
            str(payload.get("branch_role", "")).strip(),
            str(payload.get("idea_class", "")).strip(),
            str(payload.get("portfolio_id", "")).strip(),
            str(payload.get("motivation_summary", "")).strip(),
            bool(payload.get("policy_trace")),
            bool(payload.get("branch_memory_ids")),
            bool(payload.get("scheduler_hints")),
        ]
    )
    if not has_meaningful_context:
        return markdown
    context_lines = [
        "",
        "## Branch Context",
        f"- branch_role: `{payload.get('branch_role', '') or 'n/a'}`",
        f"- idea_class: `{payload.get('idea_class', '') or 'n/a'}`",
        f"- portfolio_id: `{payload.get('portfolio_id', '') or 'n/a'}`",
        f"- motivation: {payload.get('motivation_summary', '') or 'n/a'}",
    ]
    policy_trace = payload.get("policy_trace")
    if isinstance(policy_trace, list) and policy_trace:
        context_lines.extend(["", "## Policy Trace", *(f"- `{item}`" for item in policy_trace if str(item).strip())])
    branch_memory_ids = payload.get("branch_memory_ids")
    if isinstance(branch_memory_ids, list) and branch_memory_ids:
        context_lines.extend(["", "## Branch Memories", *(f"- `{item}`" for item in branch_memory_ids if str(item).strip())])
    scheduler_hints = payload.get("scheduler_hints")
    if isinstance(scheduler_hints, dict) and scheduler_hints:
        context_lines.extend(
            [
                "",
                "## Scheduler Hints",
                *(f"- {key}: `{value}`" for key, value in scheduler_hints.items()),
            ]
        )
    suffix = "\n".join(context_lines).strip()
    return markdown.rstrip() + ("\n\n" + suffix if suffix else "")


def build_codegen(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    plan = latest_stage_payload(state, run_id, "plan")
    critic = latest_stage_payload(state, run_id, "critic")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="codegen",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    plan_status = str(plan.get("plan_status", "hold"))
    if plan_status != "planned":
        write_input_manifest(
            input_manifest_path,
            {
                "run": run.to_dict(),
                "plan": plan,
                "codegen_attempt_number": 0,
                "skip_reason": f"plan_status={plan_status}",
            },
        )
        payload = {
            "stage": "codegen",
            "status": "noop",
            "reason": f"plan_status={plan_status}",
            "generated_config_path": "",
            "run_bundle_path": "",
            "patch_path": "",
            "code_state_ref": "",
            "verify_artifacts_ref": "",
            "verify_command": "",
            "verify_status": "skipped",
            "verify_summary": f"Codegen provider was not invoked because the plan is `{plan_status}`.",
            "worktree_path": "",
            "base_commit": "",
            "head_commit": "",
            "changed_files": [],
            "provider_runtime": "deterministic-skip",
            "allowed_edit_roots": list(FALLBACK_ALLOWED_EDIT_ROOTS),
            "smoke_status": "skipped",
            "smoke_summary": f"Codegen provider was not invoked because the plan is `{plan_status}`.",
        }
        payload = _decorate_codegen_payload(payload, plan)
        markdown = stage_markdown(
            f"Codegen {run_id}",
            [
                "- Status: `noop`",
                f"- Reason: {payload['reason']}",
                "- Provider invocation: skipped",
                f"- Verify summary: {payload['verify_summary']}",
                f"- Branch role: `{payload.get('branch_role', '') or 'n/a'}`",
                f"- Idea class: `{payload.get('idea_class', '') or 'n/a'}`",
            ],
        )
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
        return stage_run

    previous_payload: dict[str, object] | None = None
    previous_markdown = ""
    for attempt_number in range(1, CODEGEN_MAX_REPAIR_ATTEMPTS + 1):
        write_input_manifest(
            input_manifest_path,
            _codegen_retry_manifest(
                run,
                plan,
                previous_payload=previous_payload,
                previous_critic=critic,
                attempt_number=attempt_number,
            ),
        )
        adapted = run_configured_stage_adapter(
            config,
            state,
            stage_run,
            input_manifest_path=input_manifest_path,
            extra_env={"KAGGLE_AGENT_RUN_ID": run_id, "KAGGLE_AGENT_CODEGEN_ATTEMPT": str(attempt_number)},
        )
        if adapted is None:
            break
        payload, markdown = adapted
        payload = _decorate_codegen_payload(payload, plan)
        markdown = _append_codegen_context_markdown(markdown, plan, payload)
        previous_payload = payload
        previous_markdown = markdown
        if not _should_retry_codegen_payload(payload, plan_status=plan_status, attempt_number=attempt_number):
            complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
            return stage_run

    if previous_payload is not None:
        complete_stage_run(stage_run, state=state, payload=previous_payload, markdown=previous_markdown)
        return stage_run
    output_dir = Path(stage_run.output_dir)
    config_path = Path(str(plan["config_path"]))
    if not config_path.is_absolute():
        config_path = config.root / config_path
    generated_copy = output_dir / "generated_config.yaml"
    generated_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    run_bundle_path = output_dir / "run_bundle.json"
    run_bundle = {
        "spec_type": "experiment",
        "title": plan["title"],
        "family": plan["family"],
        "config_path": str(config_path),
        "launch_mode": plan.get("launch_mode", "background"),
        "dedupe_key": plan.get("dedupe_key", ""),
        "notes": ["deterministic codegen fallback"],
        "verify_status": "skipped",
    }
    atomic_write_text(run_bundle_path, json.dumps(run_bundle, indent=2) + "\n")
    patch_path = output_dir / "patch.diff"
    atomic_write_text(patch_path, "")
    payload = {
        "stage": "codegen",
        "status": "generated",
        "reason": "Deterministic fallback copied the planned config and produced an empty patch.",
        "generated_config_path": str(generated_copy),
        "run_bundle_path": str(run_bundle_path),
        "patch_path": str(patch_path),
        "code_state_ref": "",
        "verify_artifacts_ref": "",
        "verify_command": "",
        "verify_status": "skipped",
        "verify_summary": "Deterministic fallback does not run an isolated verify command.",
        "worktree_path": "",
        "base_commit": "",
        "head_commit": "",
        "changed_files": [],
        "provider_runtime": "deterministic-fallback",
        "allowed_edit_roots": list(FALLBACK_ALLOWED_EDIT_ROOTS),
        "smoke_status": "skipped",
        "smoke_summary": "Deterministic fallback does not produce an isolated code snapshot.",
    }
    payload = _decorate_codegen_payload(payload, plan)
    markdown = stage_markdown(
        f"Codegen {run_id}",
        [
            f"- Status: `generated`",
            f"- Generated config: `{generated_copy}`",
            f"- Run bundle: `{run_bundle_path}`",
            f"- Branch role: `{payload.get('branch_role', '') or 'n/a'}`",
            f"- Idea class: `{payload.get('idea_class', '') or 'n/a'}`",
        ],
    )
    complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
    return stage_run
