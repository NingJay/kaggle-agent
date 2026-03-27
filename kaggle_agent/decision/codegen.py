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


def build_codegen(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    plan = latest_stage_payload(state, run_id, "plan")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="codegen",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    write_input_manifest(input_manifest_path, {"run": run.to_dict(), "plan": plan})
    adapted = run_configured_stage_adapter(
        config,
        state,
        stage_run,
        input_manifest_path=input_manifest_path,
        extra_env={"KAGGLE_AGENT_RUN_ID": run_id},
    )
    if adapted is not None:
        payload, markdown = adapted
        complete_stage_run(stage_run, payload=payload, markdown=markdown)
        return stage_run

    plan_status = str(plan.get("plan_status", "hold"))
    output_dir = Path(stage_run.output_dir)
    if plan_status != "planned":
        payload = {
            "stage": "codegen",
            "status": "noop",
            "reason": f"plan_status={plan_status}",
            "generated_config_path": "",
            "run_bundle_path": "",
            "patch_path": "",
            "code_state_ref": "",
            "worktree_path": "",
            "base_commit": "",
            "head_commit": "",
            "changed_files": [],
            "smoke_status": "skipped",
            "smoke_summary": "Codegen did not run because the plan was not in planned state.",
        }
        markdown = stage_markdown(
            f"Codegen {run_id}",
            [f"- Status: `noop`", f"- Reason: {payload['reason']}"],
        )
        complete_stage_run(stage_run, payload=payload, markdown=markdown)
        return stage_run

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
        "worktree_path": "",
        "base_commit": "",
        "head_commit": "",
        "changed_files": [],
        "smoke_status": "skipped",
        "smoke_summary": "Deterministic fallback does not produce an isolated code snapshot.",
    }
    markdown = stage_markdown(
        f"Codegen {run_id}",
        [
            f"- Status: `generated`",
            f"- Generated config: `{generated_copy}`",
            f"- Run bundle: `{run_bundle_path}`",
        ],
    )
    complete_stage_run(stage_run, payload=payload, markdown=markdown)
    return stage_run
