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
    codegen = latest_stage_payload(state, run_id, "codegen")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="critic",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    write_input_manifest(input_manifest_path, {"run": run.to_dict(), "codegen": codegen})
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

    status = "approved"
    concerns: list[str] = []
    config_path = str(codegen.get("generated_config_path", ""))
    if codegen.get("status") == "noop":
        concerns.append("No new code was generated for this plan.")
    if config_path and not Path(config_path).exists():
        status = "rejected"
        concerns.append(f"Missing generated config: {config_path}")
    payload = {
        "stage": "critic",
        "status": status,
        "concerns": concerns,
    }
    markdown = stage_markdown(
        f"Critic Review {run_id}",
        [f"- Status: `{status}`", *(f"- Concern: {item}" for item in concerns)] or ["- Status: `approved`"],
    )
    complete_stage_run(stage_run, payload=payload, markdown=markdown)
    return stage_run
