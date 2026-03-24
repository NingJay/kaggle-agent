from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggle_agent.adapters.command import parse_json_payload, run_stage_adapter
from kaggle_agent.schema import AgentRun, RunRecord, StageRun, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_json, atomic_write_text, ensure_directory, now_utc_iso, truncate


STAGE_SEQUENCE = [
    "evidence",
    "report",
    "research",
    "decision",
    "plan",
    "codegen",
    "critic",
    "validate",
    "submission",
]

STAGE_TO_ADAPTER_FIELD = {
    "evidence": "evidence_command",
    "report": "report_command",
    "research": "research_command",
    "decision": "decision_command",
    "plan": "planner_command",
    "codegen": "codegen_command",
    "critic": "critic_command",
    "submission": "submission_command",
}


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = parse_json_payload(path)
    return payload if isinstance(payload, dict) else {}


def load_run_result(run: RunRecord) -> dict[str, Any]:
    path = run.artifact_paths.get("result", "")
    return load_json_file(Path(path)) if path else {}


def latest_stage_run(state: WorkspaceState, run_id: str, stage_name: str | None = None) -> StageRun | None:
    for stage_run in reversed(state.stage_runs):
        if stage_run.run_id != run_id:
            continue
        if stage_name is not None and stage_run.stage_name != stage_name:
            continue
        return stage_run
    return None


def latest_stage_payload(state: WorkspaceState, run_id: str, stage_name: str) -> dict[str, Any]:
    stage_run = latest_stage_run(state, run_id, stage_name)
    if stage_run is None:
        return {}
    return load_json_file(Path(stage_run.output_json_path))


def next_stage_name(stage_name: str) -> str:
    if stage_name not in STAGE_SEQUENCE:
        return "complete"
    index = STAGE_SEQUENCE.index(stage_name)
    if index == len(STAGE_SEQUENCE) - 1:
        return "complete"
    return STAGE_SEQUENCE[index + 1]


def prompt_path_for_stage(config: WorkspaceConfig, stage_name: str) -> Path | None:
    path = config.prompt_path(f"{stage_name}.md")
    return path if path.exists() else None


def _work_item_from_run(state: WorkspaceState, run: RunRecord):
    return next((item for item in state.work_items if item.id == run.work_item_id), None)


def begin_stage_run(
    config: WorkspaceConfig,
    state: WorkspaceState,
    run: RunRecord,
    *,
    stage_name: str,
    input_ref: str,
) -> tuple[StageRun, Path]:
    stage_run_id = f"stage-{state.runtime.next_stage_number:04d}-{stage_name}"
    state.runtime.next_stage_number += 1
    output_dir = ensure_directory(config.stage_dir(stage_name, stage_run_id))
    prompt_path = prompt_path_for_stage(config, stage_name)
    stage_run = StageRun(
        stage_run_id=stage_run_id,
        run_id=run.run_id,
        work_item_id=run.work_item_id,
        stage_name=stage_name,
        status="running",
        input_ref=input_ref,
        output_dir=str(output_dir),
        output_json_path=str(output_dir / f"{stage_name}.json"),
        output_md_path=str(output_dir / f"{stage_name}.md"),
        prompt_path=str(prompt_path) if prompt_path is not None else "",
        created_at=now_utc_iso(),
        updated_at=now_utc_iso(),
    )
    state.stage_runs.append(stage_run)
    run.latest_stage_run_id = stage_run.stage_run_id
    run.stage_updated_at = stage_run.updated_at
    work_item = _work_item_from_run(state, run)
    if work_item is not None:
        work_item.latest_stage_run_id = stage_run.stage_run_id
        work_item.updated_at = stage_run.updated_at
    return stage_run, output_dir / "input_manifest.json"


def register_agent_run(
    config: WorkspaceConfig,
    state: WorkspaceState,
    stage_run: StageRun,
    *,
    command: str,
    agent_role: str,
) -> AgentRun:
    agent_run = AgentRun(
        agent_run_id=f"agent-{state.runtime.next_agent_run_number:04d}",
        stage_run_id=stage_run.stage_run_id,
        agent_role=agent_role,
        adapter_command=command,
        prompt_path=stage_run.prompt_path,
        status="running",
        created_at=now_utc_iso(),
    )
    state.runtime.next_agent_run_number += 1
    state.agent_runs.append(agent_run)
    return agent_run


def complete_stage_run(
    stage_run: StageRun,
    *,
    payload: dict[str, Any],
    markdown: str,
    status: str = "completed",
    validator_status: str = "",
) -> None:
    atomic_write_json(Path(stage_run.output_json_path), payload)
    atomic_write_text(Path(stage_run.output_md_path), markdown if markdown.endswith("\n") else markdown + "\n")
    stage_run.status = status
    stage_run.validator_status = validator_status
    stage_run.updated_at = now_utc_iso()


def fail_stage_run(stage_run: StageRun, error: Exception | str) -> None:
    message = truncate(str(error), limit=800)
    stage_run.status = "failed"
    stage_run.error = message
    stage_run.updated_at = now_utc_iso()


def stage_markdown(title: str, lines: list[str]) -> str:
    body = [f"# {title}", "", *lines]
    return "\n".join(body).rstrip() + "\n"


def write_input_manifest(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def run_configured_stage_adapter(
    config: WorkspaceConfig,
    state: WorkspaceState,
    stage_run: StageRun,
    *,
    input_manifest_path: Path,
    extra_env: dict[str, str] | None = None,
) -> tuple[dict[str, Any], str] | None:
    adapter_field = STAGE_TO_ADAPTER_FIELD.get(stage_run.stage_name)
    if adapter_field is None:
        return None
    command = getattr(config.adapters, adapter_field, "").strip()
    if not command:
        return None

    agent_run = register_agent_run(
        config,
        state,
        stage_run,
        command=command,
        agent_role=stage_run.stage_name,
    )
    try:
        result = run_stage_adapter(
            command,
            stage=stage_run.stage_name,
            workspace_root=config.root,
            input_manifest_path=input_manifest_path,
            output_dir=Path(stage_run.output_dir),
            prompt_path=Path(stage_run.prompt_path) if stage_run.prompt_path else None,
            extra_env=extra_env,
        )
        agent_run.status = "completed"
        agent_run.output_json_path = str(result["json_path"])
        agent_run.output_md_path = str(result["md_path"])
        payload = load_json_file(result["json_path"])
        markdown = result["md_path"].read_text(encoding="utf-8")
        stage_run.adapter_name = adapter_field
        return payload, markdown
    except Exception as error:  # noqa: BLE001
        agent_run.status = "failed"
        agent_run.log_path = truncate(str(error), limit=800)
        raise
