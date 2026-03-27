from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kaggle_agent.adapters.command import CommandAdapterUnavailable, parse_json_payload, run_stage_adapter
from kaggle_agent.layout import STAGE_SEQUENCE, current_attempt_slug, run_label, stage_label
from kaggle_agent.schema import AgentRun, RunRecord, StageRun, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_json, atomic_write_text, ensure_directory, now_utc_iso, truncate

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


def schema_path_for_stage(config: WorkspaceConfig, stage_name: str) -> Path | None:
    path = config.root / "schemas" / f"{stage_name}.schema.json"
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
    work_item = _work_item_from_run(state, run)
    attempt_slug = current_attempt_slug(state.runtime)
    readable_run_label = run_label(run.run_id, work_item.title if work_item is not None else run.experiment_id)
    output_dir = ensure_directory(
        config.stage_dir(
            attempt_slug,
            readable_run_label,
            stage_label(stage_name, stage_status="running"),
        )
    )
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
        schema_path=str(schema_path_for_stage(config, stage_name) or ""),
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


def _rebase_stage_path(path_value: str, old_dir: Path, new_dir: Path) -> str:
    if not path_value:
        return path_value
    path = Path(path_value)
    try:
        return str(new_dir / path.relative_to(old_dir))
    except ValueError:
        return path_value


def _relocate_stage_output(
    state: WorkspaceState | None,
    stage_run: StageRun,
    *,
    payload: dict[str, Any] | None = None,
    status: str,
    validator_status: str = "",
) -> tuple[Path, Path]:
    old_dir = Path(stage_run.output_dir)
    new_dir = old_dir.parent / stage_label(
        stage_run.stage_name,
        payload,
        stage_status=status,
        validator_status=validator_status,
    )
    if new_dir == old_dir:
        return old_dir, new_dir
    if old_dir.exists():
        ensure_directory(new_dir.parent)
        old_dir.rename(new_dir)
    stage_run.output_dir = str(new_dir)
    stage_run.output_json_path = _rebase_stage_path(stage_run.output_json_path, old_dir, new_dir)
    stage_run.output_md_path = _rebase_stage_path(stage_run.output_md_path, old_dir, new_dir)
    stage_run.spec_path = _rebase_stage_path(stage_run.spec_path, old_dir, new_dir)
    stage_run.provider_meta_path = _rebase_stage_path(stage_run.provider_meta_path, old_dir, new_dir)
    if state is None:
        return old_dir, new_dir
    for agent_run in state.agent_runs:
        if agent_run.stage_run_id != stage_run.stage_run_id:
            continue
        agent_run.output_json_path = _rebase_stage_path(agent_run.output_json_path, old_dir, new_dir)
        agent_run.output_md_path = _rebase_stage_path(agent_run.output_md_path, old_dir, new_dir)
        agent_run.raw_stdout_path = _rebase_stage_path(agent_run.raw_stdout_path, old_dir, new_dir)
        agent_run.raw_stderr_path = _rebase_stage_path(agent_run.raw_stderr_path, old_dir, new_dir)
        agent_run.raw_event_log_path = _rebase_stage_path(agent_run.raw_event_log_path, old_dir, new_dir)
        agent_run.provider_meta_path = _rebase_stage_path(agent_run.provider_meta_path, old_dir, new_dir)
    for spec in state.specs:
        if spec.source_stage_run_id == stage_run.stage_run_id:
            spec.payload_path = _rebase_stage_path(spec.payload_path, old_dir, new_dir)
    for validation in state.validations:
        if validation.source_stage_run_id != stage_run.stage_run_id:
            continue
        validation.output_json_path = _rebase_stage_path(validation.output_json_path, old_dir, new_dir)
        validation.output_md_path = _rebase_stage_path(validation.output_md_path, old_dir, new_dir)
    for json_path in new_dir.rglob("*.json"):
        _rewrite_json_file_paths(json_path, old_dir, new_dir)
    return old_dir, new_dir


def _rebase_payload_paths(value: Any, old_dir: Path, new_dir: Path) -> Any:
    if isinstance(value, str):
        return _rebase_stage_path(value, old_dir, new_dir)
    if isinstance(value, list):
        return [_rebase_payload_paths(item, old_dir, new_dir) for item in value]
    if isinstance(value, dict):
        return {key: _rebase_payload_paths(item, old_dir, new_dir) for key, item in value.items()}
    return value


def _rewrite_json_file_paths(path: Path, old_dir: Path, new_dir: Path) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    atomic_write_json(path, _rebase_payload_paths(payload, old_dir, new_dir))


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
        schema_path=stage_run.schema_path,
        status="running",
        created_at=now_utc_iso(),
        started_at=now_utc_iso(),
    )
    state.runtime.next_agent_run_number += 1
    state.agent_runs.append(agent_run)
    return agent_run


def _read_meta_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = load_json_file(path)
    return payload if isinstance(payload, dict) else {}


def _apply_provider_metadata(agent_run: AgentRun, stage_run: StageRun, result: dict[str, Path]) -> None:
    meta_path = result.get("meta_path")
    meta = _read_meta_payload(meta_path if isinstance(meta_path, Path) else None)
    if isinstance(meta_path, Path) and meta_path.exists():
        stage_run.provider_meta_path = str(meta_path)
        agent_run.provider_meta_path = str(meta_path)
    if "spec_path" in result and isinstance(result["spec_path"], Path):
        stage_run.spec_path = str(result["spec_path"])
    if meta:
        agent_run.provider = str(meta.get("provider", ""))
        agent_run.model = str(meta.get("model", ""))
        agent_run.schema_path = str(meta.get("schema_path", agent_run.schema_path))
        agent_run.raw_stdout_path = str(meta.get("raw_stdout_path", ""))
        agent_run.raw_stderr_path = str(meta.get("raw_stderr_path", ""))
        agent_run.raw_event_log_path = str(meta.get("raw_event_log_path", ""))
        agent_run.session_id = str(meta.get("session_id", ""))
        agent_run.thread_id = str(meta.get("thread_id", ""))
        if meta.get("exit_code") is not None:
            try:
                agent_run.exit_code = int(meta["exit_code"])
            except (TypeError, ValueError):
                agent_run.exit_code = None
        agent_run.started_at = str(meta.get("started_at", agent_run.started_at))
        agent_run.completed_at = str(meta.get("completed_at", ""))
        stage_run.schema_path = str(meta.get("schema_path", stage_run.schema_path))
    else:
        agent_run.completed_at = now_utc_iso()


def complete_stage_run(
    stage_run: StageRun,
    *,
    state: WorkspaceState | None = None,
    payload: dict[str, Any],
    markdown: str,
    status: str = "completed",
    validator_status: str = "",
) -> None:
    old_dir, new_dir = _relocate_stage_output(state, stage_run, payload=payload, status=status, validator_status=validator_status)
    persisted_payload = _rebase_payload_paths(payload, old_dir, new_dir)
    atomic_write_json(Path(stage_run.output_json_path), persisted_payload)
    atomic_write_text(Path(stage_run.output_md_path), markdown if markdown.endswith("\n") else markdown + "\n")
    stage_run.status = status
    stage_run.validator_status = validator_status
    stage_run.updated_at = now_utc_iso()


def fail_stage_run(stage_run: StageRun, error: Exception | str, *, state: WorkspaceState | None = None) -> None:
    message = truncate(str(error), limit=800)
    _relocate_stage_output(state, stage_run, status="failed")
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
            shell_init=config.runtime.shell_init,
            conda_env=config.runtime.conda_env,
        )
        agent_run.status = "completed"
        agent_run.output_json_path = str(result["json_path"])
        agent_run.output_md_path = str(result["md_path"])
        _apply_provider_metadata(agent_run, stage_run, result)
        payload = load_json_file(result["json_path"])
        markdown = result["md_path"].read_text(encoding="utf-8")
        stage_run.adapter_name = adapter_field
        return payload, markdown
    except CommandAdapterUnavailable as error:
        agent_run.status = "skipped"
        agent_run.log_path = truncate(str(error), limit=800)
        agent_run.completed_at = now_utc_iso()
        return None
    except Exception as error:  # noqa: BLE001
        agent_run.status = "failed"
        agent_run.log_path = truncate(str(error), limit=800)
        agent_run.completed_at = now_utc_iso()
        raise
