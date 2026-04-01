from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from kaggle_agent.control.lifecycle import (
    entry_stage_for_plan,
    infer_lifecycle_template,
    next_stage,
    resolve_stage_plan,
    validate_stage_plan,
)
from kaggle_agent.control.scheduler import ensure_seed_spec, register_experiment_for_work_item
from kaggle_agent.control.store import find_run, find_work_item
from kaggle_agent.layout import current_attempt_slug, run_label
from kaggle_agent.schema import RunRecord, SpecRecord, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text, now_utc_iso, truncate


def choose_idle_gpu() -> str | None:
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if query.returncode != 0:
        return None
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    busy = {line.strip() for line in apps.stdout.splitlines() if line.strip()}
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid = [part.strip() for part in line.split(",", maxsplit=1)]
        if uuid not in busy:
            return index
    return None


def _runtime_overrides(config: WorkspaceConfig) -> list[str]:
    overrides = [
        f"paths.data_root={Path(config.data.root).expanduser().resolve()}",
        f"data.train_csv={config.data.train_csv}",
        f"data.taxonomy_csv={config.data.taxonomy_csv}",
        f"data.sample_submission_csv={config.data.sample_submission_csv}",
        f"data.train_audio_dir={config.data.train_audio_dir}",
        f"data.train_soundscapes_dir={config.data.train_soundscapes_dir}",
        f"data.train_soundscapes_labels_csv={config.data.train_soundscapes_labels_csv}",
        f"data.test_soundscapes_dir={config.data.test_soundscapes_dir}",
    ]
    if config.data.perch_cache_dir.strip():
        overrides.append(f"data.perch_cache_dir={Path(config.data.perch_cache_dir).expanduser().resolve()}")
    if config.data.perch_model_dir.strip():
        overrides.append(f"model.perch_model_dir={Path(config.data.perch_model_dir).expanduser().resolve()}")
    return overrides


def _next_run_id(state: WorkspaceState, work_item_id: str) -> str:
    suffix = work_item_id.replace("workitem-", "")
    run_id = f"run-{state.runtime.next_run_number:04d}-{suffix}"
    state.runtime.next_run_number += 1
    return run_id


def _validated_spec_for_work_item(state: WorkspaceState, work_item) -> SpecRecord | None:
    if work_item.latest_spec_id:
        direct = next((item for item in state.specs if item.spec_id == work_item.latest_spec_id and item.status == "validated"), None)
        if direct is not None:
            return direct
    candidates = [item for item in state.specs if item.work_item_id == work_item.id and item.status == "validated"]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.updated_at, item.spec_id))
    return candidates[-1]


def _resolve_execution_workspace(config: WorkspaceConfig, spec: SpecRecord, run: RunRecord) -> tuple[Path, Path, Path]:
    config_path = Path(spec.config_path)
    train_entrypoint = Path(config.runtime.train_entrypoint)
    if spec.code_state_ref.strip():
        source_root = Path(spec.code_state_ref).expanduser().resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"Code state not found: {source_root}")
        run_workspace = Path(run.run_dir) / "code_state_workspace"
        if run_workspace.exists():
            shutil.rmtree(run_workspace)
        shutil.copytree(source_root, run_workspace)
        if config_path.is_absolute() and config_path.is_relative_to(config.root):
            config_path = run_workspace / config_path.relative_to(config.root)
        elif not config_path.is_absolute():
            config_path = run_workspace / config_path
        if train_entrypoint.is_absolute() and train_entrypoint.is_relative_to(config.root):
            train_entrypoint = run_workspace / train_entrypoint.relative_to(config.root)
        elif not train_entrypoint.is_absolute():
            train_entrypoint = run_workspace / train_entrypoint
        return run_workspace, config_path, train_entrypoint

    if not config_path.is_absolute():
        config_path = config.root / config_path
    if not train_entrypoint.is_absolute():
        train_entrypoint = Path(config.runtime.train_workdir) / train_entrypoint
    return Path(config.runtime.train_workdir), config_path, train_entrypoint


def _launch_script(config: WorkspaceConfig, work_item_id: str, spec: SpecRecord, run: RunRecord) -> str:
    execution_root, config_path, train_entrypoint = _resolve_execution_workspace(config, spec, run)
    exit_code_path = Path(run.run_dir) / "exit_code.txt"
    runtime_overrides = _runtime_overrides(config)
    env_exports = [
        f"export KAGGLE_AGENT_WORKSPACE_ROOT={shlex.quote(str(config.root))}",
        f"export KAGGLE_AGENT_WORK_ITEM_ID={shlex.quote(work_item_id)}",
        f"export KAGGLE_AGENT_EXPERIMENT_ID={shlex.quote(run.experiment_id)}",
        f"export KAGGLE_AGENT_RUN_ID={shlex.quote(run.run_id)}",
        f"export KAGGLE_AGENT_RUN_DIR={shlex.quote(run.run_dir)}",
        f"export KAGGLE_AGENT_SPEC_ID={shlex.quote(spec.spec_id)}",
        f"export KAGGLE_AGENT_PRIMARY_METRIC={shlex.quote(config.metrics.primary)}",
        f"export KAGGLE_AGENT_SECONDARY_METRICS={shlex.quote(json.dumps(config.metrics.secondary))}",
        f"export KAGGLE_AGENT_CODE_STATE_REF={shlex.quote(spec.code_state_ref)}",
    ]
    if run.gpu_id:
        env_exports.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(run.gpu_id)}")
    lines = ["#!/usr/bin/env bash", "set +e"]
    if config.runtime.shell_init.strip():
        lines.append(config.runtime.shell_init)
    if config.runtime.conda_env.strip():
        lines.append(f"conda activate {shlex.quote(config.runtime.conda_env)}")
    override_args = " ".join(shlex.quote(item) for item in runtime_overrides)
    python_command = f"python {shlex.quote(str(train_entrypoint))} --config {shlex.quote(str(config_path))}"
    if override_args:
        python_command = f"{python_command} {override_args}"
    lines.extend(
        [
            f"cd {shlex.quote(str(execution_root))}",
            *env_exports,
            python_command,
            "status=$?",
            f"echo \"$status\" > {shlex.quote(str(exit_code_path))}",
            "exit \"$status\"",
        ]
    )
    return "\n".join(lines) + "\n"


def _resolved_work_item_stage_plan(config: WorkspaceConfig, work_item) -> list[str]:
    lifecycle_template = str(work_item.lifecycle_template or "").strip()
    if work_item.pipeline:
        stage_plan = validate_stage_plan(work_item.pipeline, strict=config.automation.strict_stage_graph)
        inferred = infer_lifecycle_template(stage_plan)
        if not lifecycle_template or (
            lifecycle_template == "recursive_experiment"
            and stage_plan != resolve_stage_plan("recursive_experiment", strict=False)
        ):
            lifecycle_template = inferred
    else:
        if not lifecycle_template:
            lifecycle_template = "recursive_experiment"
        stage_plan = resolve_stage_plan(lifecycle_template, strict=config.automation.strict_stage_graph)
    work_item.lifecycle_template = lifecycle_template or "recursive_experiment"
    work_item.pipeline = list(stage_plan)
    return stage_plan


def launch_execute_stage(
    config: WorkspaceConfig,
    state: WorkspaceState,
    work_item,
    spec: SpecRecord,
    experiment,
    run: RunRecord,
    *,
    background: bool,
) -> RunRecord:
    run_dir = Path(run.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(run.log_path)
    started_at = run.started_at or now_utc_iso()
    run.started_at = started_at
    run.completed_at = ""
    run.status = "running"
    run.error = ""
    run.stage_cursor = ""
    run.stage_error = ""
    run.stage_updated_at = now_utc_iso()
    run.root_cause = ""
    run.verdict = ""
    gpu_id = choose_idle_gpu() or ""
    execution_root, config_path, train_entrypoint = _resolve_execution_workspace(config, spec, run)
    run.command = f"python {train_entrypoint} --config {config_path}"
    run.cwd = str(execution_root)
    run.gpu_id = gpu_id
    launch_script = run_dir / "launch.sh"
    atomic_write_text(launch_script, _launch_script(config, work_item.id, spec, run))
    launch_script.chmod(0o755)
    work_item.status = "running"
    work_item.latest_run_id = run.run_id
    work_item.latest_spec_id = spec.spec_id
    work_item.updated_at = now_utc_iso()
    experiment.status = "running"
    experiment.latest_run_id = run.run_id
    experiment.spec_id = spec.spec_id
    experiment.updated_at = now_utc_iso()
    if run.run_id not in state.runtime.active_run_ids:
        state.runtime.active_run_ids.append(run.run_id)
    with log_path.open("w", encoding="utf-8") as handle:
        if background:
            process = subprocess.Popen(
                ["/usr/bin/bash", str(launch_script)],
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=config.root,
                start_new_session=True,
            )
            run.pid = process.pid
            return run
        completed = subprocess.run(
            ["/usr/bin/bash", str(launch_script)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=config.root,
            check=False,
        )
        run.pid = None
        if completed.returncode != 0:
            run.error = f"process exited with code {completed.returncode}"
    finalize_run(config, state, run.run_id)
    return run


def start_run(
    config: WorkspaceConfig,
    state: WorkspaceState,
    work_item_id: str,
    *,
    background: bool = True,
) -> RunRecord:
    work_item = find_work_item(state, work_item_id)
    stage_plan = _resolved_work_item_stage_plan(config, work_item)
    lifecycle_template = str(work_item.lifecycle_template or "recursive_experiment")
    entry_stage = entry_stage_for_plan(stage_plan)
    spec = _validated_spec_for_work_item(state, work_item)
    if spec is None:
        spec = ensure_seed_spec(state, work_item)
    experiment = register_experiment_for_work_item(state, work_item, spec)
    run_id = _next_run_id(state, work_item.id)
    attempt_slug = current_attempt_slug(state.runtime)
    readable_run_label = run_label(run_id, work_item.title)
    run_dir = config.run_runtime_dir(attempt_slug, readable_run_label)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.run_log_path(attempt_slug, readable_run_label)
    started_at = now_utc_iso()
    run = RunRecord(
        run_id=run_id,
        experiment_id=experiment.id,
        work_item_id=work_item.id,
        spec_id=spec.spec_id,
        status="running" if entry_stage == "execute" else "succeeded",
        command="" if entry_stage == "execute" else f"synthetic:{entry_stage}",
        cwd="" if entry_stage == "execute" else str(config.root),
        run_dir=str(run_dir),
        log_path=str(log_path),
        started_at=started_at,
        completed_at=started_at if entry_stage != "execute" else "",
        gpu_id="",
        code_state_ref=spec.code_state_ref,
        lifecycle_template=lifecycle_template,
        stage_plan=list(stage_plan),
        stage_cursor=entry_stage if entry_stage != "execute" else "",
        stage_updated_at=started_at if entry_stage != "execute" else "",
        root_cause="synthetic_lifecycle_entry" if entry_stage != "execute" else "",
        verdict="synthetic_lifecycle_entry" if entry_stage != "execute" else "",
    )
    if entry_stage != "execute":
        atomic_write_text(
            log_path,
            f"synthetic lifecycle entry for `{entry_stage}` using template `{lifecycle_template}`\n",
        )
    work_item.status = "running"
    work_item.latest_run_id = run.run_id
    work_item.latest_spec_id = spec.spec_id
    work_item.updated_at = now_utc_iso()
    state.runs.append(run)
    if entry_stage == "execute":
        return launch_execute_stage(
            config,
            state,
            work_item,
            spec,
            experiment,
            run,
            background=background,
        )
    experiment.status = run.status
    experiment.latest_run_id = run.run_id
    experiment.spec_id = spec.spec_id
    experiment.updated_at = now_utc_iso()
    return run


def _process_exists(pid: int) -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            state = proc_stat.read_text(encoding="utf-8").split()[2]
        except (IndexError, OSError):
            state = ""
        if state == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_has_terminal_artifacts(run: RunRecord) -> bool:
    run_root = Path(run.run_dir)
    exit_code_path = run_root / "exit_code.txt"
    result_path = run_root / "result.json"
    if not exit_code_path.exists() or not result_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and bool(payload)


def _resolve_primary_metric(result_payload: dict[str, object], default_metric: str) -> tuple[str, float | None]:
    all_metrics = result_payload.get("all_metrics", {})
    if isinstance(all_metrics, dict) and "val_soundscape_macro_roc_auc" in all_metrics:
        return "val_soundscape_macro_roc_auc", float(all_metrics["val_soundscape_macro_roc_auc"])

    primary_name = str(result_payload.get("primary_metric_name", default_metric) or default_metric)
    primary_value = result_payload.get("primary_metric_value")
    if primary_value is not None:
        return primary_name, float(primary_value)

    if isinstance(all_metrics, dict):
        if primary_name in all_metrics:
            return primary_name, float(all_metrics[primary_name])
        if "soundscape_macro_roc_auc" in all_metrics:
            return "soundscape_macro_roc_auc", float(all_metrics["soundscape_macro_roc_auc"])
    return primary_name, None


def collect_finished_runs(config: WorkspaceConfig, state: WorkspaceState) -> list[RunRecord]:
    reconcile_active_run_ids(state)
    finished: list[RunRecord] = []
    for run in state.runs:
        if run.status != "running":
            continue
        terminal_artifacts_ready = _run_has_terminal_artifacts(run)
        if not terminal_artifacts_ready and run.pid is not None and _process_exists(run.pid):
            continue
        finalize_run(config, state, run.run_id)
        finished.append(run)
    reconcile_active_run_ids(state)
    return finished


def reconcile_active_run_ids(state: WorkspaceState) -> None:
    running_run_ids = [run.run_id for run in state.runs if run.status == "running"]
    state.runtime.active_run_ids = [run_id for run_id in state.runtime.active_run_ids if run_id in running_run_ids]
    for run_id in running_run_ids:
        if run_id not in state.runtime.active_run_ids:
            state.runtime.active_run_ids.append(run_id)


def finalize_run(config: WorkspaceConfig, state: WorkspaceState, run_id: str) -> RunRecord:
    run = find_run(state, run_id)
    work_item = find_work_item(state, run.work_item_id)
    stage_plan = list(run.stage_plan) if run.stage_plan else _resolved_work_item_stage_plan(config, work_item)
    if not run.stage_plan:
        run.stage_plan = list(stage_plan)
    if not run.lifecycle_template:
        run.lifecycle_template = work_item.lifecycle_template
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    exit_code_path = Path(run.run_dir) / "exit_code.txt"
    result_path = Path(run.run_dir) / "result.json"
    metrics_path = Path(run.run_dir) / "metrics.json"
    artifacts_path = Path(run.run_dir) / "artifacts.json"
    exit_code = None
    if exit_code_path.exists():
        exit_code = int(exit_code_path.read_text(encoding="utf-8").strip() or "1")
    result_payload: dict[str, object] = {}
    if result_path.exists():
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    run.completed_at = now_utc_iso()
    run.artifact_paths = {
        "result": str(result_path) if result_path.exists() else "",
        "metrics": str(metrics_path) if metrics_path.exists() else "",
        "artifacts": str(artifacts_path) if artifacts_path.exists() else "",
    }
    run.primary_metric_name, run.primary_metric_value = _resolve_primary_metric(result_payload, config.metrics.primary)
    secondary = result_payload.get("secondary_metrics", {})
    if isinstance(secondary, dict):
        run.secondary_metrics = {
            str(key): float(value)
            for key, value in secondary.items()
            if str(key) != run.primary_metric_name
        }
    run.root_cause = str(result_payload.get("root_cause", run.error or "unknown"))
    run.verdict = str(result_payload.get("verdict", "unknown"))
    run.status = "succeeded" if exit_code == 0 and result_payload else "failed"
    run.pid = None
    if run.status == "failed" and not run.error:
        if not exit_code_path.exists() and not result_path.exists():
            run.error = "launch process exited before writing exit_code.txt or result.json"
        else:
            log_tail = Path(run.log_path).read_text(encoding="utf-8")[-1000:] if Path(run.log_path).exists() else ""
            run.error = truncate(str(result_payload.get("error", "")) or log_tail or "run failed without structured result")
    if run.run_id in state.runtime.active_run_ids:
        state.runtime.active_run_ids.remove(run.run_id)
    experiment.status = "succeeded" if run.status == "succeeded" else "failed"
    experiment.latest_run_id = run.run_id
    experiment.updated_at = now_utc_iso()
    work_item.latest_run_id = run.run_id
    work_item.status = "reviewing"
    work_item.updated_at = now_utc_iso()
    if not run.stage_cursor:
        next_stage_name = next_stage(stage_plan, "execute") if "execute" in stage_plan else None
        run.stage_cursor = next_stage_name or "complete"
        run.stage_error = ""
        run.stage_updated_at = run.completed_at
    return run
