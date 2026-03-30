from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from kaggle_agent.config import load_workspace_config
from kaggle_agent.control.monitor import maybe_start_next_run, process_completed_runs, tick_workspace, watch_workspace
from kaggle_agent.control.reporting import ensure_surface_files, write_reports
from kaggle_agent.control.scheduler import queue_config_experiment, register_work_item, runnable_work_items
from kaggle_agent.control.store import initialize_workspace, load_state, save_state
from kaggle_agent.control.submission import build_submission_candidate, dry_run_submission_candidate, plan_submission_slots
from kaggle_agent.layout import current_attempt_slug, visible_runs
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import workspace_lock


def load_config(root: Path | None = None) -> WorkspaceConfig:
    return load_workspace_config(root)


def _load_runtime_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def init_workspace(config: WorkspaceConfig, *, archive_legacy: bool = True, force: bool = False) -> WorkspaceState:
    with workspace_lock(config.lock_path()):
        state = initialize_workspace(config, archive_legacy=archive_legacy, force=force)
        ensure_surface_files(config)
        write_reports(config, state)
        save_state(config, state)
        return state


def doctor_checks(config: WorkspaceConfig) -> list[tuple[bool, str, str]]:
    state = load_state(config)
    ensure_surface_files(config)
    checks: list[tuple[bool, str, str]] = []
    default_config = config.runtime_root() / "configs" / "default.yaml"
    debug_config = config.runtime_root() / "configs" / "debug.yaml"
    default_payload = _load_runtime_yaml(default_config)
    training_backend = str(default_payload.get("training", {}).get("backend", ""))
    model_provider = str(default_payload.get("model", {}).get("backbone_provider", ""))

    checks.append((config.root.exists(), "workspace_root", str(config.root)))
    checks.append((config.runtime_root().exists(), "runtime_root", str(config.runtime_root())))
    checks.append(((config.root / config.runtime.train_entrypoint).exists(), "train_entrypoint", str(config.root / config.runtime.train_entrypoint)))
    checks.append((default_config.exists(), "default_config", str(default_config)))
    checks.append((debug_config.exists(), "debug_config", str(debug_config)))
    checks.append((bool(config.runtime.seed_notebook_path.strip()), "seed_notebook_path", config.runtime.seed_notebook_path or "(unset)"))
    if config.runtime.seed_notebook_path.strip():
        checks.append((Path(config.runtime.seed_notebook_path).expanduser().exists(), "seed_notebook_exists", config.runtime.seed_notebook_path))
    checks.append((Path(config.data.root).exists(), "data_root", config.data.root))
    checks.append((_load_runtime_yaml(default_config) != {}, "runtime_yaml_parse", "default runtime yaml"))
    checks.append((config.ledger_path().exists(), "ledger_db", str(config.ledger_path())))
    checks.append((config.root_doc_path("COMPETITION.md").exists(), "competition_doc", str(config.root_doc_path("COMPETITION.md"))))
    checks.append((config.root_doc_path("CHECKLIST.md").exists(), "checklist_doc", str(config.root_doc_path("CHECKLIST.md"))))
    checks.append((config.prompt_path("report.md").exists(), "prompt_report", str(config.prompt_path("report.md"))))
    checks.append((True, "report_adapter", config.adapters.report_command or "(internal fallback)"))
    checks.append((True, "research_adapter", config.adapters.research_command or "(internal fallback)"))
    checks.append((True, "decision_adapter", config.adapters.decision_command or "(internal fallback)"))
    checks.append((True, "planner_adapter", config.adapters.planner_command or "(internal fallback)"))
    checks.append((True, "codegen_adapter", config.adapters.codegen_command or "(internal fallback)"))
    checks.append((True, "critic_adapter", config.adapters.critic_command or "(internal fallback)"))
    checks.append((True, "submission_adapter", config.adapters.submission_command or "(internal fallback)"))
    checks.append((len(state.runtime.active_run_ids) <= config.automation.max_active_runs, "active_run_limit", ",".join(state.runtime.active_run_ids) or "none"))
    checks.append((bool(state.work_items), "seeded_work_items", str(len(state.work_items))))
    checks.append((Path(config.data.root, config.data.sample_submission_csv).exists(), "sample_submission_csv", str(Path(config.data.root, config.data.sample_submission_csv))))
    checks.append((not config.kaggle.cpu_submission_only or not config.kaggle.enable_gpu, "cpu_submission_contract", f"cpu_only={config.kaggle.cpu_submission_only}, enable_gpu={config.kaggle.enable_gpu}"))
    checks.append((not config.kaggle.enable_internet, "internet_off_default", str(config.kaggle.enable_internet)))

    if training_backend == "sklearn_cached_probe":
        checks.append((bool(config.data.perch_cache_dir.strip()), "perch_cache_dir", config.data.perch_cache_dir or "(unset)"))
        if config.data.perch_cache_dir.strip():
            cache_root = Path(config.data.perch_cache_dir)
            checks.append((cache_root.exists(), "perch_cache_root_exists", str(cache_root)))
            checks.append(((cache_root / "full_perch_meta.parquet").exists(), "perch_cache_meta", str(cache_root / "full_perch_meta.parquet")))
            checks.append(((cache_root / "full_perch_arrays.npz").exists(), "perch_cache_arrays", str(cache_root / "full_perch_arrays.npz")))
        checks.append((importlib.util.find_spec("numpy") is not None, "numpy", "required for cached probe backend"))
        checks.append((importlib.util.find_spec("pandas") is not None, "pandas", "required for cached probe backend"))
        checks.append((importlib.util.find_spec("sklearn") is not None, "sklearn", "required for cached probe backend"))
        checks.append((importlib.util.find_spec("pyarrow") is not None, "pyarrow", "required for cached parquet metadata"))
    if model_provider == "perch_saved_model" and training_backend != "sklearn_cached_probe":
        checks.append((importlib.util.find_spec("tensorflow") is not None, "tensorflow", "required for real Perch backend"))
        checks.append((importlib.util.find_spec("soundfile") is not None, "soundfile", "required for audio decode with saved-model backend"))
        checks.append((bool(config.data.perch_model_dir.strip()), "perch_model_dir", config.data.perch_model_dir or "(unset)"))
    return checks


def get_status_state(config: WorkspaceConfig) -> WorkspaceState:
    with workspace_lock(config.lock_path()):
        state = load_state(config)
        write_reports(config, state)
        save_state(config, state)
        return state


def enqueue_config(
    config: WorkspaceConfig,
    config_path: str,
    *,
    title: str | None = None,
    family: str = "ad_hoc",
    priority: int = 50,
) -> WorkspaceState:
    with workspace_lock(config.lock_path()):
        state = load_state(config)
        queue_config_experiment(config, state, config_path, title=title, family=family, priority=priority)
        write_reports(config, state)
        save_state(config, state)
        return state


def enqueue_preflight(config: WorkspaceConfig, *, priority: int = 5, allow_debug: bool = False) -> WorkspaceState:
    if not allow_debug and not config.runtime.allow_debug_preflight:
        raise ValueError("enqueue-preflight is disabled by default; rerun with --allow-debug for an explicit debug-only check.")
    with workspace_lock(config.lock_path()):
        state = load_state(config)
        attempt_slug = current_attempt_slug(state.runtime)
        debug_config = str((config.runtime_root() / "configs" / "debug.yaml").relative_to(config.root))
        register_work_item(
            state,
            title=f"Preflight debug check for {attempt_slug}",
            work_type="preflight_check",
            family="perch_head_debug",
            config_path=debug_config,
            priority=priority,
            pipeline=["execute", "evidence", "report", "research", "decision", "plan", "codegen", "critic", "validate", "submission"],
            dedupe_key=f"manual:preflight:{attempt_slug}",
            notes=["Explicit debug-only preflight work item."],
        )
        write_reports(config, state)
        save_state(config, state)
        return state


def start_next(config: WorkspaceConfig, *, background: bool = True) -> str | None:
    with workspace_lock(config.lock_path()):
        state = load_state(config)
        process_completed_runs(config, state)
        run, run_in_background = maybe_start_next_run(config, state, background=background)
        if run is not None and not run_in_background:
            process_completed_runs(config, state)
        write_reports(config, state)
        save_state(config, state)
        return None if run is None else run.run_id


def tick(config: WorkspaceConfig, *, auto_start: bool = True) -> WorkspaceState:
    return tick_workspace(config, auto_start=auto_start)


def watch(config: WorkspaceConfig, *, interval_seconds: int, iterations: int, auto_start: bool = True) -> None:
    watch_workspace(config, interval_seconds=interval_seconds, iterations=iterations, auto_start=auto_start)


def build_submission(config: WorkspaceConfig, run_id: str | None = None) -> str:
    with workspace_lock(config.lock_path()):
        state = load_state(config)
        if run_id is None:
            candidate_run = next(
                (
                    item
                    for item in sorted(visible_runs(state), key=lambda value: (value.primary_metric_value or -1.0, value.completed_at), reverse=True)
                    if item.status == "succeeded"
                ),
                None,
            )
        else:
            candidate_run = next((item for item in state.runs if item.run_id == run_id), None)
        if candidate_run is None:
            raise ValueError("No eligible run available for submission candidate generation.")
        candidate = build_submission_candidate(config, state, candidate_run.run_id)
        write_reports(config, state)
        save_state(config, state)
        return candidate.id


def dry_run_submission(config: WorkspaceConfig, candidate_id: str) -> dict[str, Any]:
    with workspace_lock(config.lock_path()):
        state = load_state(config)
        result = dry_run_submission_candidate(config, state, candidate_id)
        write_reports(config, state)
        save_state(config, state)
        return result


def plan_submission(config: WorkspaceConfig) -> dict[str, Any]:
    with workspace_lock(config.lock_path()):
        state = load_state(config)
        result = plan_submission_slots(config, state)
        write_reports(config, state)
        save_state(config, state)
        return result


def list_ready_work_items(config: WorkspaceConfig) -> list[str]:
    state = load_state(config)
    return [item.id for item in runnable_work_items(config, state)]
