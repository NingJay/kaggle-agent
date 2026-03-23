from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from kaggle_agent.config import load_workspace_config
from kaggle_agent.control.monitor import maybe_start_next_run, process_completed_runs, tick_workspace, watch_workspace
from kaggle_agent.control.reporting import best_run, write_reports
from kaggle_agent.control.scheduler import queue_config_experiment, runnable_experiments
from kaggle_agent.control.store import initialize_workspace, load_state, save_state
from kaggle_agent.control.submission import build_submission_candidate
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import workspace_lock


def load_config(root: Path | None = None) -> WorkspaceConfig:
    return load_workspace_config(root)


def init_workspace(config: WorkspaceConfig, *, archive_legacy: bool = True, force: bool = False) -> WorkspaceState:
    with workspace_lock(config.lock_path()):
        state = initialize_workspace(config, archive_legacy=archive_legacy, force=force)
        write_reports(config, state)
        save_state(config, state)
        return state


def doctor_checks(config: WorkspaceConfig) -> list[tuple[bool, str, str]]:
    state = load_state(config)
    checks: list[tuple[bool, str, str]] = []
    default_config = config.runtime_root() / "configs" / "default.yaml"
    default_payload = _load_runtime_yaml(default_config)
    training_backend = str(default_payload.get("training", {}).get("backend", ""))
    model_provider = str(default_payload.get("model", {}).get("backbone_provider", ""))
    checks.append((config.root.exists(), "workspace_root", str(config.root)))
    checks.append((config.runtime_root().exists(), "runtime_root", str(config.runtime_root())))
    checks.append(((config.root / config.runtime.train_entrypoint).exists(), "train_entrypoint", str(config.root / config.runtime.train_entrypoint)))
    checks.append((Path(config.data.root).exists(), "data_root", config.data.root))
    checks.append(((Path(config.data.root) / config.data.train_csv).exists(), "train_csv", str(Path(config.data.root) / config.data.train_csv)))
    checks.append(((Path(config.data.root) / config.data.taxonomy_csv).exists(), "taxonomy_csv", str(Path(config.data.root) / config.data.taxonomy_csv)))
    checks.append(((Path(config.data.root) / config.data.sample_submission_csv).exists(), "sample_submission_csv", str(Path(config.data.root) / config.data.sample_submission_csv)))
    checks.append(
        (
            (Path(config.data.root) / config.data.train_soundscapes_labels_csv).exists(),
            "train_soundscapes_labels_csv",
            str(Path(config.data.root) / config.data.train_soundscapes_labels_csv),
        )
    )
    checks.append((importlib.util.find_spec("yaml") is not None, "pyyaml", "PyYAML importable"))
    checks.append((True, "research_adapter", config.adapters.research_command or "(internal fallback)"))
    checks.append((True, "decision_adapter", config.adapters.decision_command or "(internal fallback)"))
    checks.append((True, "planner_adapter", config.adapters.planner_command or "(internal fallback)"))
    checks.append((len(state.runtime.active_run_ids) <= config.automation.max_active_runs, "active_run_limit", ",".join(state.runtime.active_run_ids) or "none"))
    checks.append((bool(state.experiments), "seeded_experiments", str(len(state.experiments))))
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
        if config.data.perch_model_dir.strip():
            checks.append((Path(config.data.perch_model_dir).exists(), "perch_model_root_exists", config.data.perch_model_dir))
    return checks


def _load_runtime_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def get_status_state(config: WorkspaceConfig) -> WorkspaceState:
    with workspace_lock(config.lock_path()):
        state = load_state(config)
        write_reports(config, state)
        save_state(config, state)
        return state


def enqueue_config(config: WorkspaceConfig, config_path: str, *, title: str | None = None, family: str = "ad_hoc", priority: int = 50) -> WorkspaceState:
    with workspace_lock(config.lock_path()):
        state = load_state(config)
        queue_config_experiment(config, state, config_path, title=title, family=family, priority=priority)
        write_reports(config, state)
        save_state(config, state)
        return state


def start_next(config: WorkspaceConfig, *, background: bool = True) -> str | None:
    with workspace_lock(config.lock_path()):
        state = load_state(config)
        process_completed_runs(config, state)
        run, run_in_background = maybe_start_next_run(config, state, background=background)
        write_reports(config, state)
        save_state(config, state)
        if run is None:
            return None
        if not run_in_background:
            process_completed_runs(config, state)
            write_reports(config, state)
            save_state(config, state)
        return run.run_id


def tick(config: WorkspaceConfig, *, auto_start: bool = True) -> WorkspaceState:
    return tick_workspace(config, auto_start=auto_start)


def watch(config: WorkspaceConfig, *, interval_seconds: int, iterations: int, auto_start: bool = True) -> None:
    watch_workspace(config, interval_seconds=interval_seconds, iterations=iterations, auto_start=auto_start)


def build_submission(config: WorkspaceConfig, run_id: str | None = None) -> str:
    with workspace_lock(config.lock_path()):
        state = load_state(config)
        target_run = next((item for item in state.runs if item.run_id == run_id), None) if run_id else best_run(state)
        if target_run is None:
            raise ValueError("No eligible run available for submission candidate generation.")
        candidate = build_submission_candidate(config, state, target_run.run_id)
        write_reports(config, state)
        save_state(config, state)
        return candidate.id


def list_ready_experiments(config: WorkspaceConfig) -> list[str]:
    state = load_state(config)
    return [item.id for item in runnable_experiments(config, state)]
