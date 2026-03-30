from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from kaggle_agent.schema import (
    AdapterConfig,
    AutomationConfig,
    CompetitionConfig,
    DataConfig,
    KaggleConfig,
    MetricsConfig,
    PathsConfig,
    RuntimeConfig,
    WorkspaceConfig,
)


WORKSPACE_FILE = "workspace.toml"


def load_workspace_config(root: Path | None = None) -> WorkspaceConfig:
    workspace_root = (root or Path.cwd()).resolve()
    config_path = workspace_root / WORKSPACE_FILE
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    competition = CompetitionConfig(
        contract=str(raw["competition"].get("contract", "birdclef_2026")),
        **{key: raw["competition"][key] for key in ["name", "slug", "url", "track", "description"]},
    )
    metrics = MetricsConfig(
        primary=raw["metrics"]["primary"],
        secondary=list(raw["metrics"].get("secondary", [])),
    )
    data = DataConfig(**raw["data"])
    paths_raw = dict(raw["paths"])
    paths = PathsConfig(
        state_dir=paths_raw["state_dir"],
        artifact_dir=paths_raw["artifact_dir"],
        legacy_dir=paths_raw["legacy_dir"],
        runtime_dir=paths_raw["runtime_dir"],
        knowledge_dir=str(paths_raw.get("knowledge_dir", "knowledge")),
        prompt_dir=str(paths_raw.get("prompt_dir", "prompts")),
        report_dir=str(paths_raw.get("report_dir", "reports")),
    )
    automation_raw = dict(raw["automation"])
    automation = AutomationConfig(
        monitor_interval_seconds=int(automation_raw["monitor_interval_seconds"]),
        report_interval_seconds=int(automation_raw["report_interval_seconds"]),
        submission_interval_hours=int(automation_raw["submission_interval_hours"]),
        default_timeout_minutes=int(automation_raw["default_timeout_minutes"]),
        max_active_runs=int(automation_raw["max_active_runs"]),
        auto_execute_plans=bool(automation_raw["auto_execute_plans"]),
        auto_start_planned_runs=bool(automation_raw["auto_start_planned_runs"]),
        strict_stage_graph=bool(automation_raw.get("strict_stage_graph", True)),
    )
    adapters_raw = dict(raw["adapters"])
    adapters = AdapterConfig(
        evidence_command=str(adapters_raw.get("evidence_command", "")),
        report_command=str(adapters_raw.get("report_command", "")),
        research_command=str(adapters_raw.get("research_command", "")),
        decision_command=str(adapters_raw.get("decision_command", "")),
        planner_command=str(adapters_raw.get("planner_command", "")),
        codegen_command=str(adapters_raw.get("codegen_command", "")),
        critic_command=str(adapters_raw.get("critic_command", "")),
        submission_command=str(adapters_raw.get("submission_command", "")),
    )
    runtime_raw = dict(raw["runtime"])
    runtime = RuntimeConfig(
        conda_env=str(runtime_raw["conda_env"]),
        shell_init=str(runtime_raw["shell_init"]),
        train_workdir=str(runtime_raw["train_workdir"]),
        train_entrypoint=str(runtime_raw["train_entrypoint"]),
        generated_config_dir=str(runtime_raw["generated_config_dir"]),
        seed_notebook_path=str(runtime_raw.get("seed_notebook_path", "")),
        allow_debug_preflight=bool(runtime_raw.get("allow_debug_preflight", False)),
    )
    kaggle_raw = dict(raw["kaggle"])
    dataset_sources = list(kaggle_raw.get("dataset_sources", []))
    if kaggle_raw.get("model_dataset_id") and not dataset_sources:
        dataset_sources.append(str(kaggle_raw["model_dataset_id"]))
    kaggle = KaggleConfig(
        username=str(kaggle_raw["username"]),
        model_dataset_id=str(kaggle_raw["model_dataset_id"]),
        kernel_slug=str(kaggle_raw["kernel_slug"]),
        enable_gpu=bool(kaggle_raw.get("enable_gpu", False)),
        enable_internet=bool(kaggle_raw.get("enable_internet", False)),
        is_private=bool(kaggle_raw.get("is_private", True)),
        cpu_submission_only=bool(kaggle_raw.get("cpu_submission_only", True)),
        scored_max_runtime_minutes=int(kaggle_raw.get("scored_max_runtime_minutes", 90)),
        max_daily_submissions=int(kaggle_raw.get("max_daily_submissions", 5)),
        max_final_submissions=int(kaggle_raw.get("max_final_submissions", 2)),
        dataset_sources=[str(item) for item in dataset_sources],
    )
    notes = list(raw.get("notes", []))
    return WorkspaceConfig(
        root=workspace_root,
        competition=competition,
        metrics=metrics,
        data=data,
        paths=paths,
        automation=automation,
        adapters=adapters,
        runtime=runtime,
        kaggle=kaggle,
        notes=notes,
    )
