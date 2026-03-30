from __future__ import annotations

from pathlib import Path

from kaggle_agent.decision.helpers import latest_stage_payload
from kaggle_agent.decision.helpers import load_run_result
from kaggle_agent.layout import visible_runs
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text, ensure_directory


def ensure_knowledge_layout(config: WorkspaceConfig) -> None:
    ensure_directory(config.knowledge_root())
    ensure_directory(config.knowledge_path("research"))
    ensure_directory(config.knowledge_path("papers"))


def read_knowledge_context(config: WorkspaceConfig) -> str:
    ensure_knowledge_layout(config)
    parts: list[str] = []
    for path in sorted(config.knowledge_root().rglob("*.md")):
        if not path.is_file():
            continue
        relative = path.relative_to(config.knowledge_root())
        parts.append(f"## {relative}\n\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(part for part in parts if part.strip())


def write_experiment_conclusions(config: WorkspaceConfig, state: WorkspaceState) -> Path:
    ensure_knowledge_layout(config)
    completed_runs = [run for run in visible_runs(state) if run.status in {"succeeded", "failed"}]
    completed_runs.sort(key=lambda item: (item.completed_at, item.run_id))

    lines = ["# Experiment Conclusions", ""]
    if not completed_runs:
        lines.append("- No completed experiments yet.")
    for run in completed_runs:
        result = load_run_result(run)
        decision = latest_stage_payload(state, run.run_id, "decision")
        verdict = str(result.get("verdict", "unknown"))
        root_cause = str(decision.get("root_cause") or result.get("root_cause", run.error or "unknown"))
        metric = "-" if run.primary_metric_value is None else f"{run.primary_metric_value:.6f}"
        experiment_id = run.experiment_id
        lines.extend(
            [
                f"## {run.run_id}",
                f"- Experiment: `{experiment_id}`",
                f"- Best AUC: {metric}",
                f"- Root cause: {root_cause}",
                f"- Verdict: {verdict}",
                "",
            ]
        )
    path = config.knowledge_root() / "experiment_conclusions.md"
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")
    return path
