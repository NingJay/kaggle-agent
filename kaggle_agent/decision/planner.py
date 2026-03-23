from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kaggle_agent.adapters.command import parse_json_payload, run_command_adapter
from kaggle_agent.knowledge import read_knowledge_context
from kaggle_agent.schema import DecisionRecord, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text, now_utc_iso, slugify


@dataclass(frozen=True)
class PlanResult:
    plan_path: Path
    payload: dict[str, Any]
    planned_experiment_payload: dict[str, Any] | None = None
    submission_candidate_requested: bool = False


def _default_launch_mode(decision: DecisionRecord) -> str:
    return "background" if decision.launch_policy == "auto" else "sync"


def _normalize_planned_experiment_payload(
    decision: DecisionRecord,
    experiment_priority: int,
    experiment_id: str,
    experiment_family: str,
    raw_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "title": str(raw_payload["title"]),
        "hypothesis": str(raw_payload.get("hypothesis") or f"Follow-up planned from {decision.decision_id}: {decision.why}"),
        "family": str(raw_payload.get("family") or decision.next_experiment_family or experiment_family),
        "config_path": str(raw_payload["config_path"]),
        "priority": int(raw_payload.get("priority", experiment_priority + 10)),
        "depends_on": [str(item) for item in raw_payload.get("depends_on", [experiment_id])],
        "tags": [str(item) for item in raw_payload.get("tags", [])],
        "launch_mode": str(raw_payload.get("launch_mode", _default_launch_mode(decision))),
        "dedupe_key": str(raw_payload.get("dedupe_key", f"decision:{decision.decision_id}:experiment")),
        "source_decision_id": decision.decision_id,
    }


def _config_payload(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _probe_signature(path: Path) -> tuple[int, float, int] | None:
    payload = _config_payload(path)
    training = payload.get("training", {})
    if training.get("backend") != "sklearn_cached_probe":
        return None
    return (
        int(training.get("probe_pca_dim", 32)),
        float(training.get("probe_c", 0.25)),
        int(training.get("probe_min_pos", 8)),
    )


def _used_probe_signatures(config: WorkspaceConfig, state: WorkspaceState, family: str) -> set[tuple[int, float, int]]:
    used: set[tuple[int, float, int]] = set()
    for experiment in state.experiments:
        if experiment.family != family:
            continue
        config_path = Path(experiment.config_path)
        if not config_path.is_absolute():
            config_path = config.root / config_path
        if not config_path.exists():
            continue
        signature = _probe_signature(config_path)
        if signature is not None:
            used.add(signature)
    return used


def _next_probe_variant(used: set[tuple[int, float, int]], knowledge_context: str, decision: DecisionRecord) -> tuple[int, float, int]:
    lower_context = f"{knowledge_context}\n{decision.root_cause}\n{decision.why}".lower()
    ordered = [
        (64, 0.25, 8),
        (64, 0.50, 8),
        (64, 0.25, 12),
        (96, 0.25, 8),
        (96, 0.50, 8),
        (64, 0.25, 4),
        (96, 0.25, 4),
        (128, 0.50, 4),
    ]
    if "sparse" in lower_context or "positive" in lower_context:
        ordered = [
            (64, 0.25, 4),
            (96, 0.25, 4),
            (128, 0.50, 4),
            *ordered,
        ]
    for candidate in ordered:
        if candidate not in used:
            return candidate
    return (128, 0.50, 4)


def build_plan(config: WorkspaceConfig, state: WorkspaceState, decision: DecisionRecord) -> PlanResult:
    run = next(item for item in state.runs if item.run_id == decision.source_run_id)
    experiment = next(item for item in state.experiments if item.id == decision.experiment_id)
    output_path = config.artifact_path("plans", f"{run.run_id}.json")
    if decision.next_action not in {"run_new_experiment", "submit_candidate"}:
        payload = {
            "status": "hold",
            "decision_id": decision.decision_id,
            "source_run_id": run.run_id,
            "reason": decision.why,
        }
        atomic_write_text(output_path, json.dumps(payload, indent=2) + "\n")
        run.plan_path = str(output_path)
        return PlanResult(plan_path=output_path, payload=payload)

    if decision.next_action == "submit_candidate":
        payload = {
            "status": "submission_candidate",
            "decision_id": decision.decision_id,
            "source_run_id": run.run_id,
        }
        atomic_write_text(output_path, json.dumps(payload, indent=2) + "\n")
        run.plan_path = str(output_path)
        return PlanResult(plan_path=output_path, payload=payload, submission_candidate_requested=True)

    if config.adapters.planner_command.strip():
        text = run_command_adapter(
            config.adapters.planner_command,
            stage="planner",
            workspace_root=config.root,
            input_path=Path(run.decision_record_path),
            output_path=output_path,
            extra_env={"KAGGLE_AGENT_DECISION_ID": decision.decision_id},
        )
        raw_payload = parse_json_payload(text)
        planned_payload = _normalize_planned_experiment_payload(
            decision,
            experiment.priority,
            experiment.id,
            experiment.family,
            raw_payload,
        )
        payload = {
            "status": "planned",
            "decision_id": decision.decision_id,
            "source_run_id": run.run_id,
            "created_at": now_utc_iso(),
            **planned_payload,
        }
        atomic_write_text(output_path, json.dumps(payload, indent=2) + "\n")
        run.plan_path = str(output_path)
        return PlanResult(plan_path=output_path, payload=payload, planned_experiment_payload=planned_payload)

    source_config_path = Path(decision.next_experiment_config or experiment.config_path)
    source_config_abs = source_config_path if source_config_path.is_absolute() else (config.root / source_config_path)
    source_config = yaml.safe_load(source_config_abs.read_text(encoding="utf-8"))
    source_config.setdefault("experiment", {})
    source_config.setdefault("training", {})
    source_config.setdefault("data", {})
    title = decision.next_experiment_title or f"{experiment.title} follow-up"
    family = decision.next_experiment_family or experiment.family
    source_config["experiment"]["name"] = slugify(title)
    knowledge_context = read_knowledge_context(config)
    if source_config.get("training", {}).get("backend") == "sklearn_cached_probe":
        used_signatures = _used_probe_signatures(config, state, family)
        probe_pca_dim, probe_c, probe_min_pos = _next_probe_variant(used_signatures, knowledge_context, decision)
        source_config["training"]["probe_pca_dim"] = probe_pca_dim
        source_config["training"]["probe_c"] = probe_c
        source_config["training"]["probe_min_pos"] = probe_min_pos
        title = f"{title} pca{probe_pca_dim}-c{probe_c:.2f}-min{probe_min_pos}"
        source_config["experiment"]["name"] = slugify(title)
    else:
        source_config["training"]["epochs"] = max(int(source_config["training"].get("epochs", 1)), 3)
        if "max_train_rows" in source_config["data"]:
            source_config["data"]["max_train_rows"] = max(int(source_config["data"]["max_train_rows"]), 32) * 4
        if "learning_rate" in source_config["training"]:
            source_config["training"]["learning_rate"] = float(source_config["training"]["learning_rate"]) * 0.5
    generated_name = f"{slugify(title)}.yaml"
    generated_path = config.generated_config_root() / generated_name
    atomic_write_text(generated_path, yaml.safe_dump(source_config, sort_keys=False))
    planned_payload = {
        "title": title,
        "hypothesis": f"Follow-up planned from {decision.decision_id}: {decision.why}",
        "family": family,
        "config_path": str(generated_path.relative_to(config.root)),
        "priority": experiment.priority + 10,
        "depends_on": [experiment.id],
        "tags": list(dict.fromkeys(experiment.tags + ["planned"])),
        "launch_mode": _default_launch_mode(decision),
        "dedupe_key": f"decision:{decision.decision_id}:experiment",
        "source_decision_id": decision.decision_id,
    }
    payload = {
        "status": "planned",
        "decision_id": decision.decision_id,
        "source_run_id": run.run_id,
        "created_at": now_utc_iso(),
        **planned_payload,
    }
    atomic_write_text(output_path, json.dumps(payload, indent=2) + "\n")
    run.plan_path = str(output_path)
    return PlanResult(plan_path=output_path, payload=payload, planned_experiment_payload=planned_payload)
