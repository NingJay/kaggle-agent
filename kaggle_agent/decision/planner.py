from __future__ import annotations

from pathlib import Path

import yaml

from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    latest_stage_payload,
    run_configured_stage_adapter,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.knowledge import read_knowledge_context
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text, now_utc_iso, slugify


def _config_payload(path: Path) -> dict:
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
    signatures: set[tuple[int, float, int]] = set()
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
            signatures.add(signature)
    return signatures


def _next_probe_variant(used: set[tuple[int, float, int]], knowledge_context: str) -> tuple[int, float, int]:
    ordered = [
        (64, 0.25, 8),
        (64, 0.50, 8),
        (64, 0.25, 4),
        (96, 0.25, 8),
        (96, 0.50, 4),
        (128, 0.50, 4),
    ]
    if "long tail" in knowledge_context.lower() or "proxy" in knowledge_context.lower():
        ordered = [(64, 0.25, 4), (96, 0.25, 4), *ordered]
    for candidate in ordered:
        if candidate not in used:
            return candidate
    return ordered[-1]


def _relative_config_path(config: WorkspaceConfig, path: Path) -> str:
    return str(path.relative_to(config.root)) if path.is_relative_to(config.root) else str(path)


def build_plan(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    decision = latest_stage_payload(state, run_id, "decision")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="plan",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    knowledge_context = read_knowledge_context(config)
    write_input_manifest(
        input_manifest_path,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
            "decision": decision,
            "knowledge_context": knowledge_context,
        },
    )
    adapted = run_configured_stage_adapter(
        config,
        state,
        stage_run,
        input_manifest_path=input_manifest_path,
        extra_env={"KAGGLE_AGENT_RUN_ID": run_id},
    )
    if adapted is not None:
        payload, markdown = adapted
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
        return stage_run

    next_action = str(decision.get("next_action", "hold"))
    if next_action == "hold":
        payload = {
            "stage": "plan",
            "plan_status": "hold",
            "source_run_id": run_id,
            "reason": decision.get("why", "No automatic action scheduled."),
        }
        markdown = stage_markdown(f"Plan {run_id}", [f"- Status: `hold`", f"- Reason: {payload['reason']}"])
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
        return stage_run

    if next_action == "submit_candidate":
        payload = {
            "stage": "plan",
            "plan_status": "submission_candidate",
            "source_run_id": run_id,
            "reason": decision.get("why", "Promote into submission loop."),
        }
        markdown = stage_markdown(
            f"Plan {run_id}",
            [f"- Status: `submission_candidate`", f"- Reason: {payload['reason']}"],
        )
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
        return stage_run

    source_config_path = Path(str(decision.get("next_config_path", experiment.config_path)))
    if not source_config_path.is_absolute():
        source_config_path = config.root / source_config_path
    source_config = _config_payload(source_config_path)
    source_config.setdefault("experiment", {})
    source_config.setdefault("training", {})
    source_config.setdefault("data", {})

    title = str(decision.get("next_title") or f"{experiment.title} follow-up")
    family = str(decision.get("next_family") or experiment.family)
    hypothesis = str(decision.get("why") or f"Follow-up from {run_id}")
    source_config["experiment"]["name"] = slugify(title)

    if source_config.get("training", {}).get("backend") == "sklearn_cached_probe":
        used = _used_probe_signatures(config, state, family)
        probe_pca_dim, probe_c, probe_min_pos = _next_probe_variant(used, knowledge_context)
        source_config["training"]["probe_pca_dim"] = probe_pca_dim
        source_config["training"]["probe_c"] = probe_c
        source_config["training"]["probe_min_pos"] = probe_min_pos
        title = f"{title} pca{probe_pca_dim}-c{probe_c:.2f}-min{probe_min_pos}"
        source_config["experiment"]["name"] = slugify(title)
    else:
        source_config["training"]["epochs"] = max(int(source_config["training"].get("epochs", 1)), 3)
        if "learning_rate" in source_config["training"]:
            source_config["training"]["learning_rate"] = float(source_config["training"]["learning_rate"]) * 0.5

    generated_name = f"{slugify(title)}.yaml"
    generated_path = config.generated_config_root() / generated_name
    atomic_write_text(generated_path, yaml.safe_dump(source_config, sort_keys=False))

    payload = {
        "stage": "plan",
        "plan_status": "planned",
        "source_run_id": run_id,
        "created_at": now_utc_iso(),
        "title": title,
        "family": family,
        "hypothesis": hypothesis,
        "config_path": _relative_config_path(config, generated_path),
        "priority": experiment.priority + int(decision.get("priority_delta", 10)),
        "depends_on": [run.work_item_id],
        "tags": [experiment.family, "planned", "v2"],
        "launch_mode": str(decision.get("launch_mode", "background")),
        "dedupe_key": (
            "seed:perch-baseline"
            if experiment.family == "perch_head_debug" and family == "perch_cached_probe"
            else f"decision:{run_id}:{slugify(title)}"
        ),
        "work_type": "experiment_iteration",
    }
    spec_yaml_path = Path(stage_run.output_dir) / "spec.yaml"
    atomic_write_text(spec_yaml_path, yaml.safe_dump(payload, sort_keys=False))
    stage_run.spec_path = str(spec_yaml_path)
    markdown = stage_markdown(
        f"Plan {run_id}",
        [
            f"- Status: `planned`",
            f"- Title: {title}",
            f"- Config: `{payload['config_path']}`",
            f"- Launch mode: `{payload['launch_mode']}`",
            f"- Hypothesis: {hypothesis}",
        ],
    )
    complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
    return stage_run
