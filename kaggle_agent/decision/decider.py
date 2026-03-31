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
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState


def _submission_threshold(config: WorkspaceConfig, config_path: str) -> float:
    path = Path(config_path)
    if not path.is_absolute():
        path = config.root / path
    if not path.exists():
        return 0.75
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    metrics = payload.get("metrics", {})
    return float(metrics.get("submission_candidate_threshold", 0.75))


def build_decision(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    research = latest_stage_payload(state, run_id, "research")
    report = latest_stage_payload(state, run_id, "report")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="decision",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    write_input_manifest(
        input_manifest_path,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
            "report": report,
            "research": research,
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

    threshold = _submission_threshold(config, experiment.config_path)
    root_cause = str(report.get("root_cause", run.root_cause or run.error or "unknown"))
    problem_frame = research.get("problem_frame", {}) if isinstance(research.get("problem_frame"), dict) else {}
    knowledge_card_ids = [str(item) for item in research.get("knowledge_card_ids", [])]
    rejected_axes = [str(item) for item in research.get("negative_priors", [])][:5]
    if run.status == "failed":
        decision_type = "blocked" if any(token in root_cause.lower() for token in ["missing", "module", "dependency"]) else "fix"
        payload = {
            "stage": "decision",
            "decision_type": decision_type,
            "next_action": "hold" if decision_type == "blocked" else "run_new_experiment",
            "submission_recommendation": "no",
            "root_cause": root_cause,
            "why": "The runtime failed and needs a direct repair before more exploration.",
            "next_title": f"{experiment.title} runtime repair",
            "next_family": experiment.family,
            "next_config_path": experiment.config_path,
            "priority_delta": -5 if decision_type == "blocked" else 5,
            "launch_mode": "sync" if decision_type == "blocked" else "background",
            "problem_frame": problem_frame,
            "knowledge_card_ids": knowledge_card_ids,
            "rejected_axes": rejected_axes,
            "branch_portfolio": [],
        }
    elif experiment.family == "perch_head_debug":
        payload = {
            "stage": "decision",
            "decision_type": "promote_baseline",
            "next_action": "run_new_experiment",
            "submission_recommendation": "no",
            "root_cause": root_cause,
            "why": "The debug smoke is healthy enough to promote the cached-probe baseline.",
            "next_title": "Perch cached-probe baseline",
            "next_family": "perch_cached_probe",
            "next_config_path": str(config.runtime_root() / "configs" / "default.yaml"),
            "priority_delta": 10,
            "launch_mode": "background",
            "problem_frame": problem_frame,
            "knowledge_card_ids": knowledge_card_ids,
            "rejected_axes": rejected_axes,
            "branch_portfolio": [],
        }
    elif run.primary_metric_value is not None and run.primary_metric_value >= threshold:
        payload = {
            "stage": "decision",
            "decision_type": "submit_candidate",
            "next_action": "submit_candidate",
            "submission_recommendation": "candidate",
            "root_cause": root_cause,
            "why": "This run cleared the local submission threshold and should enter the submission intelligence loop.",
            "next_title": "",
            "next_family": experiment.family,
            "next_config_path": experiment.config_path,
            "priority_delta": 0,
            "launch_mode": "background",
            "problem_frame": problem_frame,
            "knowledge_card_ids": knowledge_card_ids,
            "rejected_axes": rejected_axes,
            "branch_portfolio": [],
        }
    else:
        payload = {
            "stage": "decision",
            "decision_type": "tune",
            "next_action": "run_new_experiment",
            "submission_recommendation": "no",
            "root_cause": root_cause,
            "why": "The run is useful but still below the keep threshold, so the next config should directly target the root cause.",
            "next_title": f"{experiment.title} follow-up",
            "next_family": experiment.family,
            "next_config_path": experiment.config_path,
            "priority_delta": 10,
            "launch_mode": "background",
            "problem_frame": problem_frame,
            "knowledge_card_ids": knowledge_card_ids,
            "rejected_axes": rejected_axes,
            "branch_portfolio": [],
        }
    markdown = stage_markdown(
        f"Decision {run_id}",
        [
            f"- Decision type: `{payload['decision_type']}`",
            f"- Next action: `{payload['next_action']}`",
            f"- Submission recommendation: `{payload['submission_recommendation']}`",
            f"- Root cause: {payload['root_cause']}",
            f"- Why: {payload['why']}",
        ],
    )
    complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
    return stage_run
