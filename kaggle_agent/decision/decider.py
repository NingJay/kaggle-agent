from __future__ import annotations

import json
from pathlib import Path

import yaml

from kaggle_agent.adapters.command import parse_json_payload, run_command_adapter
from kaggle_agent.decision.helpers import load_run_result
from kaggle_agent.knowledge import write_experiment_conclusions
from kaggle_agent.schema import DecisionRecord, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text, now_utc_iso


def _submission_threshold(config: WorkspaceConfig, experiment_config_path: str) -> float:
    config_path = Path(experiment_config_path)
    if not config_path.is_absolute():
        config_path = config.root / config_path
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return float(payload.get("metrics", {}).get("submission_candidate_threshold", 0.75))


def make_decision(config: WorkspaceConfig, state: WorkspaceState, run_id: str) -> DecisionRecord:
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    result = load_run_result(run)
    output_path = config.artifact_path("decisions", f"{run.run_id}.json")
    if config.adapters.decision_command.strip():
        text = run_command_adapter(
            config.adapters.decision_command,
            stage="decision",
            workspace_root=config.root,
            input_path=Path(run.research_summary_path),
            output_path=output_path,
            extra_env={
                "KAGGLE_AGENT_RUN_ID": run.run_id,
                "KAGGLE_AGENT_EXPERIMENT_ID": experiment.id,
            },
        )
        payload = parse_json_payload(text)
    else:
        root_cause = str(result.get("root_cause", run.error or "unknown"))
        metric_value = run.primary_metric_value or 0.0
        threshold = _submission_threshold(config, experiment.config_path)
        if run.status == "failed":
            blocked = any(token in root_cause.lower() for token in ["missing", "not found", "tensorflow", "soundfile", "dependency"])
            payload = {
                "decision_type": "blocked" if blocked else "fix_root_cause",
                "next_action": "hold" if blocked else "run_new_experiment",
                "evidence_strength": "50",
                "root_cause": root_cause,
                "why": "The run failed; the next step should either unblock dependencies or retry with a narrower fix.",
                "next_experiment_title": "" if blocked else f"{experiment.title} root-cause fix",
                "next_experiment_family": experiment.family,
                "next_experiment_config": experiment.config_path,
                "launch_policy": "manual_review" if blocked else "auto",
                "submission_recommendation": "no",
            }
        elif experiment.family == "perch_head_debug":
            payload = {
                "decision_type": "promote_baseline",
                "next_action": "run_new_experiment",
                "evidence_strength": "80",
                "root_cause": str(result.get("root_cause", "debug run passed")),
                "why": "The smoke run succeeded, so the system should promote the first cached Perch probe baseline.",
                "next_experiment_title": "Perch cached-probe baseline",
                "next_experiment_family": "perch_cached_probe",
                "next_experiment_config": str(config.runtime_root() / "configs" / "default.yaml"),
                "launch_policy": "auto",
                "submission_recommendation": "no",
            }
        else:
            strong_score = metric_value >= threshold
            promising_score = metric_value >= 0.60
            payload = {
                "decision_type": "submit_candidate" if strong_score else ("tune_probe" if promising_score else "fix_probe_gap"),
                "next_action": "submit_candidate" if strong_score else "run_new_experiment",
                "evidence_strength": "85" if strong_score else ("70" if promising_score else "55"),
                "root_cause": str(result.get("root_cause", "baseline needs tuning")),
                "why": (
                    "The cached probe cleared the submission threshold and should be prepared as a submission candidate."
                    if strong_score
                    else (
                        "The cached probe baseline is promising but still below the target threshold; continue along the notebook-inspired probe grid."
                        if promising_score
                        else "The cached probe baseline is healthy but under target; prioritize fixes for sparse positives, priors, or stronger probe settings."
                    )
                ),
                "next_experiment_title": "" if strong_score else f"{experiment.title} tuned",
                "next_experiment_family": experiment.family,
                "next_experiment_config": experiment.config_path,
                "launch_policy": "auto",
                "submission_recommendation": "candidate" if strong_score else "no",
            }
    decision = DecisionRecord(
        decision_id=f"decision-{state.runtime.next_decision_number:04d}",
        source_run_id=run.run_id,
        experiment_id=experiment.id,
        decision_type=str(payload["decision_type"]),
        next_action=str(payload["next_action"]),
        evidence_strength=str(payload["evidence_strength"]),
        root_cause=str(payload["root_cause"]),
        why=str(payload["why"]),
        next_experiment_title=str(payload.get("next_experiment_title", "")),
        next_experiment_family=str(payload.get("next_experiment_family", "")),
        next_experiment_config=str(payload.get("next_experiment_config", "")),
        launch_policy=str(payload.get("launch_policy", "auto")),
        submission_recommendation=str(payload.get("submission_recommendation", "no")),
        created_at=now_utc_iso(),
    )
    state.runtime.next_decision_number += 1
    state.decisions.append(decision)
    atomic_write_text(output_path, json.dumps(decision.to_dict(), indent=2) + "\n")
    run.decision_record_path = str(output_path)
    experiment.latest_decision_id = decision.decision_id
    experiment.updated_at = now_utc_iso()
    write_experiment_conclusions(config, state)
    return decision
