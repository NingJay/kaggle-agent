from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    latest_stage_payload,
    run_configured_stage_adapter,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.knowledge import apply_knowledge_stage_outputs
from kaggle_agent.knowledge_reducer import record_constraints_from_decision, record_search_envelope
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import slugify


REQUIRED_PATTERN_BY_COMPONENT = {
    "class_coverage": "coverage_first",
    "pseudo_label": "pseudo_label_expansion",
    "probe_head": "probe_training_change",
    "optimization": "schedule_recovery",
    "backbone": "conditional_backbone_recovery",
}


def _submission_threshold(config: WorkspaceConfig, config_path: str) -> float:
    path = Path(config_path)
    if not path.is_absolute():
        path = config.root / path
    if not path.exists():
        return 0.75
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    metrics = payload.get("metrics", {})
    return float(metrics.get("submission_candidate_threshold", 0.75))


def _rule_components(policy_rules: list[dict[str, Any]], kinds: set[str]) -> list[str]:
    ordered: list[str] = []
    for item in policy_rules:
        if str(item.get("policy_type", "")) not in kinds:
            continue
        component = str(item.get("component", "") or "").strip()
        if component and component not in ordered:
            ordered.append(component)
    return ordered


def _portfolio_policy(config: WorkspaceConfig, research: dict[str, Any], family: str, run_id: str) -> dict[str, Any]:
    policy_rules = [item for item in research.get("policy_rules", []) if isinstance(item, dict)]
    capability_results = research.get("capability_results", {}) if isinstance(research.get("capability_results"), dict) else {}
    diversifier = capability_results.get("branch_diversifier", {})
    if isinstance(diversifier, dict) and diversifier.get("recommended_components"):
        components = [str(item) for item in diversifier.get("recommended_components", []) if str(item)]
    else:
        components = _rule_components(policy_rules, {"require", "prefer", "conditional"})
    negative_axes = _rule_components(policy_rules, {"veto", "avoid"})
    components = [item for item in components if item not in negative_axes] or ["class_coverage", "probe_head", "optimization"]
    branch_mix = (
        [item for item in diversifier.get("recommended_branch_mix", []) if isinstance(item, dict)]
        if isinstance(diversifier, dict)
        else []
    )
    if not branch_mix:
        roles = ["primary", "hedge", "explore"]
        branch_mix = [
            {"branch_role": roles[index], "target_component": component}
            for index, component in enumerate(components[: len(roles)])
        ]
    target_branch_count = int(diversifier.get("target_branch_count", 0) or 0) if isinstance(diversifier, dict) else 0
    if target_branch_count <= 0:
        target_branch_count = min(3, max(2, len(branch_mix)))
    return {
        "portfolio_id": f"portfolio-{run_id}-{slugify(family)}",
        "target_branch_count": target_branch_count,
        "branch_mix": branch_mix[:target_branch_count],
        "per_portfolio_cap": 1 if config.automation.max_active_runs <= 3 else 2,
        "per_idea_class_cap": 1,
        "dispatch_strategy": "prefer-diverse-frontier-before-follow-on-support",
        "selected_policy_cards": [str(item.get("rule_id", "")) for item in policy_rules[:6]],
        "deprioritized_axes": negative_axes,
        "branch_memory_ids": [str(item) for item in research.get("branch_memory_ids", [])][:6],
    }


def _forbidden_patterns(research: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    capability_results = research.get("capability_results", {}) if isinstance(research.get("capability_results"), dict) else {}
    veto_checker = capability_results.get("veto_checker", {})
    patterns = [str(item) for item in veto_checker.get("forbidden_patterns", []) if str(item)] if isinstance(veto_checker, dict) else []
    for axis in policy.get("deprioritized_axes", []):
        value = str(axis)
        if value and value not in patterns:
            patterns.append(value)
    return patterns


def _required_patterns(policy: dict[str, Any]) -> list[str]:
    patterns: list[str] = []
    for item in policy.get("branch_mix", []):
        if not isinstance(item, dict):
            continue
        pattern = REQUIRED_PATTERN_BY_COMPONENT.get(str(item.get("target_component", "")))
        if pattern and pattern not in patterns:
            patterns.append(pattern)
    return patterns


def _grounded_and_novel_slots(research: dict[str, Any], policy: dict[str, Any]) -> tuple[int, int]:
    repeated_failures = []
    capability_results = research.get("capability_results", {}) if isinstance(research.get("capability_results"), dict) else {}
    ledger = capability_results.get("ledger_miner", {})
    if isinstance(ledger, dict):
        repeated_failures = [str(item) for item in ledger.get("repeated_failures", []) if str(item)]
    target_branch_count = int(policy.get("target_branch_count", 2) or 2)
    novel_slots = 1 if target_branch_count >= 3 or repeated_failures else 0
    grounded_slots = max(1, target_branch_count - novel_slots)
    return grounded_slots, novel_slots


def _build_search_envelope(payload: dict[str, Any], *, default_turn_id: str) -> dict[str, Any]:
    existing = payload.get("search_envelope")
    if isinstance(existing, dict) and existing:
        envelope = dict(existing)
    else:
        portfolio_policy = payload.get("portfolio_policy", {})
        policy = dict(portfolio_policy) if isinstance(portfolio_policy, dict) else {}
        envelope = {
            "turn_id": default_turn_id,
            "portfolio_mode": str(payload.get("portfolio_mode", "") or ""),
            "slot_budget": int(payload.get("grounded_branch_slots", 0) or 0) + int(payload.get("novel_branch_slots", 0) or 0),
            "grounded_branch_slots": int(payload.get("grounded_branch_slots", 0) or 0),
            "novel_branch_slots": int(payload.get("novel_branch_slots", 0) or 0),
            "branch_budget_by_role": dict(payload.get("branch_budget_by_role", {}))
            if isinstance(payload.get("branch_budget_by_role"), dict)
            else {},
            "forbidden_patterns": [str(item) for item in payload.get("forbidden_plan_patterns", []) if str(item)],
            "required_patterns": [str(item) for item in payload.get("required_plan_patterns", []) if str(item)],
            "minimum_information_gain_bar": float(payload.get("minimum_information_gain_bar", 0.0) or 0.0),
            "per_portfolio_cap": int(policy.get("per_portfolio_cap", 1) or 1),
            "per_idea_class_cap": int(policy.get("per_idea_class_cap", 1) or 1),
            "dispatch_strategy": str(policy.get("dispatch_strategy", "") or ""),
            "max_budget_share": {"grounded": 0.85, "novel": 0.35},
            "smoke_only_first": True,
            "canary_eval_required": True,
            "novel_max_cost_tier": "medium",
        }
    envelope["turn_id"] = str(envelope.get("turn_id", "") or default_turn_id)
    return envelope


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
        envelope = _build_search_envelope(payload, default_turn_id=stage_run.stage_run_id)
        payload["search_envelope"] = envelope
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
        record_constraints_from_decision(
            state,
            run_id=run_id,
            stage_run_id=stage_run.stage_run_id,
            family=str(payload.get("next_family", "") or experiment.family),
            decision_payload=payload,
        )
        record_search_envelope(
            state,
            run_id=run_id,
            stage_run_id=stage_run.stage_run_id,
            family=str(payload.get("next_family", "") or experiment.family),
            envelope_payload=envelope,
        )
        apply_knowledge_stage_outputs(config, run_id=run_id, stage="decision", payload=payload)
        return stage_run

    threshold = _submission_threshold(config, experiment.config_path)
    family = experiment.family
    root_cause = str(report.get("root_cause", run.root_cause or run.error or "unknown"))
    policy = _portfolio_policy(config, research, family, run_id)
    forbidden_plan_patterns = _forbidden_patterns(research, policy)
    required_plan_patterns = _required_patterns(policy)
    grounded_branch_slots, novel_branch_slots = _grounded_and_novel_slots(research, policy)
    minimum_information_gain_bar = 0.6 if forbidden_plan_patterns else 0.48
    portfolio_mode = (
        "repair"
        if run.status == "failed"
        else ("submission_gate" if run.primary_metric_value is not None and run.primary_metric_value >= threshold else "frontier_search")
    )
    branch_budget_by_role: dict[str, int] = {}
    for item in policy.get("branch_mix", []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("branch_role", "") or "explore")
        branch_budget_by_role[role] = branch_budget_by_role.get(role, 0) + 1

    capability_results = research.get("capability_results", {}) if isinstance(research.get("capability_results"), dict) else {}
    novel = capability_results.get("novel_hypothesis_generator", {})
    branch_portfolio = [dict(item) for item in policy.get("branch_mix", []) if isinstance(item, dict)]
    if novel_branch_slots > 0 and isinstance(novel, dict) and str(novel.get("novel_component", "")):
        branch_portfolio.append(
            {
                "branch_role": "explore",
                "target_component": str(novel.get("novel_component", "")),
                "grounding_mode": "novel",
                "unsupported_claims": [str(item) for item in novel.get("unsupported_claims", []) if str(item)],
                "required_evidence": [str(item) for item in novel.get("required_evidence", []) if str(item)],
            }
        )

    common_fields = {
        "problem_frame": research.get("problem_frame", {}),
        "knowledge_card_ids": [str(item) for item in research.get("knowledge_card_ids", [])],
        "rejected_axes": [str(item) for item in research.get("negative_vetoes", []) if str(item)],
        "portfolio_intent": f"Allocate {grounded_branch_slots} grounded slots and {novel_branch_slots} novel slots around the current frontier.",
        "portfolio_mode": portfolio_mode,
        "portfolio_policy": policy,
        "branch_budget_by_role": branch_budget_by_role,
        "grounded_branch_slots": grounded_branch_slots,
        "novel_branch_slots": novel_branch_slots,
        "selected_policy_cards": [str(item) for item in policy.get("selected_policy_cards", [])],
        "branch_memory_ids": [str(item) for item in policy.get("branch_memory_ids", [])],
        "deprioritized_axes": [str(item) for item in policy.get("deprioritized_axes", [])],
        "forbidden_plan_patterns": forbidden_plan_patterns,
        "required_plan_patterns": required_plan_patterns,
        "minimum_information_gain_bar": minimum_information_gain_bar,
        "why_not": {
            str(item): f"deterministic constraint from reducer/capability state blocks `{item}` without explicit override"
            for item in forbidden_plan_patterns
        },
        "selected_capability_packs": [str(item) for item in research.get("selected_capability_packs", []) if str(item)],
        "selected_memory_files": [str(item) for item in research.get("selected_memory_files", []) if str(item)],
        "capability_results": capability_results,
        "open_questions": [str(item) for item in research.get("open_questions", []) if str(item)],
        "hypothesis_backlog": [dict(item) for item in research.get("hypothesis_backlog", []) if isinstance(item, dict)],
        "session_memory": research.get("session_memory", {}) if isinstance(research.get("session_memory"), dict) else {},
        "branch_mix": [dict(item) for item in policy.get("branch_mix", []) if isinstance(item, dict)],
        "branch_portfolio": branch_portfolio,
        "requires_human": False,
    }

    if run.status == "failed":
        payload = {
            "stage": "decision",
            "decision_type": "fix",
            "next_action": "run_new_experiment",
            "submission_recommendation": "no",
            "root_cause": root_cause,
            "why": "Repair the runtime or validation bottleneck before spending more portfolio budget.",
            "next_title": f"{experiment.title} runtime repair",
            "next_family": family,
            "next_config_path": experiment.config_path,
            "priority_delta": 5,
            "launch_mode": "background",
            **common_fields,
        }
    elif run.primary_metric_value is not None and run.primary_metric_value >= threshold:
        payload = {
            "stage": "decision",
            "decision_type": "submit_candidate",
            "next_action": "submit_candidate",
            "submission_recommendation": "candidate",
            "root_cause": root_cause,
            "why": "This run cleared the local submission threshold and should enter submission intelligence.",
            "next_title": experiment.title,
            "next_family": family,
            "next_config_path": experiment.config_path,
            "priority_delta": 0,
            "launch_mode": "background",
            **common_fields,
        }
    else:
        payload = {
            "stage": "decision",
            "decision_type": "tune",
            "next_action": "run_new_experiment",
            "submission_recommendation": "no",
            "root_cause": root_cause,
            "why": "Continue frontier search, but enforce grounded-vs-novel slot allocation and minimum information gain.",
            "next_title": f"{experiment.title} follow-up",
            "next_family": family,
            "next_config_path": experiment.config_path,
            "priority_delta": 10,
            "launch_mode": "background",
            **common_fields,
        }

    markdown = stage_markdown(
        f"Decision {run_id}",
        [
            f"- Decision type: `{payload['decision_type']}`",
            f"- Next action: `{payload['next_action']}`",
            f"- Submission recommendation: `{payload['submission_recommendation']}`",
            f"- Root cause: {payload['root_cause']}",
            f"- Why: {payload['why']}",
            f"- Portfolio mode: `{payload['portfolio_mode']}`",
            f"- Grounded slots: `{payload['grounded_branch_slots']}`",
            f"- Novel slots: `{payload['novel_branch_slots']}`",
            "",
            "## Branch Mix",
            *(
                f"- `{item.get('branch_role', '')}` -> `{item.get('target_component', '')}`"
                for item in payload.get("branch_mix", [])
                if isinstance(item, dict)
            ),
            "",
            "## Forbidden Patterns",
            *(f"- `{item}`" for item in payload.get("forbidden_plan_patterns", [])),
            "",
            "## Required Patterns",
            *(f"- `{item}`" for item in payload.get("required_plan_patterns", [])),
            "",
            "## Capability Packs",
            *(f"- `{item}`" for item in payload.get("selected_capability_packs", [])),
        ],
    )
    payload["search_envelope"] = _build_search_envelope(payload, default_turn_id=stage_run.stage_run_id)
    complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
    record_constraints_from_decision(
        state,
        run_id=run_id,
        stage_run_id=stage_run.stage_run_id,
        family=family,
        decision_payload=payload,
    )
    record_search_envelope(
        state,
        run_id=run_id,
        stage_run_id=stage_run.stage_run_id,
        family=family,
        envelope_payload=dict(payload.get("search_envelope", {})),
    )
    apply_knowledge_stage_outputs(config, run_id=run_id, stage="decision", payload=payload)
    return stage_run
