from __future__ import annotations

from typing import Any, Callable

from kaggle_agent.knowledge_reducer import active_constraints, active_policy_rules
from kaggle_agent.schema import CapabilityInvocationRecord, WorkspaceState
from kaggle_agent.utils import now_utc_iso


CapabilityFn = Callable[..., dict[str, Any]]


def _relevant_rules(state: WorkspaceState, family: str) -> list[dict[str, Any]]:
    rows = []
    for rule in active_policy_rules(state, family=family):
        rows.append(
            {
                "rule_id": rule.rule_id,
                "family": rule.family,
                "component": rule.component,
                "policy_type": rule.policy_type,
                "summary": rule.summary,
                "confidence": rule.confidence,
                "override_required": rule.override_required,
                "claim_ids": list(rule.claim_ids),
                "contradiction_ids": list(rule.contradiction_ids),
            }
        )
    return rows


def _ledger_miner(
    *,
    state: WorkspaceState,
    frame: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    family = str(frame.get("family", "") or "")
    by_component: dict[str, dict[str, int]] = {}
    for memory in state.branch_memories:
        if family and memory.family != family:
            continue
        component = str(memory.idea_class or "general")
        bucket = by_component.setdefault(component, {"positive": 0, "negative": 0, "other": 0})
        if memory.outcome in {"leader", "improved", "submission_candidate"}:
            bucket["positive"] += 1
        elif memory.outcome in {"regressed", "critic_rejected", "run_failed", "validate_failed"}:
            bucket["negative"] += 1
        else:
            bucket["other"] += 1
    ranked = sorted(
        (
            {
                "component": component,
                "positive": counts["positive"],
                "negative": counts["negative"],
                "net": counts["positive"] - counts["negative"],
            }
            for component, counts in by_component.items()
        ),
        key=lambda item: (-item["net"], -item["positive"], item["negative"], item["component"]),
    )
    return {
        "top_components": ranked[:5],
        "repeated_failures": [item["component"] for item in ranked if item["negative"] >= 2][:4],
        "strong_components": [item["component"] for item in ranked if item["positive"] > item["negative"]][:4],
    }


def _veto_checker(
    *,
    state: WorkspaceState,
    frame: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    family = str(frame.get("family", "") or "")
    rules = _relevant_rules(state, family)
    constraints = [
        {
            "constraint_id": row.constraint_id,
            "constraint_type": row.constraint_type,
            "value": dict(row.value),
        }
        for row in active_constraints(state, family=family, run_id=str(frame.get("run_id", "") or ""))
    ]
    forbidden_patterns: list[str] = []
    override_required_components: list[str] = []
    for rule in rules:
        if rule["policy_type"] in {"veto", "avoid"}:
            component = str(rule["component"])
            forbidden_patterns.append(component)
            if bool(rule["override_required"]):
                override_required_components.append(component)
    for constraint in constraints:
        if constraint["constraint_type"] == "forbidden_patterns":
            forbidden_patterns.extend(str(item) for item in constraint["value"].get("patterns", []) if str(item))
    return {
        "forbidden_patterns": list(dict.fromkeys(forbidden_patterns)),
        "override_required_components": list(dict.fromkeys(override_required_components)),
        "active_veto_rules": [rule for rule in rules if rule["policy_type"] in {"veto", "avoid"}],
    }


def _branch_diversifier(
    *,
    state: WorkspaceState,
    frame: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    family = str(frame.get("family", "") or "")
    rules = _relevant_rules(state, family)
    prioritized = [
        rule["component"]
        for rule in rules
        if rule["policy_type"] in {"require", "prefer", "conditional"}
    ]
    ordered = list(dict.fromkeys(prioritized))
    recommended_roles = ["primary", "hedge", "explore"]
    recommended_mix = [
        {"branch_role": recommended_roles[index], "target_component": component}
        for index, component in enumerate(ordered[: len(recommended_roles)])
    ]
    return {
        "recommended_components": ordered[:4],
        "recommended_branch_mix": recommended_mix,
        "target_branch_count": min(3, max(2, len(recommended_mix))) if recommended_mix else 2,
    }


def _submission_bar_checker(
    *,
    state: WorkspaceState,
    frame: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    family = str(frame.get("family", "") or "")
    leader = None
    for run in state.runs:
        if family and run.primary_metric_value is not None:
            experiment = next((item for item in state.experiments if item.id == run.experiment_id), None)
            if experiment is None or experiment.family != family:
                continue
            if leader is None or run.primary_metric_value > leader.primary_metric_value:
                leader = run
    leader_metric = leader.primary_metric_value if leader is not None else None
    ready = bool(leader_metric is not None and leader_metric >= 0.75)
    return {
        "leader_run_id": leader.run_id if leader is not None else "",
        "leader_metric": leader_metric,
        "submission_ready": ready,
        "summary": "Family is above the local submission threshold." if ready else "Family remains below the local submission threshold.",
    }


def _novel_hypothesis_generator(
    *,
    state: WorkspaceState,
    frame: dict[str, Any],
    capability_results: dict[str, dict[str, Any]] | None = None,
    **_: Any,
) -> dict[str, Any]:
    family = str(frame.get("family", "") or "")
    seen_components = {
        memory.idea_class
        for memory in state.branch_memories
        if not family or memory.family == family
    }
    candidates = ["pseudo_label", "class_coverage", "backbone", "preprocessing_aug", "optimization"]
    novel_component = next((component for component in candidates if component and component not in seen_components), "backbone")
    open_questions = []
    if capability_results and isinstance(capability_results.get("ledger_miner"), dict):
        repeated_failures = capability_results["ledger_miner"].get("repeated_failures", [])
        if repeated_failures:
            open_questions.append(f"Find a structurally new branch outside repeated failures: {', '.join(str(item) for item in repeated_failures)}.")
    open_questions.append(f"Test an unsupported but plausible {novel_component} hypothesis with explicit follow-up evidence.")
    return {
        "novel_component": novel_component,
        "unsupported_claims": [f"{novel_component} may unlock holdout improvements despite limited direct support."],
        "required_evidence": open_questions,
        "grounding_mode": "novel",
    }


CAPABILITY_PACKS: dict[str, CapabilityFn] = {
    "ledger_miner": _ledger_miner,
    "veto_checker": _veto_checker,
    "branch_diversifier": _branch_diversifier,
    "submission_bar_checker": _submission_bar_checker,
    "novel_hypothesis_generator": _novel_hypothesis_generator,
}


def invoke_capability_pack(
    state: WorkspaceState,
    *,
    run_id: str,
    stage_name: str,
    pack_id: str,
    input_summary: str,
    frame: dict[str, Any],
    capability_results: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if pack_id not in CAPABILITY_PACKS:
        raise KeyError(f"Unknown capability pack: {pack_id}")
    payload = CAPABILITY_PACKS[pack_id](
        state=state,
        frame=frame,
        capability_results=capability_results,
        **kwargs,
    )
    output_summary = str(payload.get("summary", "") or payload.get("novel_component", "") or pack_id)
    state.capability_invocations.append(
        CapabilityInvocationRecord(
            invocation_id=f"cap-{len(state.capability_invocations) + 1:04d}-{pack_id}",
            run_id=run_id,
            stage_name=stage_name,
            pack_id=pack_id,
            input_summary=input_summary,
            output_summary=output_summary,
            payload=dict(payload),
            created_at=now_utc_iso(),
        )
    )
    return payload
