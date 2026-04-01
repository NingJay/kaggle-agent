from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from kaggle_agent.control.lifecycle import resolve_lifecycle_template, resolve_stage_plan, resolve_target_run_id
from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    latest_stage_payload,
    run_configured_stage_adapter,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.knowledge import compact_knowledge_bundle, retrieve_knowledge_bundle
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text, now_utc_iso, slugify


DEFAULT_BRANCH_PLAN_LIMIT = 3
ROLE_WEIGHTS = {
    "primary": 8.0,
    "aspiration": 8.0,
    "hedge": 5.0,
    "explore": 3.0,
    "support": 1.0,
}


def _branch_plan_limit() -> int:
    raw = os.environ.get("KAGGLE_AGENT_BRANCH_PLAN_LIMIT", "").strip()
    if not raw:
        return DEFAULT_BRANCH_PLAN_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_BRANCH_PLAN_LIMIT
    return max(1, min(value, 8))


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


def _relative_config_path(config: WorkspaceConfig, path: Path) -> str:
    return str(path.relative_to(config.root)) if path.is_relative_to(config.root) else str(path)


def _normalize_override_ops(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    overrides: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        overrides.append({"path": path, "value": item.get("value")})
    return overrides


def _apply_override_ops(config_payload: dict[str, Any], overrides: list[dict[str, Any]]) -> dict[str, Any]:
    updated = copy.deepcopy(config_payload)
    for item in overrides:
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        parts = [part for part in path.split(".") if part]
        if not parts:
            continue
        target = updated
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[parts[-1]] = item.get("value")
    return updated


def _next_unused_probe_signature(
    candidates: list[tuple[int, float, int]],
    *,
    used: set[tuple[int, float, int]],
    already_selected: set[tuple[int, float, int]],
) -> tuple[int, float, int]:
    for candidate in candidates:
        if candidate in already_selected:
            continue
        if candidate not in used:
            return candidate
    for candidate in candidates:
        if candidate not in already_selected:
            return candidate
    return candidates[-1]


def _summaries_with_component(cards: list[dict[str, Any]], component: str, *, stances: set[str] | None = None) -> list[str]:
    summaries: list[str] = []
    for card in cards:
        if str(card.get("component", "")) != component:
            continue
        stance = str(card.get("stance", ""))
        if stances is not None and stance not in stances:
            continue
        summaries.append(f"{card.get('title', '')}: {card.get('summary', '')}")
    return summaries


def _bundle_cards(knowledge_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    cards = knowledge_bundle.get("cards", [])
    return [item for item in cards if isinstance(item, dict)] if isinstance(cards, list) else []


def _fallback_probe_branch_candidates(
    *,
    config: WorkspaceConfig,
    state: WorkspaceState,
    run_id: str,
    experiment,
    source_config: dict[str, Any],
    research: dict[str, Any],
    knowledge_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    used = _used_probe_signatures(config, state, experiment.family)
    selected: set[tuple[int, float, int]] = set()
    cards = _bundle_cards(knowledge_bundle)
    coverage_reason = _summaries_with_component(cards, "class_coverage", stances={"positive", "conditional"})
    calibration_veto = _summaries_with_component(cards, "prior_calibration", stances={"negative"})

    coverage_sig = _next_unused_probe_signature(
        [(64, 0.25, 4), (96, 0.25, 4), (96, 0.35, 4), (128, 0.35, 4)],
        used=used,
        already_selected=selected,
    )
    selected.add(coverage_sig)
    capacity_sig = _next_unused_probe_signature(
        [(96, 0.25, 8), (128, 0.25, 8), (128, 0.50, 8), (160, 0.50, 8)],
        used=used,
        already_selected=selected,
    )
    selected.add(capacity_sig)
    regularization_sig = _next_unused_probe_signature(
        [(64, 0.15, 4), (96, 0.15, 4), (128, 0.15, 4), (96, 0.25, 2)],
        used=used,
        already_selected=selected,
    )
    selected.add(regularization_sig)

    candidates = [
        {
            "title": f"{experiment.title} coverage-first pca{coverage_sig[0]}-c{coverage_sig[1]:.2f}-min{coverage_sig[2]}",
            "family": experiment.family,
            "hypothesis": "Improve holdout ROC-AUC by expanding minority-class coverage before calibration-only tuning.",
            "reason": "; ".join(coverage_reason[:2]) or "Coverage-related knowledge suggests long-tail fixes should outrank calibration-only tuning.",
            "branch_role": "primary",
            "idea_class": "class_coverage",
            "target_component": "probe_head",
            "priority_delta": 10,
            "launch_mode": "background",
            "config_overrides": [
                {"path": "training.probe_pca_dim", "value": coverage_sig[0]},
                {"path": "training.probe_c", "value": coverage_sig[1]},
                {"path": "training.probe_min_pos", "value": coverage_sig[2]},
            ],
            "knowledge_card_ids": [
                str(card.get("card_id", ""))
                for card in cards
                if str(card.get("component", "")) in {"class_coverage", "probe_head"}
            ][:4],
        },
        {
            "title": f"{experiment.title} representation-step pca{capacity_sig[0]}-c{capacity_sig[1]:.2f}-min{capacity_sig[2]}",
            "family": experiment.family,
            "hypothesis": "Increase probe representation capacity after a coverage-first branch to test whether hidden structure is underfit.",
            "reason": "Probe-head priors suggest representation capacity is still worth exploring once coverage is protected.",
            "branch_role": "hedge",
            "idea_class": "probe_head",
            "target_component": "probe_head",
            "priority_delta": 12,
            "launch_mode": "background",
            "config_overrides": [
                {"path": "training.probe_pca_dim", "value": capacity_sig[0]},
                {"path": "training.probe_c", "value": capacity_sig[1]},
                {"path": "training.probe_min_pos", "value": capacity_sig[2]},
            ],
            "knowledge_card_ids": [
                str(card.get("card_id", ""))
                for card in cards
                if str(card.get("component", "")) == "probe_head"
            ][:4],
        },
        {
            "title": f"{experiment.title} robust-regularization pca{regularization_sig[0]}-c{regularization_sig[1]:.2f}-min{regularization_sig[2]}",
            "family": experiment.family,
            "hypothesis": "Test whether a more conservative probe regularization setting improves holdout stability without collapsing rare classes.",
            "reason": (
                "Negative calibration priors were found, so this branch stays inside the probe training surface instead of doing post-hoc calibration."
                if calibration_veto
                else "Keep a low-risk structural hedge inside the probe training surface."
            ),
            "branch_role": "explore",
            "idea_class": "probe_regularization",
            "target_component": "probe_head",
            "priority_delta": 14,
            "launch_mode": "background",
            "config_overrides": [
                {"path": "training.probe_pca_dim", "value": regularization_sig[0]},
                {"path": "training.probe_c", "value": regularization_sig[1]},
                {"path": "training.probe_min_pos", "value": regularization_sig[2]},
            ],
            "knowledge_card_ids": [str(item) for item in knowledge_bundle.get("knowledge_card_ids", [])][:4],
        },
    ]
    return candidates[: _branch_plan_limit()]


def _fallback_generic_branch_candidates(
    *,
    experiment,
    source_config: dict[str, Any],
    knowledge_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    training = source_config.get("training", {})
    epochs = int(training.get("epochs", 1))
    learning_rate = training.get("learning_rate")
    cards = _bundle_cards(knowledge_bundle)
    lr_value = float(learning_rate) if isinstance(learning_rate, (int, float)) else 0.001
    candidates = [
        {
            "title": f"{experiment.title} longer-schedule",
            "family": experiment.family,
            "hypothesis": "Extend training schedule before changing downstream calibration so the model has a chance to realize existing priors.",
            "reason": "General fallback branch that increases schedule depth rather than doing post-hoc tuning only.",
            "branch_role": "primary",
            "idea_class": "optimization",
            "target_component": "training",
            "priority_delta": 10,
            "launch_mode": "background",
            "config_overrides": [{"path": "training.epochs", "value": max(epochs + 2, 3)}],
            "knowledge_card_ids": [str(item.get("card_id", "")) for item in cards[:3]],
        },
        {
            "title": f"{experiment.title} lower-lr-stability",
            "family": experiment.family,
            "hypothesis": "Lower the learning rate to improve holdout stability if the current branch looks noisy.",
            "reason": "Fallback hedge branch for training instability.",
            "branch_role": "hedge",
            "idea_class": "optimization",
            "target_component": "training",
            "priority_delta": 12,
            "launch_mode": "background",
            "config_overrides": [{"path": "training.learning_rate", "value": lr_value * 0.5}],
            "knowledge_card_ids": [str(item.get("card_id", "")) for item in cards[:3]],
        },
        {
            "title": f"{experiment.title} higher-lr-recovery",
            "family": experiment.family,
            "hypothesis": "Raise the learning rate moderately to test whether the branch is under-updating.",
            "reason": "Fallback exploration branch for underfitting.",
            "branch_role": "explore",
            "idea_class": "optimization",
            "target_component": "training",
            "priority_delta": 14,
            "launch_mode": "background",
            "config_overrides": [{"path": "training.learning_rate", "value": lr_value * 1.25}],
            "knowledge_card_ids": [str(item.get("card_id", "")) for item in cards[:3]],
        },
    ]
    return candidates[: _branch_plan_limit()]


def _fallback_branch_candidates(
    *,
    config: WorkspaceConfig,
    state: WorkspaceState,
    run_id: str,
    experiment,
    source_config: dict[str, Any],
    research: dict[str, Any],
    knowledge_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    backend = str(source_config.get("training", {}).get("backend", ""))
    if backend == "sklearn_cached_probe":
        return _fallback_probe_branch_candidates(
            config=config,
            state=state,
            run_id=run_id,
            experiment=experiment,
            source_config=source_config,
            research=research,
            knowledge_bundle=knowledge_bundle,
        )
    return _fallback_generic_branch_candidates(
        experiment=experiment,
        source_config=source_config,
        knowledge_bundle=knowledge_bundle,
    )


def _policy_cards(research: dict[str, Any], decision: dict[str, Any], knowledge_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for source in (
        research.get("policy_cards", []),
        knowledge_bundle.get("policy_cards", []),
    ):
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            card_id = str(item.get("card_id", ""))
            if not card_id or card_id in seen:
                continue
            seen.add(card_id)
            merged.append(dict(item))
    selected_ids = {str(item) for item in decision.get("selected_policy_cards", []) if str(item)}
    if selected_ids:
        merged.sort(key=lambda item: (str(item.get("card_id", "")) not in selected_ids, -float(item.get("confidence", 0.0) or 0.0)))
    return merged


def _branch_memories(research: dict[str, Any], decision: dict[str, Any], knowledge_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for source in (
        research.get("branch_memories", []),
        knowledge_bundle.get("branch_memories", []),
    ):
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get("memory_id", ""))
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)
            merged.append(dict(item))
    selected_ids = {str(item) for item in decision.get("branch_memory_ids", []) if str(item)}
    if selected_ids:
        merged.sort(key=lambda item: (str(item.get("memory_id", "")) not in selected_ids, -abs(float(item.get("signal_score", 0.0) or 0.0))))
    return merged


def _portfolio_policy(decision: dict[str, Any]) -> dict[str, Any]:
    value = decision.get("portfolio_policy")
    return dict(value) if isinstance(value, dict) else {}


def _combined_branch_inputs(
    *,
    payload: dict[str, Any],
    decision: dict[str, Any],
    fallback_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for source in (
        payload.get("candidate_branches", []),
        payload.get("branch_plans", []),
        decision.get("branch_portfolio", []),
    ):
        if not isinstance(source, list):
            continue
        combined.extend(item for item in source if isinstance(item, dict))
    if payload and not combined and str(payload.get("plan_status", "")) == "planned":
        combined.append(payload)
    if not combined:
        combined.extend(fallback_candidates)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in combined:
        key = (
            str(item.get("title", "")),
            str(item.get("idea_class") or item.get("target_component") or ""),
            str(item.get("dedupe_key", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
    return deduped[: max(_branch_plan_limit() * 3, 6)]


def _evaluate_branch_candidate(
    branch_input: dict[str, Any],
    *,
    policy_cards: list[dict[str, Any]],
    branch_memories: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    branch = dict(branch_input)
    idea_class = str(branch.get("idea_class") or branch.get("target_component") or "")
    branch_role = str(branch.get("branch_role") or "explore")
    override_reason = str(branch.get("override_reason") or branch.get("policy_override_reason") or "").strip()
    deprioritized_axes = {str(item) for item in policy.get("deprioritized_axes", []) if str(item)}
    cooldown_axes = {str(item) for item in policy.get("cooldown_idea_classes", []) if str(item)}
    score = ROLE_WEIGHTS.get(branch_role, 0.0)
    policy_trace: list[str] = [f"role:{branch_role}:{score:.1f}"]
    veto_reasons: list[str] = []
    branch_memory_ids: list[str] = []

    for card in policy_cards:
        component = str(card.get("component", ""))
        if component != idea_class:
            continue
        policy_type = str(card.get("policy_type", "context"))
        confidence = float(card.get("confidence", 0.0) or 0.0)
        branch.setdefault("knowledge_card_ids", [])
        if str(card.get("card_id", "")) and str(card.get("card_id", "")) not in branch["knowledge_card_ids"]:
            branch["knowledge_card_ids"].append(str(card.get("card_id", "")))
        if policy_type == "require":
            score += 8.0 * confidence
        elif policy_type == "prefer":
            score += 4.0 * confidence
        elif policy_type == "conditional":
            score += 2.5 * confidence
        elif policy_type in {"avoid", "veto"}:
            reason = f"{policy_type}:{component}:{card.get('card_id', '')}"
            if override_reason:
                score -= 2.0 * confidence
                policy_trace.append(f"override:{reason}")
            else:
                veto_reasons.append(reason)
        policy_trace.append(f"policy:{policy_type}:{component}:{card.get('card_id', '')}")

    if idea_class in deprioritized_axes and not override_reason:
        veto_reasons.append(f"deprioritized-axis:{idea_class}")
    elif idea_class in deprioritized_axes:
        policy_trace.append(f"override:deprioritized-axis:{idea_class}")
        score -= 2.0

    if idea_class in cooldown_axes:
        score -= 5.0
        policy_trace.append(f"cooldown:{idea_class}")

    strong_memories = 0
    weak_memories = 0
    for memory in branch_memories:
        if str(memory.get("idea_class", "")) != idea_class:
            continue
        memory_id = str(memory.get("memory_id", ""))
        if memory_id:
            branch_memory_ids.append(memory_id)
        outcome = str(memory.get("outcome", ""))
        if outcome in {"leader", "improved", "submission_candidate"}:
            strong_memories += 1
        elif outcome in {"regressed", "critic_rejected", "run_failed", "validate_failed"}:
            weak_memories += 1
    if strong_memories:
        score += min(8.0, strong_memories * 2.5)
        policy_trace.append(f"memory-positive:{idea_class}:{strong_memories}")
    if weak_memories:
        score -= min(10.0, weak_memories * 3.0)
        policy_trace.append(f"memory-negative:{idea_class}:{weak_memories}")

    branch["idea_class"] = idea_class
    branch["branch_role"] = branch_role
    branch["policy_score"] = round(score, 3)
    branch["policy_trace"] = list(dict.fromkeys(policy_trace))
    branch["branch_memory_ids"] = list(dict.fromkeys(branch_memory_ids))
    branch["override_reason"] = override_reason
    branch["veto_reasons"] = veto_reasons
    branch["scheduler_hints"] = {
        "portfolio_cap": int(policy.get("per_portfolio_cap", 1) or 1),
        "idea_class_cap": int(policy.get("per_idea_class_cap", 1) or 1),
        "cooldown": idea_class in cooldown_axes,
        "dispatch_priority": round(score, 3),
    }
    return branch


def _prune_branch_candidates(
    branch_inputs: list[dict[str, Any]],
    *,
    policy_cards: list[dict[str, Any]],
    branch_memories: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    target_branch_count = int(policy.get("target_branch_count", _branch_plan_limit()) or _branch_plan_limit())
    target_branch_count = max(1, min(target_branch_count, _branch_plan_limit()))
    scored = [
        _evaluate_branch_candidate(
            branch_input,
            policy_cards=policy_cards,
            branch_memories=branch_memories,
            policy=policy,
        )
        for branch_input in branch_inputs
    ]
    scored.sort(
        key=lambda item: (
            bool(item.get("veto_reasons")),
            -float(item.get("policy_score", 0.0) or 0.0),
            int(item.get("priority_delta", 0) or 0),
            str(item.get("title", "")),
        )
    )

    selected: list[dict[str, Any]] = []
    pruned: list[dict[str, Any]] = []
    seen_idea_classes: set[str] = set()
    for item in scored:
        if item.get("veto_reasons"):
            pruned.append({**item, "pruned_reason": "policy_veto"})
            continue
        idea_class = str(item.get("idea_class", ""))
        if len(selected) < target_branch_count and idea_class and idea_class not in seen_idea_classes:
            selected.append(item)
            seen_idea_classes.add(idea_class)
            continue
        if len(selected) < target_branch_count and not idea_class:
            selected.append(item)
            continue
        pruned.append({**item, "pruned_reason": "portfolio_budget"})
    if len(selected) < target_branch_count:
        for item in scored:
            if item in selected or item.get("veto_reasons"):
                continue
            selected.append(item)
            if len(selected) >= target_branch_count:
                break
    selected = selected[:target_branch_count]
    overridden = [item for item in selected if str(item.get("override_reason", "")).strip()]
    policy_trace = list(
        dict.fromkeys(
            [
                *(f"card:{item.get('card_id', '')}:{item.get('policy_type', '')}" for item in policy_cards[:6]),
                *(trace for item in selected for trace in item.get("policy_trace", [])),
            ]
        )
    )
    return scored, selected, pruned, overridden, policy_trace


def _resolve_source_config_path(config: WorkspaceConfig, experiment, decision: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    next_config_path_value = str(decision.get("next_config_path") or "").strip()
    if next_config_path_value:
        next_config_path = Path(next_config_path_value)
        if not next_config_path.is_absolute():
            next_config_path = config.root / next_config_path
        candidates.append(next_config_path)

    experiment_config_path = Path(str(experiment.config_path))
    if not experiment_config_path.is_absolute():
        experiment_config_path = config.root / experiment_config_path
    candidates.append(experiment_config_path)

    default_config_path = config.runtime_root() / "configs" / "default.yaml"
    if default_config_path not in candidates:
        candidates.append(default_config_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _canonical_branch_plan(
    *,
    config: WorkspaceConfig,
    run,
    experiment,
    decision: dict[str, Any],
    source_config_path: Path,
    source_config: dict[str, Any],
    portfolio_id: str,
    branch_input: dict[str, Any],
    branch_rank: int,
    default_title: str,
    default_family: str,
    default_hypothesis: str,
    default_reason: str,
    default_launch_mode: str,
    default_priority: int,
    default_knowledge_ids: list[str],
) -> dict[str, Any]:
    title = str(branch_input.get("title") or default_title).strip() or default_title
    family = str(branch_input.get("family") or default_family).strip() or default_family
    hypothesis = str(branch_input.get("hypothesis") or default_hypothesis).strip() or default_hypothesis
    reason = str(branch_input.get("reason") or branch_input.get("rationale") or default_reason).strip() or default_reason
    branch_role = str(branch_input.get("branch_role") or ("primary" if branch_rank == 0 else "explore")).strip() or "explore"
    idea_class = str(branch_input.get("idea_class") or branch_input.get("target_component") or family).strip() or family
    launch_mode = str(branch_input.get("launch_mode") or default_launch_mode or "background").strip() or "background"
    priority_delta = int(branch_input.get("priority_delta", 0) or 0)
    priority = int(branch_input.get("priority") or (default_priority + priority_delta + branch_rank * 2))
    knowledge_card_ids = [str(item) for item in branch_input.get("knowledge_card_ids", [])] or list(default_knowledge_ids)
    policy_trace = [str(item) for item in branch_input.get("policy_trace", []) if str(item).strip()]
    branch_memory_ids = [str(item) for item in branch_input.get("branch_memory_ids", []) if str(item).strip()]
    scheduler_hints = dict(branch_input.get("scheduler_hints", {})) if isinstance(branch_input.get("scheduler_hints"), dict) else {}
    config_overrides = _normalize_override_ops(branch_input.get("config_overrides"))
    config_path_value = str(branch_input.get("config_path") or "").strip()

    config_path = Path(config_path_value) if config_path_value else source_config_path
    if not config_path.is_absolute():
        config_path = config.root / config_path
    config_payload = copy.deepcopy(source_config)
    if config_path.exists():
        config_payload = _config_payload(config_path)
    if config_overrides or not config_path.exists() or not config_path_value:
        config_payload = _apply_override_ops(config_payload, config_overrides)
        config_payload.setdefault("experiment", {})
        config_payload["experiment"]["name"] = slugify(title)
        generated_name = f"{slugify(title)}.yaml"
        generated_path = config.generated_config_root() / generated_name
        atomic_write_text(generated_path, yaml.safe_dump(config_payload, sort_keys=False))
        config_path = generated_path

    tags = [str(item) for item in branch_input.get("tags", []) if str(item).strip()]
    if not tags:
        tags = [experiment.family, "planned", "branch-search", branch_role, slugify(idea_class)]

    dedupe_key = str(branch_input.get("dedupe_key") or f"plan:{run.run_id}:{portfolio_id}:{branch_rank:02d}:{slugify(title)}")
    depends_on = [str(item) for item in branch_input.get("depends_on", [run.work_item_id])]
    work_type = str(branch_input.get("work_type") or "experiment_iteration")
    lifecycle_template = resolve_lifecycle_template(branch_input)
    stage_plan = resolve_stage_plan(lifecycle_template, strict=config.automation.strict_stage_graph)
    target_run_id = resolve_target_run_id(branch_input, lifecycle_template=lifecycle_template, default_run_id=run.run_id)

    return {
        "title": title,
        "family": family,
        "hypothesis": hypothesis,
        "reason": reason,
        "config_path": _relative_config_path(config, config_path),
        "priority": priority,
        "depends_on": depends_on,
        "tags": tags,
        "launch_mode": launch_mode,
        "dedupe_key": dedupe_key,
        "work_type": work_type,
        "lifecycle_template": lifecycle_template,
        "target_run_id": target_run_id,
        "stage_plan": stage_plan,
        "portfolio_id": portfolio_id,
        "idea_class": idea_class,
        "branch_role": branch_role,
        "branch_rank": branch_rank,
        "knowledge_card_ids": knowledge_card_ids,
        "policy_trace": policy_trace,
        "branch_memory_ids": branch_memory_ids,
        "scheduler_hints": scheduler_hints,
        "policy_score": float(branch_input.get("policy_score", 0.0) or 0.0),
        "config_overrides": config_overrides,
    }


def _canonicalize_plan_payload(
    *,
    config: WorkspaceConfig,
    state: WorkspaceState,
    run,
    experiment,
    decision: dict[str, Any],
    research: dict[str, Any],
    knowledge_bundle: dict[str, Any],
    source_config_path: Path,
    source_config: dict[str, Any],
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    next_action = str(decision.get("next_action", "hold"))
    if payload is None:
        payload = {}
    plan_status = str(payload.get("plan_status") or ("submission_candidate" if next_action == "submit_candidate" else ("planned" if next_action != "hold" else "hold")))
    reason = str(payload.get("reason") or decision.get("why") or f"Follow-up from {run.run_id}")
    if plan_status == "hold":
        return {
            "stage": "plan",
            "plan_status": "hold",
            "source_run_id": run.run_id,
            "reason": reason,
            "title": str(payload.get("title") or decision.get("next_title") or experiment.title),
            "family": str(payload.get("family") or decision.get("next_family") or experiment.family),
            "hypothesis": str(payload.get("hypothesis") or reason),
            "config_path": str(payload.get("config_path") or experiment.config_path),
            "priority": int(payload.get("priority") or experiment.priority),
            "depends_on": [run.work_item_id],
            "tags": [experiment.family, "hold"],
            "launch_mode": str(payload.get("launch_mode") or decision.get("launch_mode") or "background"),
            "dedupe_key": str(payload.get("dedupe_key") or f"plan:hold:{run.run_id}"),
            "work_type": str(payload.get("work_type") or "experiment_iteration"),
            "lifecycle_template": str(payload.get("lifecycle_template") or "recursive_experiment"),
            "target_run_id": str(payload.get("target_run_id") or ""),
            "stage_plan": resolve_stage_plan(str(payload.get("lifecycle_template") or "recursive_experiment"), strict=config.automation.strict_stage_graph),
            "portfolio_id": "",
            "knowledge_card_ids": [str(item) for item in knowledge_bundle.get("knowledge_card_ids", [])],
            "problem_frame": knowledge_bundle.get("problem_frame", {}),
            "candidate_branches": [],
            "branch_plans": [],
            "pruned_branches": [],
            "overridden_branches": [],
            "policy_trace": [],
            "scheduler_hints": {},
        }
    if plan_status == "submission_candidate":
        return {
            "stage": "plan",
            "plan_status": "submission_candidate",
            "source_run_id": run.run_id,
            "reason": reason,
            "title": str(payload.get("title") or decision.get("next_title") or experiment.title),
            "family": str(payload.get("family") or decision.get("next_family") or experiment.family),
            "hypothesis": str(payload.get("hypothesis") or reason),
            "config_path": str(payload.get("config_path") or experiment.config_path),
            "priority": int(payload.get("priority") or experiment.priority),
            "depends_on": [run.work_item_id],
            "tags": [experiment.family, "submission_candidate"],
            "launch_mode": str(payload.get("launch_mode") or decision.get("launch_mode") or "background"),
            "dedupe_key": str(payload.get("dedupe_key") or f"plan:submission:{run.run_id}"),
            "work_type": str(payload.get("work_type") or "experiment_iteration"),
            "lifecycle_template": str(payload.get("lifecycle_template") or "recursive_experiment"),
            "target_run_id": str(payload.get("target_run_id") or ""),
            "stage_plan": resolve_stage_plan(str(payload.get("lifecycle_template") or "recursive_experiment"), strict=config.automation.strict_stage_graph),
            "portfolio_id": "",
            "knowledge_card_ids": [str(item) for item in knowledge_bundle.get("knowledge_card_ids", [])],
            "problem_frame": knowledge_bundle.get("problem_frame", {}),
            "candidate_branches": [],
            "branch_plans": [],
            "pruned_branches": [],
            "overridden_branches": [],
            "policy_trace": [],
            "scheduler_hints": {},
        }

    fallback_candidates = _fallback_branch_candidates(
        config=config,
        state=state,
        run_id=run.run_id,
        experiment=experiment,
        source_config=source_config,
        research=research,
        knowledge_bundle=knowledge_bundle,
    )
    branch_inputs = _combined_branch_inputs(payload=payload, decision=decision, fallback_candidates=fallback_candidates)
    portfolio_id = str(payload.get("portfolio_id") or f"portfolio-{run.run_id}-{slugify(experiment.family)}")
    default_title = str(payload.get("title") or decision.get("next_title") or f"{experiment.title} follow-up")
    default_family = str(payload.get("family") or decision.get("next_family") or experiment.family)
    default_hypothesis = str(payload.get("hypothesis") or decision.get("why") or f"Follow-up from {run.run_id}")
    default_priority = int(payload.get("priority") or (experiment.priority + int(decision.get("priority_delta", 10) or 10)))
    default_launch_mode = str(payload.get("launch_mode") or decision.get("launch_mode") or "background")
    default_knowledge_ids = [str(item) for item in payload.get("knowledge_card_ids", [])] or [
        str(item) for item in knowledge_bundle.get("knowledge_card_ids", [])
    ]
    policy = _portfolio_policy(decision)
    policy_cards = _policy_cards(research, decision, knowledge_bundle)
    branch_memories = _branch_memories(research, decision, knowledge_bundle)
    candidate_branches, selected_branch_inputs, pruned_branches, overridden_branches, policy_trace = _prune_branch_candidates(
        branch_inputs,
        policy_cards=policy_cards,
        branch_memories=branch_memories,
        policy=policy,
    )

    branch_plans = [
        _canonical_branch_plan(
            config=config,
            run=run,
            experiment=experiment,
            decision=decision,
            source_config_path=source_config_path,
            source_config=source_config,
            portfolio_id=portfolio_id,
            branch_input=branch_input,
            branch_rank=index,
            default_title=default_title if index == 0 else f"{default_title} branch {index + 1}",
            default_family=default_family,
            default_hypothesis=default_hypothesis,
            default_reason=reason,
            default_launch_mode=default_launch_mode,
            default_priority=default_priority,
            default_knowledge_ids=default_knowledge_ids,
        )
        for index, branch_input in enumerate(selected_branch_inputs)
    ]
    if not branch_plans and candidate_branches:
        fallback_primary = candidate_branches[0]
        branch_plans = [
            _canonical_branch_plan(
                config=config,
                run=run,
                experiment=experiment,
                decision=decision,
                source_config_path=source_config_path,
                source_config=source_config,
                portfolio_id=portfolio_id,
                branch_input=fallback_primary,
                branch_rank=0,
                default_title=default_title,
                default_family=default_family,
                default_hypothesis=default_hypothesis,
                default_reason=reason,
                default_launch_mode=default_launch_mode,
                default_priority=default_priority,
                default_knowledge_ids=default_knowledge_ids,
            )
        ]
    primary = branch_plans[0]
    return {
        "stage": "plan",
        "plan_status": "planned",
        "source_run_id": run.run_id,
        "reason": primary["reason"],
        "title": primary["title"],
        "family": primary["family"],
        "hypothesis": primary["hypothesis"],
        "config_path": primary["config_path"],
        "priority": primary["priority"],
        "depends_on": primary["depends_on"],
        "tags": primary["tags"],
        "launch_mode": primary["launch_mode"],
        "dedupe_key": primary["dedupe_key"],
        "work_type": primary["work_type"],
        "lifecycle_template": primary["lifecycle_template"],
        "target_run_id": primary["target_run_id"],
        "stage_plan": primary["stage_plan"],
        "portfolio_id": portfolio_id,
        "knowledge_card_ids": primary["knowledge_card_ids"],
        "problem_frame": knowledge_bundle.get("problem_frame", {}),
        "candidate_branches": candidate_branches,
        "branch_plans": branch_plans,
        "pruned_branches": pruned_branches,
        "overridden_branches": overridden_branches,
        "policy_trace": policy_trace,
        "scheduler_hints": {
            "per_portfolio_cap": int(policy.get("per_portfolio_cap", 1) or 1),
            "per_idea_class_cap": int(policy.get("per_idea_class_cap", 1) or 1),
            "target_branch_count": len(branch_plans),
            "dispatch_strategy": str(policy.get("dispatch_strategy", "prefer-diverse-frontier-before-follow-on-support")),
        },
        "created_at": now_utc_iso(),
    }


def build_plan(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    decision = latest_stage_payload(state, run_id, "decision")
    research = latest_stage_payload(state, run_id, "research")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="plan",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    knowledge_bundle = retrieve_knowledge_bundle(
        config,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
            "decision": decision,
            "research": research,
        },
        stage="plan",
        state=state,
    )
    write_input_manifest(
        input_manifest_path,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
            "decision": decision,
            "research": research,
            "retrieved_knowledge": compact_knowledge_bundle(knowledge_bundle),
        },
    )
    source_config_path = _resolve_source_config_path(config, experiment, decision)
    source_config = _config_payload(source_config_path)
    source_config.setdefault("experiment", {})
    source_config.setdefault("training", {})
    source_config.setdefault("data", {})
    adapted = run_configured_stage_adapter(
        config,
        state,
        stage_run,
        input_manifest_path=input_manifest_path,
        extra_env={"KAGGLE_AGENT_RUN_ID": run_id},
    )
    adapted_payload = adapted[0] if adapted is not None else None
    payload = _canonicalize_plan_payload(
        config=config,
        state=state,
        run=run,
        experiment=experiment,
        decision=decision,
        research=research,
        knowledge_bundle=knowledge_bundle,
        source_config_path=source_config_path,
        source_config=source_config,
        payload=adapted_payload,
    )

    spec_yaml_path = Path(stage_run.output_dir) / "spec.yaml"
    atomic_write_text(spec_yaml_path, yaml.safe_dump(payload, sort_keys=False))
    stage_run.spec_path = str(spec_yaml_path)
    branch_plans = payload.get("branch_plans", [])
    markdown_lines = [
        f"- Status: `{payload['plan_status']}`",
        f"- Title: {payload['title']}",
        f"- Config: `{payload['config_path']}`",
        f"- Launch mode: `{payload['launch_mode']}`",
        f"- Lifecycle: `{payload.get('lifecycle_template', '') or 'n/a'}`",
        f"- Stage plan: `{' -> '.join(payload.get('stage_plan', [])) or 'n/a'}`",
        f"- Target run: `{payload.get('target_run_id', '') or 'n/a'}`",
        f"- Hypothesis: {payload['hypothesis']}",
    ]
    if isinstance(branch_plans, list) and branch_plans:
        markdown_lines.extend(["", "## Branch Portfolio"])
        for branch in branch_plans:
            if not isinstance(branch, dict):
                continue
            markdown_lines.append(
                f"- `{branch.get('branch_role', 'branch')}` | {branch.get('title', '')} | idea={branch.get('idea_class', '')} | lifecycle=`{branch.get('lifecycle_template', '')}` | target_run=`{branch.get('target_run_id', '') or 'n/a'}` | stages=`{' -> '.join(branch.get('stage_plan', []))}` | score={branch.get('policy_score', 0.0)} | config=`{branch.get('config_path', '')}`"
            )
    pruned = payload.get("pruned_branches")
    if isinstance(pruned, list) and pruned:
        markdown_lines.extend(["", "## Pruned Branches"])
        for branch in pruned:
            if not isinstance(branch, dict):
                continue
            markdown_lines.append(
                f"- {branch.get('title', '')} | reason={branch.get('pruned_reason', '')} | vetoes={', '.join(str(item) for item in branch.get('veto_reasons', [])) or 'n/a'}"
            )
    overrides = payload.get("overridden_branches")
    if isinstance(overrides, list) and overrides:
        markdown_lines.extend(["", "## Overrides"])
        for branch in overrides:
            if not isinstance(branch, dict):
                continue
            markdown_lines.append(f"- {branch.get('title', '')} | override={branch.get('override_reason', '')}")
    policy_trace = payload.get("policy_trace")
    if isinstance(policy_trace, list) and policy_trace:
        markdown_lines.extend(["", "## Policy Trace", *(f"- `{item}`" for item in policy_trace)])
    markdown = stage_markdown(f"Plan {run_id}", markdown_lines)
    complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
    return stage_run
