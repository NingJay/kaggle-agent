from __future__ import annotations

import json
from typing import Any

from kaggle_agent.schema import (
    BranchMemoryRecord,
    ClaimRecord,
    ConstraintRecord,
    EvidenceLinkRecord,
    ObservationAtomRecord,
    PolicyRuleRecord,
    SearchEnvelopeRecord,
    WorkspaceState,
)
from kaggle_agent.utils import now_utc_iso, slugify, truncate


POSITIVE_OUTCOMES = {"leader", "improved", "submission_candidate"}
NEGATIVE_OUTCOMES = {"regressed", "critic_rejected", "run_failed", "validate_failed"}


def _find_by_id(rows: list[Any], key: str, attr: str):
    return next((item for item in rows if getattr(item, attr) == key), None)


def _claim_identity(claim: ClaimRecord) -> str:
    payload = {
        "subject": claim.subject,
        "predicate": claim.predicate,
        "metric_key": claim.metric_key,
        "scope_vector": dict(claim.scope_vector or {}),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _merge_strings(values: list[str], additions: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*values, *additions]:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def upsert_observation_atom(
    state: WorkspaceState,
    *,
    run_id: str,
    stage_name: str,
    family: str,
    component: str,
    summary: str,
    source_type: str,
    source_ref: str,
    metric_name: str = "",
    comparator: str = "",
    value: float | None = None,
    direction: str = "",
    axis_tags: list[str] | None = None,
) -> ObservationAtomRecord:
    observation_id = f"observation-{run_id}-{slugify(stage_name)}-{slugify(source_type)}-{slugify(component or summary)[:40]}"
    record = _find_by_id(state.observations, observation_id, "observation_id")
    payload = ObservationAtomRecord(
        observation_id=observation_id,
        run_id=run_id,
        stage_name=stage_name,
        family=family,
        component=component,
        metric_name=metric_name,
        comparator=comparator,
        value=value,
        direction=direction,
        axis_tags=list(axis_tags or []),
        summary=truncate(summary, limit=320),
        source_type=source_type,
        source_ref=source_ref,
        created_at=record.created_at if record is not None else now_utc_iso(),
    )
    if record is None:
        state.observations.append(payload)
        return payload
    record.family = payload.family
    record.component = payload.component
    record.metric_name = payload.metric_name
    record.comparator = payload.comparator
    record.value = payload.value
    record.direction = payload.direction
    record.axis_tags = payload.axis_tags
    record.summary = payload.summary
    record.source_type = payload.source_type
    record.source_ref = payload.source_ref
    return record


def _stance_from_branch_memory(memory: BranchMemoryRecord) -> str:
    if memory.outcome in POSITIVE_OUTCOMES:
        return "positive"
    if memory.outcome in NEGATIVE_OUTCOMES:
        return "negative"
    if memory.outcome == "flat":
        return "conditional"
    return "general"


def _scope_key(family: str, component: str) -> str:
    return f"{family or '*'}::{component or 'general'}"


def _scope_vector(*, family: str, component: str, metric_key: str = "", condition_tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "family": family,
        "evaluation_domain": component,
        "metric": metric_key,
        "condition_tags": list(condition_tags or []),
    }


def _claim_key(*, subject: str, predicate: str, scope_vector: dict[str, Any], metric_key: str = "") -> dict[str, Any]:
    return {
        "subject": subject,
        "predicate": predicate,
        "scope_vector": dict(scope_vector),
        "metric_key": metric_key,
    }


def _upsert_claim(state: WorkspaceState, claim: ClaimRecord) -> ClaimRecord:
    existing = _find_by_id(state.claims, claim.claim_id, "claim_id")
    if existing is None:
        claim_identity = _claim_identity(claim)
        existing = next((item for item in state.claims if _claim_identity(item) == claim_identity), None)
    if existing is None:
        state.claims.append(claim)
        return claim
    existing.family = claim.family or existing.family
    existing.component = claim.component or existing.component
    existing.stance = claim.stance or existing.stance
    existing.summary = claim.summary if len(claim.summary) >= len(existing.summary) else existing.summary
    existing.subject = claim.subject or existing.subject
    existing.predicate = claim.predicate or existing.predicate
    existing.metric_key = claim.metric_key or existing.metric_key
    existing.claim_key = claim.claim_key or existing.claim_key
    existing.scope_vector = claim.scope_vector or existing.scope_vector
    existing.title = claim.title or existing.title
    existing.source_type = claim.source_type or existing.source_type
    existing.source_tier = claim.source_tier or existing.source_tier
    existing.source_ref = claim.source_ref or existing.source_ref
    existing.scope_key = claim.scope_key or existing.scope_key
    existing.claim_kind = claim.claim_kind or existing.claim_kind
    existing.claim_status = claim.claim_status or existing.claim_status
    existing.confidence = max(existing.confidence, claim.confidence)
    new_support_ids = [item for item in claim.support_ids if item not in existing.support_ids]
    new_contradict_ids = [item for item in claim.contradict_ids if item not in existing.contradict_ids]
    if claim.empirical_support_count:
        existing.empirical_support_count += len(new_support_ids) if claim.support_ids else claim.empirical_support_count
    if claim.empirical_contradict_count:
        existing.empirical_contradict_count += len(new_contradict_ids) if claim.contradict_ids else claim.empirical_contradict_count
    if claim.seed_support_count:
        existing.seed_support_count += len(new_support_ids) if claim.support_ids else claim.seed_support_count
    if claim.seed_contradict_count:
        existing.seed_contradict_count += len(new_contradict_ids) if claim.contradict_ids else claim.seed_contradict_count
    existing.support_ids = _merge_strings(existing.support_ids, claim.support_ids)
    existing.contradict_ids = _merge_strings(existing.contradict_ids, claim.contradict_ids)
    existing.scope_tags = _merge_strings(existing.scope_tags, claim.scope_tags)
    existing.override_required = existing.override_required or claim.override_required
    existing.updated_at = claim.updated_at
    return existing


def seed_claims_from_cards(state: WorkspaceState, cards: list[dict[str, Any]]) -> None:
    timestamp = now_utc_iso()
    for card in cards:
        if not isinstance(card, dict):
            continue
        stance = str(card.get("stance", "") or "general")
        if stance not in {"positive", "negative", "conditional"}:
            continue
        component = str(card.get("component", "") or "general")
        scope_vector = _scope_vector(family="", component=component, metric_key="")
        claim = ClaimRecord(
            claim_id=f"claim-seed-{str(card.get('card_id', ''))}",
            family="",
            component=component,
            stance=stance,
            summary=str(card.get("summary", "")),
            subject=component,
            predicate=stance,
            metric_key="",
            claim_key=_claim_key(subject=component, predicate=stance, scope_vector=scope_vector),
            scope_vector=scope_vector,
            title=str(card.get("title", "")),
            source_type="curated_seed",
            source_tier="manual",
            source_ref=str(card.get("card_id", "")),
            scope_key=_scope_key("", component),
            claim_kind="knowledge_seed",
            claim_status="seed",
            confidence=min(0.45, float(card.get("confidence", 0.0) or 0.0)),
            seed_support_count=1,
            support_ids=[str(card.get("card_id", ""))],
            scope_tags=[str(card.get("source_path", "")), component],
            created_at=timestamp,
            updated_at=timestamp,
        )
        persisted = _upsert_claim(state, claim)
        record_evidence_link(
            state,
            claim_id=persisted.claim_id,
            run_id="",
            stage_name="knowledge_seed",
            source_type="curated_seed",
            source_ref=str(card.get("card_id", "")),
            polarity=stance,
            summary=str(card.get("summary", "")),
            source_tier="manual",
        )


def seed_claims_from_branch_memories(state: WorkspaceState) -> None:
    timestamp = now_utc_iso()
    for memory in state.branch_memories:
        stance = _stance_from_branch_memory(memory)
        if stance not in {"positive", "negative", "conditional"}:
            continue
        component = str(memory.idea_class or "general")
        scope_vector = _scope_vector(
            family=memory.family,
            component=component,
            metric_key=str(memory.metric_name or ""),
            condition_tags=[str(memory.branch_role or "")] if memory.branch_role else [],
        )
        claim = ClaimRecord(
            claim_id=f"claim-branch-{memory.memory_id}",
            family=memory.family,
            component=component,
            stance=stance,
            summary=memory.summary,
            subject=component,
            predicate=stance,
            metric_key=str(memory.metric_name or ""),
            claim_key=_claim_key(subject=component, predicate=stance, scope_vector=scope_vector, metric_key=str(memory.metric_name or "")),
            scope_vector=scope_vector,
            title=memory.run_id,
            source_type="branch_outcome",
            source_tier="empirical",
            source_ref=memory.memory_id,
            scope_key=_scope_key(memory.family, component),
            claim_kind="empirical_branch_outcome",
            claim_status="active",
            confidence=min(0.95, 0.45 + abs(float(memory.signal_score)) * 0.15),
            empirical_support_count=1,
            support_ids=[memory.memory_id],
            scope_tags=[memory.family, component, memory.branch_role],
            override_required=memory.outcome in NEGATIVE_OUTCOMES,
            created_at=memory.created_at or timestamp,
            updated_at=memory.updated_at or timestamp,
        )
        persisted = _upsert_claim(state, claim)
        record_evidence_link(
            state,
            claim_id=persisted.claim_id,
            run_id=memory.run_id,
            stage_name="branch_memory",
            source_type="branch_outcome",
            source_ref=memory.memory_id,
            polarity=stance,
            summary=memory.summary,
            source_tier="empirical",
        )


def synchronize_claims(state: WorkspaceState, cards: list[dict[str, Any]]) -> None:
    seed_claims_from_cards(state, cards)
    seed_claims_from_branch_memories(state)


def _policy_summary(component: str, policy_type: str, pos_emp: int, neg_emp: int, pos_seed: int, neg_seed: int) -> tuple[str, str]:
    component_name = component or "general"
    if policy_type in {"require", "prefer"}:
        summary = f"{component_name} has stronger supporting evidence than contradictory evidence."
        rationale = f"empirical(+{pos_emp}/-{neg_emp}), seeded(+{pos_seed}/-{neg_seed})"
    elif policy_type in {"veto", "avoid"}:
        summary = f"{component_name} has stronger contradictory evidence than supporting evidence."
        rationale = f"empirical(-{neg_emp}/+{pos_emp}), seeded(-{neg_seed}/+{pos_seed})"
    else:
        summary = f"{component_name} remains conditional because the evidence is mixed or incomplete."
        rationale = f"mixed evidence: empirical(+{pos_emp}/-{neg_emp}), seeded(+{pos_seed}/-{neg_seed})"
    return summary, rationale


def reduce_policy_rules(state: WorkspaceState) -> None:
    timestamp = now_utc_iso()
    grouped: dict[tuple[str, str], list[ClaimRecord]] = {}
    for claim in state.claims:
        if claim.claim_status in {"retired", "superseded"}:
            continue
        if not claim.component or claim.component == "general":
            continue
        grouped.setdefault((claim.family, claim.component), []).append(claim)
    retained_rule_ids: set[str] = set()
    for (family, component), claims in grouped.items():
        pos_emp = sum(1 for claim in claims if claim.stance == "positive" and claim.source_type == "branch_outcome")
        neg_emp = sum(1 for claim in claims if claim.stance == "negative" and claim.source_type == "branch_outcome")
        pos_seed = sum(1 for claim in claims if claim.stance == "positive" and claim.source_type == "curated_seed")
        neg_seed = sum(1 for claim in claims if claim.stance == "negative" and claim.source_type == "curated_seed")
        conditional = sum(1 for claim in claims if claim.stance == "conditional")
        positive_score = pos_emp * 2.5 + pos_seed * 0.75
        negative_score = neg_emp * 2.5 + neg_seed * 0.75
        contradictory = positive_score > 0 and negative_score > 0
        if contradictory or conditional:
            policy_type = "conditional"
        elif positive_score - negative_score >= 4.0:
            policy_type = "require" if pos_emp >= 2 else "prefer"
        elif negative_score - positive_score >= 4.0:
            policy_type = "veto" if neg_emp >= 2 else "avoid"
        elif positive_score > negative_score:
            policy_type = "prefer"
        elif negative_score > positive_score:
            policy_type = "avoid"
        else:
            continue
        rule_id = f"policy-{slugify(_scope_key(family, component))}"
        summary, rationale = _policy_summary(component, policy_type, pos_emp, neg_emp, pos_seed, neg_seed)
        confidence = 0.32 + (pos_emp + neg_emp) * 0.18 + (pos_seed + neg_seed) * 0.04
        if contradictory:
            confidence -= 0.12
        confidence = max(0.2, min(round(confidence, 3), 0.98))
        contradiction_ids = [claim.claim_id for claim in claims if claim.stance == "negative"] if policy_type in {"require", "prefer"} else [
            claim.claim_id for claim in claims if claim.stance == "positive"
        ]
        rule = PolicyRuleRecord(
            rule_id=rule_id,
            family=family,
            component=component,
            policy_type=policy_type,
            summary=summary,
            rationale=rationale,
            confidence=confidence,
            status="active",
            override_required=policy_type in {"require", "veto"} or contradictory,
            claim_ids=[claim.claim_id for claim in claims],
            contradiction_ids=contradiction_ids if contradictory else [],
            scope_tags=[component, *(["global"] if not family else [family])],
            created_at=timestamp,
            updated_at=timestamp,
        )
        existing = _find_by_id(state.policy_rules, rule_id, "rule_id")
        if existing is None:
            state.policy_rules.append(rule)
        else:
            existing.family = rule.family
            existing.component = rule.component
            existing.policy_type = rule.policy_type
            existing.summary = rule.summary
            existing.rationale = rule.rationale
            existing.confidence = rule.confidence
            existing.status = rule.status
            existing.override_required = rule.override_required
            existing.claim_ids = rule.claim_ids
            existing.contradiction_ids = rule.contradiction_ids
            existing.scope_tags = rule.scope_tags
            existing.updated_at = rule.updated_at
        retained_rule_ids.add(rule_id)
    for rule in state.policy_rules:
        if rule.rule_id not in retained_rule_ids:
            rule.status = "inactive"
            rule.updated_at = timestamp


def synchronize_policy_state(state: WorkspaceState, cards: list[dict[str, Any]]) -> None:
    synchronize_claims(state, cards)
    reduce_policy_rules(state)


def active_policy_rules(state: WorkspaceState, *, family: str = "") -> list[PolicyRuleRecord]:
    rows: list[PolicyRuleRecord] = []
    for rule in state.policy_rules:
        if rule.status != "active":
            continue
        if rule.family and family and rule.family != family:
            continue
        if rule.family and not family:
            continue
        rows.append(rule)
    rows.sort(key=lambda item: (-item.confidence, item.component, item.rule_id))
    return rows


def active_constraints(state: WorkspaceState, *, family: str = "", run_id: str = "") -> list[ConstraintRecord]:
    rows: list[ConstraintRecord] = []
    for constraint in state.constraints:
        if constraint.status != "active":
            continue
        if run_id and constraint.run_id != run_id:
            continue
        if constraint.family and family and constraint.family != family:
            continue
        rows.append(constraint)
    rows.sort(key=lambda item: (item.created_at, item.constraint_id))
    return rows


def record_evidence_link(
    state: WorkspaceState,
    *,
    claim_id: str,
    run_id: str,
    stage_name: str,
    source_type: str,
    source_ref: str,
    polarity: str,
    summary: str,
    source_tier: str,
) -> EvidenceLinkRecord:
    link_id = f"evidence-{claim_id}-{slugify(stage_name)}-{slugify(source_type)}-{slugify(source_ref)[:24]}"
    existing = _find_by_id(state.evidence_links, link_id, "link_id")
    payload = EvidenceLinkRecord(
        link_id=link_id,
        claim_id=claim_id,
        run_id=run_id,
        stage_name=stage_name,
        source_type=source_type,
        source_ref=source_ref,
        polarity=polarity,
        summary=truncate(summary, limit=320),
        source_tier=source_tier,
        created_at=existing.created_at if existing is not None else now_utc_iso(),
    )
    if existing is None:
        state.evidence_links.append(payload)
        return payload
    existing.run_id = payload.run_id
    existing.stage_name = payload.stage_name
    existing.source_type = payload.source_type
    existing.source_ref = payload.source_ref
    existing.polarity = payload.polarity
    existing.summary = payload.summary
    existing.source_tier = payload.source_tier
    return existing


def active_search_envelope(state: WorkspaceState, *, run_id: str = "", family: str = "") -> SearchEnvelopeRecord | None:
    envelopes = [
        item
        for item in state.search_envelopes
        if item.status == "active"
        and (not run_id or item.run_id == run_id)
        and (not family or item.family == family)
    ]
    if not envelopes:
        return None
    envelopes.sort(key=lambda item: (item.updated_at or item.created_at, item.envelope_id))
    return envelopes[-1]


def record_search_envelope(
    state: WorkspaceState,
    *,
    run_id: str,
    stage_run_id: str,
    family: str,
    envelope_payload: dict[str, Any],
) -> SearchEnvelopeRecord:
    timestamp = now_utc_iso()
    envelope_id = f"envelope-{run_id}"
    existing = _find_by_id(state.search_envelopes, envelope_id, "envelope_id")
    payload = SearchEnvelopeRecord(
        envelope_id=envelope_id,
        run_id=run_id,
        stage_run_id=stage_run_id,
        family=family,
        turn_id=str(envelope_payload.get("turn_id", "") or run_id),
        envelope=dict(envelope_payload),
        status="active",
        created_at=existing.created_at if existing is not None else timestamp,
        updated_at=timestamp,
    )
    if existing is None:
        state.search_envelopes.append(payload)
        return payload
    existing.stage_run_id = payload.stage_run_id
    existing.family = payload.family
    existing.turn_id = payload.turn_id
    existing.envelope = payload.envelope
    existing.status = payload.status
    existing.updated_at = payload.updated_at
    return existing


def record_constraints_from_decision(
    state: WorkspaceState,
    *,
    run_id: str,
    stage_run_id: str,
    family: str,
    decision_payload: dict[str, Any],
) -> list[str]:
    timestamp = now_utc_iso()
    specs = {
        "forbidden_patterns": {
            "scope": "branch_generation",
            "summary": "Forbidden plan patterns for the next portfolio.",
            "value": {"patterns": [str(item) for item in decision_payload.get("forbidden_plan_patterns", []) if str(item)]},
        },
        "required_patterns": {
            "scope": "branch_generation",
            "summary": "Required plan patterns for the next portfolio.",
            "value": {"patterns": [str(item) for item in decision_payload.get("required_plan_patterns", []) if str(item)]},
        },
        "minimum_information_gain_bar": {
            "scope": "branch_selection",
            "summary": "Minimum information-gain bar for selecting branches.",
            "value": {"threshold": float(decision_payload.get("minimum_information_gain_bar", 0.0) or 0.0)},
        },
        "branch_budget": {
            "scope": "branch_budget",
            "summary": "Grounded vs novel branch slot allocation.",
            "value": {
                "grounded_branch_slots": int(decision_payload.get("grounded_branch_slots", 0) or 0),
                "novel_branch_slots": int(decision_payload.get("novel_branch_slots", 0) or 0),
                "branch_budget_by_role": dict(decision_payload.get("branch_budget_by_role", {}))
                if isinstance(decision_payload.get("branch_budget_by_role"), dict)
                else {},
            },
        },
    }
    recorded: list[str] = []
    for constraint_type, spec in specs.items():
        constraint_id = f"constraint-{run_id}-{constraint_type}"
        existing = _find_by_id(state.constraints, constraint_id, "constraint_id")
        if existing is None:
            existing = ConstraintRecord(
                constraint_id=constraint_id,
                run_id=run_id,
                stage_run_id=stage_run_id,
                family=family,
                scope=str(spec["scope"]),
                constraint_type=constraint_type,
                summary=str(spec["summary"]),
                value=dict(spec["value"]),
                status="active",
                created_at=timestamp,
                updated_at=timestamp,
            )
            state.constraints.append(existing)
        else:
            existing.stage_run_id = stage_run_id
            existing.family = family
            existing.scope = str(spec["scope"])
            existing.summary = str(spec["summary"])
            existing.value = dict(spec["value"])
            existing.status = "active"
            existing.updated_at = timestamp
        recorded.append(existing.constraint_id)
    return recorded
