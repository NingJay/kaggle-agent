from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kaggle_agent.decision.helpers import latest_stage_payload, load_run_result
from kaggle_agent.layout import visible_runs
from kaggle_agent.schema import BranchMemoryRecord, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_json, atomic_write_text, ensure_directory, now_utc_iso, slugify, truncate


SECTION_HEADING_RE = re.compile(r"^##+\s+(.*)$", re.MULTILINE)
TOKEN_RE = re.compile(r"[a-z0-9_]+")
POSITIVE_HINTS = (
    "biggest win",
    "works",
    "promising",
    "critical",
    "best",
    "validated",
    "outperform",
    "improvement",
    "improve",
    "load-bearing",
    "keep",
    "adopt",
)
NEGATIVE_HINTS = (
    "hurts",
    "hurt",
    "underperform",
    "underperforms",
    "didn't help",
    "did not help",
    "regress",
    "regression",
    "worse",
    "bad",
    "avoid",
    "reject",
    "blocked",
    "fail",
    "failed",
)
CONDITIONAL_HINTS = (
    "hypothesis",
    "maybe",
    "might",
    "if ",
    "unless",
    "open question",
    "condition",
    "expected",
    "still early",
    "needs",
)
COMPONENT_KEYWORDS = {
    "pseudo_label": ("pseudo", "teacher", "distillation", "kd", "soft label", "hard pseudo"),
    "class_coverage": ("imbalance", "coverage", "long tail", "min_pos", "rare", "class balance"),
    "probe_head": ("probe", "mlp", "embedding", "head", "pca", "logistic"),
    "prior_calibration": ("prior", "blend", "calibration", "temperature", "bayesian"),
    "preprocessing_aug": ("pcen", "specaugment", "shift", "augment", "mel", "normalization"),
    "backbone": ("backbone", "v2s", "b0", "encoder", "model"),
    "optimization": ("lr", "learning rate", "epochs", "scheduler", "optimizer"),
    "data_filtering": ("rating", "filter", "quality", "subset"),
}
SOURCE_PRIORITY_HINTS = {
    "00_experiment_rules.md": 3.0,
    "01_validated_findings.md": 5.0,
    "03_next_experiment_priors.md": 4.0,
    "04_submission_bar.md": 2.0,
    "experiment_conclusions.md": 3.0,
}


@dataclass(frozen=True)
class KnowledgeCard:
    card_id: str
    source_path: str
    title: str
    summary: str
    text: str
    stance: str
    component: str
    policy_type: str
    confidence: float
    applies_to: list[str] = field(default_factory=list)
    override_required: bool = False
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_knowledge_layout(config: WorkspaceConfig) -> None:
    ensure_directory(config.knowledge_root())
    ensure_directory(config.knowledge_path("research"))
    ensure_directory(config.knowledge_path("papers"))
    ensure_directory(config.knowledge_path("index"))


def _knowledge_root_from_workspace(workspace_root: Path) -> Path:
    return workspace_root / "knowledge"


def _index_root_from_workspace(workspace_root: Path) -> Path:
    return _knowledge_root_from_workspace(workspace_root) / "index"


def _section_summary(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip().lstrip("-*").strip()
        if candidate:
            return truncate(candidate, limit=220)
    return ""


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if len(token) >= 3]


def _keywords_for_card(source_path: str, title: str, text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in [*_tokenize(source_path.replace("/", " ")), *_tokenize(title), *_tokenize(text)]:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
        if len(ordered) >= 24:
            break
    return ordered


def _infer_stance(source_path: str, title: str, text: str) -> str:
    corpus = f"{source_path}\n{title}\n{text}".lower()
    positive_hits = sum(1 for hint in POSITIVE_HINTS if hint in corpus)
    negative_hits = sum(1 for hint in NEGATIVE_HINTS if hint in corpus)
    conditional_hits = sum(1 for hint in CONDITIONAL_HINTS if hint in corpus)
    if negative_hits > max(positive_hits, conditional_hits):
        return "negative"
    if positive_hits > max(negative_hits, conditional_hits):
        return "positive"
    if conditional_hits:
        return "conditional"
    if "validated" in source_path or "conclusions" in source_path:
        return "positive"
    return "general"


def _infer_component(source_path: str, title: str, text: str) -> str:
    corpus = f"{source_path}\n{title}\n{text}".lower()
    best_component = "general"
    best_score = 0
    for component, keywords in COMPONENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in corpus)
        if score > best_score:
            best_component = component
            best_score = score
    return best_component


def _confidence_for_card(source_path: str, title: str, text: str, stance: str) -> float:
    corpus = f"{source_path}\n{title}\n{text}".lower()
    score = 0.42
    score += min(0.24, SOURCE_PRIORITY_HINTS.get(Path(source_path).name, 0.0) * 0.04)
    if stance in {"positive", "negative"}:
        score += 0.16
    elif stance == "conditional":
        score += 0.08
    if "validated" in corpus or "biggest win" in corpus or "load-bearing" in corpus:
        score += 0.10
    if "regress" in corpus or "hurts" in corpus or "didn't help" in corpus or "did not help" in corpus:
        score += 0.10
    return max(0.35, min(0.95, round(score, 3)))


def _policy_type_for_card(stance: str, confidence: float) -> str:
    if stance == "negative":
        return "veto" if confidence >= 0.72 else "avoid"
    if stance == "positive":
        return "require" if confidence >= 0.82 else "prefer"
    if stance == "conditional":
        return "conditional"
    return "context"


def _applies_to_for_card(component: str, keywords: list[str]) -> list[str]:
    scope: list[str] = []
    if component and component != "general":
        scope.append(component)
    for keyword in keywords[:4]:
        if keyword not in scope:
            scope.append(keyword)
    return scope


def _iter_sections(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return [(path.stem.replace("_", " "), text)]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((match.group(1).strip(), body))
    return sections or [(path.stem.replace("_", " "), text)]


def _compile_knowledge_index(knowledge_root: Path) -> list[KnowledgeCard]:
    cards: list[KnowledgeCard] = []
    if not knowledge_root.exists():
        return cards
    for path in sorted(knowledge_root.rglob("*.md")):
        if not path.is_file():
            continue
        if "index" in path.parts:
            continue
        source_path = path.relative_to(knowledge_root).as_posix()
        for section_title, section_text in _iter_sections(path):
            summary = _section_summary(section_text) or section_title
            card_id = slugify(f"{source_path}-{section_title}")
            cards.append(
                KnowledgeCard(
                    card_id=card_id,
                    source_path=source_path,
                    title=section_title,
                    summary=summary,
                    text=truncate(section_text, limit=1200),
                    stance=_infer_stance(source_path, section_title, section_text),
                    component=_infer_component(source_path, section_title, section_text),
                    policy_type=_policy_type_for_card(
                        _infer_stance(source_path, section_title, section_text),
                        _confidence_for_card(
                            source_path,
                            section_title,
                            section_text,
                            _infer_stance(source_path, section_title, section_text),
                        ),
                    ),
                    confidence=_confidence_for_card(
                        source_path,
                        section_title,
                        section_text,
                        _infer_stance(source_path, section_title, section_text),
                    ),
                    applies_to=_applies_to_for_card(
                        _infer_component(source_path, section_title, section_text),
                        _keywords_for_card(source_path, section_title, section_text),
                    ),
                    override_required=_policy_type_for_card(
                        _infer_stance(source_path, section_title, section_text),
                        _confidence_for_card(
                            source_path,
                            section_title,
                            section_text,
                            _infer_stance(source_path, section_title, section_text),
                        ),
                    )
                    == "veto",
                    keywords=_keywords_for_card(source_path, section_title, section_text),
                )
            )
    return cards


def compile_knowledge_index(config: WorkspaceConfig) -> list[dict[str, Any]]:
    ensure_knowledge_layout(config)
    cards = _compile_knowledge_index(config.knowledge_root())
    atomic_write_json(config.knowledge_path("index", "cards.json"), [card.to_dict() for card in cards])
    return [card.to_dict() for card in cards]


def compile_knowledge_index_from_root(workspace_root: Path) -> list[dict[str, Any]]:
    knowledge_root = _knowledge_root_from_workspace(workspace_root)
    ensure_directory(knowledge_root / "index")
    cards = _compile_knowledge_index(knowledge_root)
    atomic_write_json(_index_root_from_workspace(workspace_root) / "cards.json", [card.to_dict() for card in cards])
    return [card.to_dict() for card in cards]


def build_problem_frame(manifest: dict[str, Any], *, stage: str = "") -> dict[str, Any]:
    run = manifest.get("run", {}) if isinstance(manifest.get("run"), dict) else {}
    experiment = manifest.get("experiment", {}) if isinstance(manifest.get("experiment"), dict) else {}
    report = manifest.get("report", {}) if isinstance(manifest.get("report"), dict) else {}
    research = manifest.get("research", {}) if isinstance(manifest.get("research"), dict) else {}
    decision = manifest.get("decision", {}) if isinstance(manifest.get("decision"), dict) else {}
    plan = manifest.get("plan", {}) if isinstance(manifest.get("plan"), dict) else {}

    family = str(plan.get("family") or decision.get("next_family") or experiment.get("family") or "")
    title = str(plan.get("title") or decision.get("next_title") or experiment.get("title") or "")
    root_cause = str(
        report.get("root_cause")
        or research.get("root_cause")
        or decision.get("root_cause")
        or run.get("root_cause")
        or run.get("error")
        or ""
    )
    hypothesis = str(plan.get("hypothesis") or decision.get("why") or "")
    metric_name = str(run.get("primary_metric_name") or "val_soundscape_macro_roc_auc")
    metric_value = run.get("primary_metric_value")
    focus = str(report.get("focus") or research.get("focus") or "")
    recent_rejects = [str(item) for item in research.get("reject", []) if str(item).strip()]
    adopt_now = [str(item) for item in research.get("adopt_now", []) if str(item).strip()]
    consider = [str(item) for item in research.get("consider", []) if str(item).strip()]
    query_terms = [family, title, root_cause, hypothesis, focus, *adopt_now[:3], *recent_rejects[:3], *consider[:2]]
    tokens = []
    seen: set[str] = set()
    for token in _tokenize(" ".join(query_terms)):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= 24:
            break
    return {
        "stage": stage or str(manifest.get("stage") or ""),
        "family": family,
        "title": title,
        "root_cause": root_cause,
        "hypothesis": hypothesis,
        "objective_metric": metric_name,
        "current_metric_value": metric_value,
        "focus": focus,
        "next_action": str(decision.get("next_action") or ""),
        "adopt_now": adopt_now[:6],
        "consider": consider[:6],
        "reject": recent_rejects[:6],
        "query_terms": tokens,
    }


def _score_card(card: dict[str, Any], frame: dict[str, Any], *, stage: str) -> float:
    score = 0.0
    query_terms = set(str(item) for item in frame.get("query_terms", []) if str(item))
    keywords = {str(item) for item in card.get("keywords", []) if str(item)}
    overlap = len(query_terms & keywords)
    score += overlap * 3.5

    component = str(card.get("component", ""))
    if component and component != "general":
        if component in " ".join(frame.get("query_terms", [])).lower():
            score += 4.0
        if component in str(frame.get("focus", "")).lower():
            score += 3.0

    source_path = str(card.get("source_path", ""))
    for filename, bonus in SOURCE_PRIORITY_HINTS.items():
        if source_path.endswith(filename):
            score += bonus

    stance = str(card.get("stance", "general"))
    if stage in {"plan", "decision"} and stance in {"positive", "negative", "conditional"}:
        score += 1.5
    if stage == "research" and stance in {"positive", "conditional"}:
        score += 1.0
    if str(card.get("policy_type", "")) in {"veto", "require"}:
        score += 2.0
    score += float(card.get("confidence", 0.0) or 0.0) * 3.0

    summary = f"{card.get('title', '')} {card.get('summary', '')}".lower()
    if "validation" in summary and "val" in str(frame.get("objective_metric", "")).lower():
        score += 2.0
    return score


def _select_diverse_cards(cards: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(cards) <= limit:
        return cards
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for stance in ("positive", "negative", "conditional"):
        for card in cards:
            if card.get("stance") != stance:
                continue
            card_id = str(card.get("card_id", ""))
            if card_id in seen_ids:
                continue
            selected.append(card)
            seen_ids.add(card_id)
            break
    for card in cards:
        if len(selected) >= limit:
            break
        card_id = str(card.get("card_id", ""))
        if card_id in seen_ids:
            continue
        selected.append(card)
        seen_ids.add(card_id)
    return selected[:limit]


def _compact_branch_memory(memory: BranchMemoryRecord) -> dict[str, Any]:
    return {
        "memory_id": memory.memory_id,
        "run_id": memory.run_id,
        "family": memory.family,
        "portfolio_id": memory.portfolio_id,
        "idea_class": memory.idea_class,
        "branch_role": memory.branch_role,
        "branch_rank": memory.branch_rank,
        "status": memory.status,
        "outcome": memory.outcome,
        "summary": memory.summary,
        "root_cause": memory.root_cause,
        "metric_name": memory.metric_name,
        "metric_value": memory.metric_value,
        "metric_delta": memory.metric_delta,
        "signal_score": memory.signal_score,
        "critic_status": memory.critic_status,
        "validation_status": memory.validation_status,
        "submission_status": memory.submission_status,
        "policy_tags": list(memory.policy_tags),
        "knowledge_card_ids": list(memory.knowledge_card_ids),
        "created_at": memory.created_at,
    }


def _score_branch_memory(memory: BranchMemoryRecord, frame: dict[str, Any], *, stage: str) -> float:
    score = abs(float(memory.signal_score)) * 1.5
    query_terms = set(str(item) for item in frame.get("query_terms", []) if str(item))
    text = " ".join(
        [
            memory.family,
            memory.idea_class,
            memory.branch_role,
            memory.summary,
            memory.root_cause,
            " ".join(memory.policy_tags),
        ]
    ).lower()
    memory_tokens = set(_tokenize(text))
    score += len(query_terms & memory_tokens) * 3.0
    if memory.family and memory.family == str(frame.get("family", "")):
        score += 6.0
    focus = str(frame.get("focus", "")).lower()
    if focus and focus in text:
        score += 3.0
    if stage in {"plan", "decision"} and memory.outcome in {"leader", "improved", "regressed", "critic_rejected"}:
        score += 2.0
    return score


def _recent_branch_memories(
    state: WorkspaceState,
    frame: dict[str, Any],
    *,
    stage: str,
    limit: int = 6,
) -> list[dict[str, Any]]:
    scored = sorted(
        (
            {
                **_compact_branch_memory(memory),
                "_score": _score_branch_memory(memory, frame, stage=stage),
            }
            for memory in state.branch_memories
        ),
        key=lambda item: (float(item.get("_score", 0.0)), str(item.get("created_at", "")), str(item.get("memory_id", ""))),
        reverse=True,
    )
    selected = [dict(item) for item in scored if float(item.get("_score", 0.0)) > 0.0][:limit]
    if not selected:
        selected = [dict(item) for item in scored[:limit]]
    for item in selected:
        item.pop("_score", None)
    return selected


def _policy_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for card in cards:
        policy_type = str(card.get("policy_type", "context"))
        if policy_type == "context":
            continue
        policies.append(
            {
                "card_id": str(card.get("card_id", "")),
                "title": str(card.get("title", "")),
                "summary": str(card.get("summary", "")),
                "source_path": str(card.get("source_path", "")),
                "stance": str(card.get("stance", "")),
                "component": str(card.get("component", "")),
                "policy_type": policy_type,
                "confidence": float(card.get("confidence", 0.0) or 0.0),
                "applies_to": [str(item) for item in card.get("applies_to", [])],
                "override_required": bool(card.get("override_required", False)),
            }
        )
    return policies


def _policy_contradictions(policy_cards: list[dict[str, Any]], branch_memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contradictions: list[dict[str, Any]] = []
    for card in policy_cards:
        component = str(card.get("component", ""))
        policy_type = str(card.get("policy_type", ""))
        if not component or component == "general":
            continue
        for memory in branch_memories:
            if str(memory.get("idea_class", "")) != component:
                continue
            outcome = str(memory.get("outcome", ""))
            if policy_type in {"veto", "avoid"} and outcome in {"leader", "improved", "submission_candidate"}:
                contradictions.append(
                    {
                        "card_id": str(card.get("card_id", "")),
                        "memory_id": str(memory.get("memory_id", "")),
                        "type": "negative-policy-overridden-by-result",
                        "summary": f"{card.get('title', '')} conflicts with successful branch memory {memory.get('run_id', '')}.",
                    }
                )
            elif policy_type in {"require", "prefer"} and outcome in {"regressed", "critic_rejected", "run_failed", "validate_failed"}:
                contradictions.append(
                    {
                        "card_id": str(card.get("card_id", "")),
                        "memory_id": str(memory.get("memory_id", "")),
                        "type": "positive-policy-underperformed-recently",
                        "summary": f"{card.get('title', '')} conflicts with recent weak outcome {memory.get('run_id', '')}.",
                    }
                )
    return contradictions[:8]


def retrieve_knowledge_bundle_from_root(
    workspace_root: Path,
    manifest: dict[str, Any],
    *,
    stage: str = "",
    limit: int = 8,
) -> dict[str, Any]:
    cards = compile_knowledge_index_from_root(workspace_root)
    frame = build_problem_frame(manifest, stage=stage)
    if not cards:
        return {
            "problem_frame": frame,
            "cards": [],
            "knowledge_files_seen": 0,
            "knowledge_card_ids": [],
        }
    scored = sorted(
        (
            {
                **card,
                "score": _score_card(card, frame, stage=stage or frame.get("stage", "")),
            }
            for card in cards
        ),
        key=lambda item: (float(item.get("score", 0.0)), str(item.get("source_path", "")), str(item.get("card_id", ""))),
        reverse=True,
    )
    selected = _select_diverse_cards([item for item in scored if float(item.get("score", 0.0)) > 0.0], limit)
    if not selected:
        selected = scored[:limit]
    return {
        "problem_frame": frame,
        "cards": selected,
        "knowledge_files_seen": len({str(item.get("source_path", "")) for item in selected}),
        "knowledge_card_ids": [str(item.get("card_id", "")) for item in selected],
    }


def retrieve_knowledge_bundle(
    config: WorkspaceConfig,
    manifest: dict[str, Any],
    *,
    stage: str = "",
    limit: int = 8,
    state: WorkspaceState | None = None,
    memory_limit: int = 6,
) -> dict[str, Any]:
    ensure_knowledge_layout(config)
    bundle = retrieve_knowledge_bundle_from_root(config.root, manifest, stage=stage, limit=limit)
    cards = bundle.get("cards", [])
    compact_cards = [item for item in cards if isinstance(item, dict)] if isinstance(cards, list) else []
    policy_cards = _policy_cards(compact_cards)
    bundle["policy_cards"] = policy_cards
    bundle["branch_memories"] = []
    bundle["branch_memory_ids"] = []
    bundle["contradictions"] = []
    if state is not None:
        frame = bundle.get("problem_frame", {}) if isinstance(bundle.get("problem_frame"), dict) else {}
        branch_memories = _recent_branch_memories(state, frame, stage=stage or str(frame.get("stage", "")), limit=memory_limit)
        bundle["branch_memories"] = branch_memories
        bundle["branch_memory_ids"] = [str(item.get("memory_id", "")) for item in branch_memories]
        bundle["contradictions"] = _policy_contradictions(policy_cards, branch_memories)
    return bundle


def render_retrieved_knowledge(bundle: dict[str, Any], *, limit: int = 16000) -> str:
    cards = bundle.get("cards", [])
    if not isinstance(cards, list) or not cards:
        return ""
    frame = bundle.get("problem_frame", {}) if isinstance(bundle.get("problem_frame"), dict) else {}
    sections: list[str] = []
    if frame:
        sections.append(
            "\n".join(
                [
                    "## problem_frame",
                    f"- family: {frame.get('family', '') or 'n/a'}",
                    f"- focus: {frame.get('focus', '') or 'n/a'}",
                    f"- root_cause: {frame.get('root_cause', '') or 'n/a'}",
                    f"- objective_metric: {frame.get('objective_metric', '') or 'n/a'}",
                ]
            )
        )
    grouped: dict[str, list[dict[str, Any]]] = {
        "positive": [],
        "negative": [],
        "conditional": [],
        "general": [],
    }
    for card in cards:
        grouped.setdefault(str(card.get("stance", "general")), []).append(card)
    for label, key in (
        ("Positive Priors", "positive"),
        ("Negative Vetoes", "negative"),
        ("Conditional Leads", "conditional"),
        ("General Context", "general"),
    ):
        bucket = grouped.get(key, [])
        if not bucket:
            continue
        lines = [f"## {label}"]
        for card in bucket:
            lines.extend(
                [
                    f"### {card.get('title', '')}",
                    f"- card_id: `{card.get('card_id', '')}`",
                    f"- source: `knowledge/{card.get('source_path', '')}`",
                    f"- component: `{card.get('component', 'general')}`",
                    f"- policy: `{card.get('policy_type', 'context')}` @ {float(card.get('confidence', 0.0) or 0.0):.2f}",
                    f"- summary: {card.get('summary', '')}",
                ]
            )
        sections.append("\n".join(lines))
    policy_cards = bundle.get("policy_cards", [])
    if isinstance(policy_cards, list) and policy_cards:
        lines = ["## Policy Cards"]
        for card in policy_cards:
            if not isinstance(card, dict):
                continue
            lines.append(
                f"- `{card.get('policy_type', 'context')}` | `{card.get('component', 'general')}` | `{card.get('card_id', '')}` | {card.get('summary', '')}"
            )
        sections.append("\n".join(lines))
    branch_memories = bundle.get("branch_memories", [])
    if isinstance(branch_memories, list) and branch_memories:
        lines = ["## Recent Branch Memories"]
        for memory in branch_memories:
            if not isinstance(memory, dict):
                continue
            lines.append(
                f"- `{memory.get('run_id', '')}` | outcome={memory.get('outcome', '')} | idea={memory.get('idea_class', '')} | {memory.get('summary', '')}"
            )
        sections.append("\n".join(lines))
    contradictions = bundle.get("contradictions", [])
    if isinstance(contradictions, list) and contradictions:
        lines = ["## Contradictions"]
        for item in contradictions:
            if not isinstance(item, dict):
                continue
            lines.append(f"- `{item.get('type', '')}` | {item.get('summary', '')}")
        sections.append("\n".join(lines))
    rendered = "\n\n".join(section for section in sections if section.strip())
    return truncate(rendered, limit=limit) if len(rendered) > limit else rendered


def compact_knowledge_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    cards = bundle.get("cards", [])
    compact_cards: list[dict[str, Any]] = []
    if isinstance(cards, list):
        for card in cards:
            compact_cards.append(
                {
                    "card_id": str(card.get("card_id", "")),
                    "title": str(card.get("title", "")),
                    "summary": str(card.get("summary", "")),
                    "source_path": str(card.get("source_path", "")),
                    "stance": str(card.get("stance", "")),
                    "component": str(card.get("component", "")),
                    "policy_type": str(card.get("policy_type", "context")),
                    "confidence": float(card.get("confidence", 0.0) or 0.0),
                }
            )
    return {
        "problem_frame": bundle.get("problem_frame", {}),
        "knowledge_files_seen": int(bundle.get("knowledge_files_seen", 0) or 0),
        "knowledge_card_ids": [str(item) for item in bundle.get("knowledge_card_ids", [])],
        "branch_memory_ids": [str(item) for item in bundle.get("branch_memory_ids", [])],
        "cards": compact_cards,
        "policy_cards": [item for item in bundle.get("policy_cards", []) if isinstance(item, dict)],
        "branch_memories": [item for item in bundle.get("branch_memories", []) if isinstance(item, dict)],
        "contradictions": [item for item in bundle.get("contradictions", []) if isinstance(item, dict)],
    }


def read_knowledge_context(config: WorkspaceConfig) -> str:
    ensure_knowledge_layout(config)
    parts: list[str] = []
    for path in sorted(config.knowledge_root().rglob("*.md")):
        if not path.is_file():
            continue
        relative = path.relative_to(config.knowledge_root())
        parts.append(f"## {relative}\n\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(part for part in parts if part.strip())


def _family_reference_metrics(state: WorkspaceState, run_id: str, family: str, *, parent_run_id: str = "") -> tuple[float | None, float | None]:
    family_best: float | None = None
    parent_metric: float | None = None
    experiment_by_id = {item.id: item for item in state.experiments}
    for candidate in state.runs:
        if candidate.run_id == run_id or candidate.status != "succeeded" or candidate.primary_metric_value is None:
            continue
        experiment = experiment_by_id.get(candidate.experiment_id)
        if experiment is None or experiment.family != family:
            continue
        family_best = candidate.primary_metric_value if family_best is None else max(family_best, candidate.primary_metric_value)
        if parent_run_id and candidate.run_id == parent_run_id:
            parent_metric = candidate.primary_metric_value
    return family_best, parent_metric


def _branch_memory_outcome(
    *,
    run,
    family_best_before: float | None,
    parent_metric: float | None,
    critic_status: str,
    validation_status: str,
    submission_status: str,
) -> tuple[str, float | None]:
    if critic_status == "rejected":
        return "critic_rejected", None
    if validation_status == "failed":
        return "validate_failed", None
    if run.status == "failed":
        return "run_failed", None
    if run.primary_metric_value is None:
        return "unscored", None
    delta_reference = parent_metric if parent_metric is not None else family_best_before
    delta = None if delta_reference is None else run.primary_metric_value - delta_reference
    if family_best_before is None or run.primary_metric_value > family_best_before + 1e-9:
        return ("submission_candidate" if submission_status == "candidate_created" else "leader"), delta
    if delta is not None and delta >= 0.002:
        return "improved", delta
    if delta is not None and delta <= -0.002:
        return "regressed", delta
    return "flat", delta


def _signal_score_for_outcome(outcome: str) -> float:
    return {
        "leader": 3.0,
        "submission_candidate": 2.6,
        "improved": 2.0,
        "flat": 0.5,
        "unscored": 0.0,
        "regressed": -1.8,
        "critic_rejected": -2.4,
        "validate_failed": -2.6,
        "run_failed": -3.0,
    }.get(outcome, 0.0)


def synchronize_branch_memory(state: WorkspaceState, run_id: str) -> BranchMemoryRecord | None:
    run = next((item for item in state.runs if item.run_id == run_id), None)
    if run is None:
        return None
    work_item = next((item for item in state.work_items if item.id == run.work_item_id), None)
    experiment = next((item for item in state.experiments if item.id == run.experiment_id), None)
    if work_item is None or experiment is None:
        return None
    critic = latest_stage_payload(state, run_id, "critic")
    validation = latest_stage_payload(state, run_id, "validate")
    decision = latest_stage_payload(state, run_id, "decision")
    submission = latest_stage_payload(state, run_id, "submission")
    report = latest_stage_payload(state, run_id, "report")
    source_run_id = str(work_item.source_run_id or "")
    family_best_before, parent_metric = _family_reference_metrics(state, run.run_id, experiment.family, parent_run_id=source_run_id)
    outcome, metric_delta = _branch_memory_outcome(
        run=run,
        family_best_before=family_best_before,
        parent_metric=parent_metric,
        critic_status=str(critic.get("status", "")),
        validation_status=str(validation.get("status", "")),
        submission_status=str(submission.get("status", "")),
    )
    summary_bits = [f"{work_item.branch_role or 'branch'} {work_item.idea_class or experiment.family} -> {outcome}"]
    if run.primary_metric_value is not None:
        summary_bits.append(f"{run.primary_metric_name or 'metric'}={run.primary_metric_value:.6f}")
    if metric_delta is not None:
        summary_bits.append(f"delta={metric_delta:+.6f}")
    if critic.get("status"):
        summary_bits.append(f"critic={critic.get('status')}")
    if validation.get("status"):
        summary_bits.append(f"validate={validation.get('status')}")
    memory = next((item for item in state.branch_memories if item.run_id == run_id), None)
    if memory is None:
        memory = BranchMemoryRecord(
            memory_id=f"memory-{state.runtime.next_branch_memory_number:04d}",
            run_id=run_id,
            work_item_id=work_item.id,
            experiment_id=experiment.id,
            family=experiment.family,
            created_at=now_utc_iso(),
        )
        state.runtime.next_branch_memory_number += 1
        state.branch_memories.append(memory)
    memory.portfolio_id = work_item.portfolio_id
    memory.idea_class = work_item.idea_class
    memory.branch_role = work_item.branch_role
    memory.branch_rank = work_item.branch_rank
    memory.source_stage_run_id = run.latest_stage_run_id
    memory.status = run.status
    memory.outcome = outcome
    memory.summary = truncate("; ".join(summary_bits), limit=320)
    memory.root_cause = str(report.get("root_cause") or decision.get("root_cause") or run.root_cause or run.error or "")
    memory.metric_name = run.primary_metric_name
    memory.metric_value = run.primary_metric_value
    memory.metric_delta = metric_delta
    memory.verify_status = str(latest_stage_payload(state, run_id, "codegen").get("verify_status", ""))
    memory.critic_status = str(critic.get("status", ""))
    memory.validation_status = str(validation.get("status", ""))
    memory.submission_status = str(submission.get("status", ""))
    memory.signal_score = _signal_score_for_outcome(outcome)
    memory.policy_tags = [
        f"outcome:{outcome}",
        *(f"role:{work_item.branch_role}" for _ in [0] if work_item.branch_role),
        *(f"idea:{work_item.idea_class}" for _ in [0] if work_item.idea_class),
    ]
    memory.knowledge_card_ids = list(work_item.knowledge_card_ids)
    memory.contradiction_ids = []
    memory.updated_at = now_utc_iso()
    return memory


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
    compile_knowledge_index(config)
    return path
