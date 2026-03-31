from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kaggle_agent.decision.helpers import latest_stage_payload, load_run_result
from kaggle_agent.layout import visible_runs
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_json, atomic_write_text, ensure_directory, slugify, truncate


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
) -> dict[str, Any]:
    ensure_knowledge_layout(config)
    return retrieve_knowledge_bundle_from_root(config.root, manifest, stage=stage, limit=limit)


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
                    f"- summary: {card.get('summary', '')}",
                ]
            )
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
                }
            )
    return {
        "problem_frame": bundle.get("problem_frame", {}),
        "knowledge_files_seen": int(bundle.get("knowledge_files_seen", 0) or 0),
        "knowledge_card_ids": [str(item) for item in bundle.get("knowledge_card_ids", [])],
        "cards": compact_cards,
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
