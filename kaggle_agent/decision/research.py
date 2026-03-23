from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from kaggle_agent.adapters.command import run_command_adapter
from kaggle_agent.decision.helpers import load_run_result
from kaggle_agent.knowledge import ensure_knowledge_layout
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _candidate_queries(root_cause: str, experiment_family: str) -> list[str]:
    root = root_cause.lower()
    queries: list[str] = []
    if "prior" in root or "calibration" in root or "context" in root:
        queries.append("bioacoustics soundscape classification calibration priors")
    if "sparse" in root or "positive" in root or "imbalance" in root:
        queries.append("long tail bioacoustics multilabel classification class imbalance")
    if "probe" in root or "embedding" in root or "perch" in root or "cached" in experiment_family:
        queries.append("frozen audio embeddings linear probe bioacoustics")
    queries.append("bird soundscape classification passive acoustic monitoring")
    deduped: list[str] = []
    for query in queries:
        if query not in deduped:
            deduped.append(query)
    return deduped[:3]


def _fetch_arxiv_papers(query: str, max_results: int = 3) -> list[dict[str, str]]:
    url = (
        "https://export.arxiv.org/api/query?search_query=all:"
        + urllib.parse.quote(query)
        + f"&start=0&max_results={max_results}"
    )
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    root = ET.fromstring(payload)
    papers: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        link = ""
        for candidate in entry.findall("atom:link", ATOM_NS):
            href = candidate.attrib.get("href", "")
            rel = candidate.attrib.get("rel", "")
            if href and rel == "alternate":
                link = href
                break
        papers.append(
            {
                "title": " ".join((entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").split()),
                "summary": " ".join((entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").split()),
                "published": (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "")[:10],
                "link": link or (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or ""),
                "authors": ", ".join(
                    " ".join((author.findtext("atom:name", default="", namespaces=ATOM_NS) or "").split())
                    for author in entry.findall("atom:author", ATOM_NS)
                ),
            }
        )
    return papers


def _internal_research_text(root_cause: str, experiment_family: str, metric_name: str, metric_value: float | None) -> tuple[str, str]:
    queries = _candidate_queries(root_cause, experiment_family)
    papers: list[dict[str, str]] = []
    seen_titles: set[str] = set()
    errors: list[str] = []
    for query in queries:
        try:
            for paper in _fetch_arxiv_papers(query):
                title_key = paper["title"].lower()
                if not title_key or title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                papers.append(paper)
        except Exception as error:  # noqa: BLE001
            errors.append(f"{query}: {error}")
        if len(papers) >= 3:
            break
    metric_text = "-" if metric_value is None else f"{metric_value:.6f}"
    lines = [
        "# Research Summary",
        "",
        f"- Root cause focus: {root_cause}",
        f"- Current metric: {metric_name}={metric_text}",
        f"- Search queries: {', '.join(queries)}",
        "",
        "## Related Papers",
    ]
    if papers:
        for paper in papers[:3]:
            lines.extend(
                [
                    f"### {paper['title']}",
                    f"- Published: {paper['published'] or 'unknown'}",
                    f"- Authors: {paper['authors'] or 'unknown'}",
                    f"- Link: {paper['link'] or 'unavailable'}",
                    f"- Why it matters: {'Prior or calibration methods are relevant to soundscape context.' if 'prior' in root_cause.lower() or 'calibration' in root_cause.lower() else 'Embedding and probe design are relevant to the current Perch baseline.'}",
                    f"- Summary: {paper['summary']}",
                    "",
                ]
            )
    else:
        lines.append("- No papers fetched; using fallback heuristic summary.")
        lines.append("")
    if errors:
        lines.extend(["## Fetch Notes", *[f"- {item}" for item in errors], ""])
    text = "\n".join(lines).rstrip() + "\n"
    return text, text


def build_research_summary(config: WorkspaceConfig, state: WorkspaceState, run_id: str) -> Path:
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    result = load_run_result(run)
    input_path = Path(run.decision_brief_path)
    output_path = config.artifact_path("research", f"{run.run_id}.md")
    ensure_knowledge_layout(config)
    if config.adapters.research_command.strip():
        text = run_command_adapter(
            config.adapters.research_command,
            stage="research",
            workspace_root=config.root,
            input_path=input_path,
            output_path=output_path,
            extra_env={
                "KAGGLE_AGENT_ROOT_CAUSE": str(result.get("root_cause", run.error)),
                "KAGGLE_AGENT_PRIMARY_METRIC_VALUE": "" if run.primary_metric_value is None else str(run.primary_metric_value),
            },
        )
        atomic_write_text(output_path, text if text.endswith("\n") else text + "\n")
    else:
        root_cause = str(result.get("root_cause", run.error or "missing context"))
        text, paper_text = _internal_research_text(root_cause, experiment.family, run.primary_metric_name, run.primary_metric_value)
        atomic_write_text(output_path, text)
        atomic_write_text(config.knowledge_path("papers", f"{run.run_id}_paper_summary.md"), paper_text)
    atomic_write_text(config.knowledge_path("research", f"{run.run_id}.md"), output_path.read_text(encoding="utf-8"))
    run.research_summary_path = str(output_path)
    return output_path
