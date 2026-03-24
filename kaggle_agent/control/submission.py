from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from kaggle_agent.schema import SubmissionCandidate, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_json, atomic_write_text, ensure_directory, now_utc_iso, slugify


def submission_dedupe_key(run_id: str) -> str:
    return f"submission_candidate:{run_id}"


def _sample_submission_path(config: WorkspaceConfig) -> Path:
    return Path(config.data.root) / config.data.sample_submission_csv


def _candidate_root(config: WorkspaceConfig, candidate_id: str) -> Path:
    return ensure_directory(config.artifact_path("submissions", candidate_id))


def _candidate_manifest(config: WorkspaceConfig, run, experiment, candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "competition": config.competition.to_dict() if hasattr(config.competition, "to_dict") else {
            "name": config.competition.name,
            "slug": config.competition.slug,
            "url": config.competition.url,
            "contract": config.competition.contract,
        },
        "run": run.to_dict(),
        "experiment": experiment.to_dict(),
        "sample_submission_csv": str(_sample_submission_path(config)),
        "contract": {
            "cpu_submission_only": config.kaggle.cpu_submission_only,
            "enable_internet": config.kaggle.enable_internet,
            "scored_max_runtime_minutes": config.kaggle.scored_max_runtime_minutes,
            "max_daily_submissions": config.kaggle.max_daily_submissions,
            "max_final_submissions": config.kaggle.max_final_submissions,
        },
    }


def _bundle_runner_text() -> str:
    return """from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="candidate_manifest.json")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_path = Path(manifest["sample_submission_csv"]).resolve()
    output_path = Path(args.output).resolve()
    rows = list(csv.reader(sample_path.open("r", encoding="utf-8", newline="")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row_index, row in enumerate(rows):
            if row_index == 0:
                writer.writerow(row)
                continue
            writer.writerow([row[0], *(["0"] * (len(row) - 1))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _notebook_payload(config: WorkspaceConfig, run, experiment, manifest_name: str) -> dict[str, Any]:
    kernel_title = config.kaggle.kernel_slug.replace("-", " ").title()
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {kernel_title}\\n",
                    "\\n",
                    f"- Source run: `{run.run_id}`\\n",
                    f"- Experiment: `{experiment.id}`\\n",
                    f"- Local metric: `{run.primary_metric_name}={run.primary_metric_value}`\\n",
                    "- This bundle is CPU-first and internet-off by contract.\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\\n",
                    "import csv\\n",
                    "import json\\n",
                    f"manifest = json.loads(Path('{manifest_name}').read_text())\\n",
                    "sample_path = Path(manifest['sample_submission_csv'])\\n",
                    "rows = list(csv.reader(sample_path.open('r', encoding='utf-8', newline='')))\\n",
                    "with Path('submission.csv').open('w', encoding='utf-8', newline='') as handle:\\n",
                    "    writer = csv.writer(handle)\\n",
                    "    for idx, row in enumerate(rows):\\n",
                    "        if idx == 0:\\n",
                    "            writer.writerow(row)\\n",
                    "        else:\\n",
                    "            writer.writerow([row[0], *(['0'] * (len(row) - 1))])\\n",
                    "print('submission.csv written')\\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _kernel_metadata(config: WorkspaceConfig) -> dict[str, Any]:
    kernel_id = f"{config.kaggle.username}/{config.kaggle.kernel_slug}" if config.kaggle.username else config.kaggle.kernel_slug
    return {
        "id": kernel_id,
        "title": config.kaggle.kernel_slug.replace("-", " ").title(),
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": config.kaggle.is_private,
        "enable_gpu": False if config.kaggle.cpu_submission_only else config.kaggle.enable_gpu,
        "enable_internet": config.kaggle.enable_internet,
        "competition_sources": [config.competition.slug],
        "dataset_sources": list(config.kaggle.dataset_sources),
    }


def _upsert_candidate(state: WorkspaceState, candidate: SubmissionCandidate) -> SubmissionCandidate:
    for index, existing in enumerate(state.submissions):
        if existing.dedupe_key == candidate.dedupe_key or existing.source_run_id == candidate.source_run_id:
            state.submissions[index] = candidate
            return candidate
    state.submissions.append(candidate)
    return candidate


def build_submission_candidate(config: WorkspaceConfig, state: WorkspaceState, run_id: str) -> SubmissionCandidate:
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    dedupe_key = submission_dedupe_key(run_id)
    existing = next((item for item in state.submissions if item.dedupe_key == dedupe_key), None)
    if existing is not None:
        candidate_id = existing.id
    else:
        candidate_id = f"submission-{state.runtime.next_submission_number:04d}-{slugify(run_id)}"
        state.runtime.next_submission_number += 1
    root = _candidate_root(config, candidate_id)
    manifest = _candidate_manifest(config, run, experiment, candidate_id)
    manifest_path = root / "candidate_manifest.json"
    candidate_md_path = root / "candidate.md"
    calibration_path = root / "calibration.json"
    notebook_path = root / "notebook.ipynb"
    metadata_path = root / "kernel-metadata.json"
    runner_path = root / "bundle_runner.py"

    atomic_write_json(manifest_path, manifest)
    atomic_write_text(
        candidate_md_path,
        "\n".join(
            [
                f"# Submission Candidate {candidate_id}",
                "",
                f"- Source run: `{run.run_id}`",
                f"- Experiment: `{experiment.id}`",
                f"- Local primary metric: `{run.primary_metric_name}={run.primary_metric_value}`",
                "- Bundle strategy: zero-filled CPU dry-run bundle with explicit contract metadata.",
            ]
        )
        + "\n",
    )
    atomic_write_json(calibration_path, {
        "predicted_public_lb": run.primary_metric_value,
        "predicted_public_lb_std": 0.03 if run.primary_metric_value is not None else None,
        "anchor_basis": "local_primary_metric",
    })
    atomic_write_text(notebook_path, json.dumps(_notebook_payload(config, run, experiment, manifest_path.name), indent=2) + "\n")
    atomic_write_json(metadata_path, _kernel_metadata(config))
    atomic_write_text(runner_path, _bundle_runner_text())

    candidate = SubmissionCandidate(
        id=candidate_id,
        source_run_id=run.run_id,
        experiment_id=experiment.id,
        status="candidate",
        primary_metric_name=run.primary_metric_name,
        primary_metric_value=run.primary_metric_value,
        secondary_metrics=dict(run.secondary_metrics),
        predicted_public_lb=run.primary_metric_value,
        predicted_public_lb_std=0.03 if run.primary_metric_value is not None else None,
        rationale=f"Candidate built from {run.run_id} after local validation.",
        notebook_dir=str(root.relative_to(config.root)),
        candidate_json_path=str(manifest_path),
        candidate_md_path=str(candidate_md_path),
        calibration_json_path=str(calibration_path),
        created_at=existing.created_at if existing is not None else now_utc_iso(),
        updated_at=now_utc_iso(),
        dedupe_key=dedupe_key,
    )
    candidate = _upsert_candidate(state, candidate)
    dry_run_submission_candidate(config, state, candidate.id)
    return candidate


def _find_candidate_root(config: WorkspaceConfig, candidate: SubmissionCandidate) -> Path:
    root = Path(candidate.notebook_dir)
    return root if root.is_absolute() else config.root / root


def dry_run_submission_candidate(config: WorkspaceConfig, state: WorkspaceState, candidate_id: str) -> dict[str, Any]:
    candidate = next(item for item in state.submissions if item.id == candidate_id)
    root = _find_candidate_root(config, candidate)
    manifest_path = Path(candidate.candidate_json_path)
    output_path = root / "submission.csv"
    completed = subprocess.run(
        [sys.executable, str(root / "bundle_runner.py"), "--manifest", str(manifest_path), "--output", str(output_path)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))
    result = {
        "candidate_id": candidate.id,
        "status": "passed" if completed.returncode == 0 and output_path.exists() else "failed",
        "cpu_only": config.kaggle.cpu_submission_only,
        "enable_gpu": bool(metadata.get("enable_gpu", False)),
        "enable_internet": bool(metadata.get("enable_internet", False)),
        "runtime_budget_minutes": config.kaggle.scored_max_runtime_minutes,
        "submission_csv_exists": output_path.exists(),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    dry_run_path = root / "dry_run.json"
    atomic_write_json(dry_run_path, result)
    candidate.dry_run_json_path = str(dry_run_path)
    candidate.cpu_ready = result["status"] == "passed" and not result["enable_gpu"] and not result["enable_internet"]
    candidate.status = "cpu_ready" if candidate.cpu_ready else "candidate"
    candidate.updated_at = now_utc_iso()
    return result


def plan_submission_slots(config: WorkspaceConfig, state: WorkspaceState) -> dict[str, Any]:
    actual_submissions = len(state.submission_results)
    remaining_daily = max(config.kaggle.max_daily_submissions - actual_submissions, 0)
    remaining_final = max(config.kaggle.max_final_submissions - actual_submissions, 0)
    ready_candidates = [item.id for item in state.submissions if item.cpu_ready]
    return {
        "actual_submissions": actual_submissions,
        "ready_candidates": ready_candidates,
        "remaining_daily_slots": remaining_daily,
        "remaining_final_slots": remaining_final,
    }
