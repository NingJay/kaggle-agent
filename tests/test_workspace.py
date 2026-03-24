from __future__ import annotations

import csv
import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_agent.control.store import load_state
from kaggle_agent.service import (
    build_submission,
    doctor_checks,
    init_workspace,
    list_ready_work_items,
    load_config,
    plan_submission,
    start_next,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_runtime(root: Path) -> None:
    shutil.copy2(REPO_ROOT / "train_sed.py", root / "train_sed.py")
    shutil.copytree(REPO_ROOT / "BirdCLEF-2026-Codebase", root / "BirdCLEF-2026-Codebase")
    outputs = root / "BirdCLEF-2026-Codebase" / "outputs"
    if outputs.exists():
        shutil.rmtree(outputs)


def _write_workspace(root: Path) -> None:
    workspace_toml = f"""
notes = ["test workspace"]

[competition]
name = "BirdCLEF 2026"
slug = "birdclef-2026"
url = "https://www.kaggle.com/competitions/birdclef-2026"
track = "code_competition"
description = "Test workspace"
contract = "birdclef_2026"

[metrics]
primary = "soundscape_macro_roc_auc"
secondary = ["padded_cmap"]

[data]
root = "{root / 'BirdCLEF-2026-Codebase' / 'birdclef-2026'}"
train_csv = "train.csv"
taxonomy_csv = "taxonomy.csv"
sample_submission_csv = "sample_submission.csv"
train_audio_dir = "train_audio"
train_soundscapes_dir = "train_soundscapes"
train_soundscapes_labels_csv = "train_soundscapes_labels.csv"
test_soundscapes_dir = "test_soundscapes"
perch_cache_dir = ""
perch_model_dir = ""

[paths]
state_dir = "state"
artifact_dir = "artifacts"
legacy_dir = "legacy"
runtime_dir = "BirdCLEF-2026-Codebase"
knowledge_dir = "knowledge"
prompt_dir = "prompts"
report_dir = "reports"

[automation]
monitor_interval_seconds = 1
report_interval_seconds = 1
submission_interval_hours = 6
default_timeout_minutes = 5
max_active_runs = 1
auto_execute_plans = true
auto_start_planned_runs = true
strict_stage_graph = true

[adapters]
evidence_command = ""
report_command = ""
research_command = ""
decision_command = ""
planner_command = ""
codegen_command = ""
critic_command = ""
submission_command = ""

[runtime]
conda_env = ""
shell_init = ""
train_workdir = "{root}"
train_entrypoint = "train_sed.py"
generated_config_dir = "BirdCLEF-2026-Codebase/configs/generated"

[kaggle]
username = "tester"
model_dataset_id = "tester/birdclef2026-model"
kernel_slug = "birdclef-2026-research-os"
enable_gpu = false
enable_internet = false
is_private = true
cpu_submission_only = true
scored_max_runtime_minutes = 90
max_daily_submissions = 5
max_final_submissions = 2
dataset_sources = ["tester/birdclef2026-model"]
""".strip()
    (root / "workspace.toml").write_text(workspace_toml + "\n", encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _build_debug_dataset(root: Path) -> None:
    data_root = root / "BirdCLEF-2026-Codebase" / "birdclef-2026"
    _write_csv(
        data_root / "train.csv",
        [
            ["primary_label", "filename"],
            ["sp1", "a.ogg"],
            ["sp1", "b.ogg"],
            ["sp1", "c.ogg"],
            ["sp2", "d.ogg"],
            ["sp2", "e.ogg"],
            ["sp2", "f.ogg"],
        ],
    )
    _write_csv(data_root / "taxonomy.csv", [["primary_label"], ["sp1"], ["sp2"]])
    _write_csv(
        data_root / "train_soundscapes_labels.csv",
        [
            ["row_id", "filename", "primary_label"],
            ["row-1", "sound1.ogg", "sp1"],
            ["row-2", "sound2.ogg", "sp2"],
        ],
    )
    _write_csv(data_root / "sample_submission.csv", [["row_id", "sp1", "sp2"], ["row-1", "0", "0"], ["row-2", "0", "0"]])
    for label, filename in [("sp1", "a.ogg"), ("sp1", "b.ogg"), ("sp1", "c.ogg"), ("sp2", "d.ogg"), ("sp2", "e.ogg"), ("sp2", "f.ogg")]:
        _touch(data_root / "train_audio" / label / filename)
    _touch(data_root / "train_soundscapes" / "sound1.ogg")
    _touch(data_root / "train_soundscapes" / "sound2.ogg")


class WorkspaceTests(unittest.TestCase):
    def test_init_creates_ledger_reports_and_surface_docs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            (root / "reports").mkdir()
            (root / "reports" / "old.html").write_text("legacy", encoding="utf-8")

            config = load_config(root)
            init_workspace(config, archive_legacy=True, force=True)

            legacy_root = root / "legacy"
            archives = list(legacy_root.iterdir())
            self.assertEqual(len(archives), 1)
            self.assertTrue((archives[0] / "reports" / "old.html").exists())
            self.assertTrue((root / "state" / "ledger.db").exists())
            self.assertTrue((root / "reports" / "master_report.html").exists())
            self.assertTrue((root / "CHECKLIST.md").exists())
            self.assertTrue((root / "COMPETITION.md").exists())
            self.assertTrue((root / "prompts" / "report.md").exists())

    def test_sync_run_generates_stage_chain_and_unblocks_baseline(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            run_id = start_next(config, background=False)
            self.assertTrue(run_id)

            state = load_state(config)
            self.assertEqual(len(state.runs), 1)
            run = state.runs[0]
            self.assertEqual(run.status, "succeeded")
            self.assertEqual(run.stage_cursor, "complete")
            self.assertEqual(run.stage_error, "")

            stage_names = [item.stage_name for item in state.stage_runs if item.run_id == run_id]
            self.assertEqual(stage_names, ["evidence", "report", "research", "decision", "plan", "codegen", "critic", "validate", "submission"])
            self.assertGreaterEqual(len(state.metrics), 1)
            self.assertGreaterEqual(len(state.findings), 1)
            self.assertGreaterEqual(len(state.research_notes), 1)
            debug_item = next(item for item in state.work_items if item.id == "workitem-perch-debug-smoke")
            self.assertEqual(debug_item.status, "complete")
            ready = list_ready_work_items(config)
            self.assertIn("workitem-perch-baseline", ready)
            self.assertTrue((root / "reports" / "master_report.html").exists())
            self.assertTrue((root / "knowledge" / "experiment_conclusions.md").exists())
            self.assertTrue((root / "FINDINGS.md").exists())

    def test_build_submission_creates_bundle_and_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            run_id = start_next(config, background=False)
            candidate_id = build_submission(config, run_id=run_id)

            state = load_state(config)
            self.assertEqual(len(state.submissions), 1)
            candidate = state.submissions[0]
            self.assertEqual(candidate.id, candidate_id)
            self.assertTrue(candidate.cpu_ready)

            candidate_root = root / "artifacts" / "submissions" / candidate_id
            self.assertTrue((candidate_root / "candidate_manifest.json").exists())
            self.assertTrue((candidate_root / "candidate.md").exists())
            self.assertTrue((candidate_root / "notebook.ipynb").exists())
            self.assertTrue((candidate_root / "kernel-metadata.json").exists())
            self.assertTrue((candidate_root / "bundle_runner.py").exists())
            self.assertTrue((candidate_root / "dry_run.json").exists())
            self.assertTrue((candidate_root / "submission.csv").exists())
            dry_run_payload = json.loads((candidate_root / "dry_run.json").read_text(encoding="utf-8"))
            self.assertEqual(dry_run_payload["status"], "passed")

    def test_build_submission_is_idempotent_for_same_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            run_id = start_next(config, background=False)
            first_candidate_id = build_submission(config, run_id=run_id)
            second_candidate_id = build_submission(config, run_id=run_id)

            state = load_state(config)
            self.assertEqual(first_candidate_id, second_candidate_id)
            self.assertEqual(len(state.submissions), 1)
            self.assertEqual(state.submissions[0].dedupe_key, f"submission_candidate:{run_id}")

    def test_doctor_flags_missing_cached_probe_inputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            checks = {name: ok for ok, name, _detail in doctor_checks(config)}
            self.assertIn("perch_cache_dir", checks)
            self.assertFalse(checks["perch_cache_dir"])

    def test_plan_submission_reports_slot_budget(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            run_id = start_next(config, background=False)
            build_submission(config, run_id=run_id)
            slots = plan_submission(config)
            self.assertEqual(slots["remaining_daily_slots"], 5)
            self.assertEqual(len(slots["ready_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
