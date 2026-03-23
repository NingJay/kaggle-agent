from __future__ import annotations

import csv
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_agent.control.store import load_state, save_state
from kaggle_agent.service import build_submission, doctor_checks, init_workspace, load_config, start_next, tick


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

[paths]
state_dir = "state"
artifact_dir = "artifacts"
legacy_dir = "legacy"
runtime_dir = "BirdCLEF-2026-Codebase"

[automation]
monitor_interval_seconds = 1
report_interval_seconds = 1
submission_interval_hours = 6
default_timeout_minutes = 5
max_active_runs = 1
auto_execute_plans = true
auto_start_planned_runs = true

[adapters]
research_command = ""
decision_command = ""
planner_command = ""
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
kernel_slug = "birdclef-2026-agent-inference"
enable_gpu = true
is_private = true
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
    _write_csv(data_root / "sample_submission.csv", [["row_id", "sp1", "sp2"], ["row-1", "0", "0"]])
    for label, filename in [("sp1", "a.ogg"), ("sp1", "b.ogg"), ("sp1", "c.ogg"), ("sp2", "d.ogg"), ("sp2", "e.ogg"), ("sp2", "f.ogg")]:
        _touch(data_root / "train_audio" / label / filename)
    _touch(data_root / "train_soundscapes" / "sound1.ogg")
    _touch(data_root / "train_soundscapes" / "sound2.ogg")


class WorkspaceTests(unittest.TestCase):
    def test_init_archives_legacy_artifacts_and_seeds_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            (root / "knowledge").mkdir()
            (root / "reports").mkdir()
            (root / "knowledge" / "old.md").write_text("legacy", encoding="utf-8")
            config = load_config(root)
            init_workspace(config, archive_legacy=True, force=True)
            legacy_root = root / "legacy"
            archives = list(legacy_root.iterdir())
            self.assertEqual(len(archives), 1)
            self.assertTrue((archives[0] / "knowledge" / "old.md").exists())
            self.assertTrue((root / "state" / "experiments.json").exists())
            self.assertTrue((root / "artifacts" / "reports" / "overview.md").exists())

    def test_sync_run_produces_decision_pipeline_outputs(self) -> None:
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
            self.assertTrue(Path(run.decision_brief_path).exists())
            self.assertTrue(Path(run.research_summary_path).exists())
            self.assertTrue(Path(run.decision_record_path).exists())
            self.assertTrue(Path(run.plan_path).exists())
            self.assertEqual(run.post_run_stage, "complete")
            self.assertEqual(run.post_run_error, "")
            self.assertEqual(len(state.experiments), 3)
            follow_ups = [item for item in state.experiments if item.source_decision_id]
            self.assertEqual(len(follow_ups), 1)
            self.assertTrue(follow_ups[0].dedupe_key)
            self.assertTrue((root / "artifacts" / "reports" / "dashboard.html").exists())
            self.assertTrue((root / "knowledge" / "experiment_conclusions.md").exists())
            self.assertTrue((root / "knowledge" / "research" / f"{run_id}.md").exists())

    def test_build_submission_creates_candidate_scaffold(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            run_id = start_next(config, background=False)
            submission_id = build_submission(config, run_id=run_id)
            candidate_root = root / "artifacts" / "submissions" / submission_id
            self.assertTrue((candidate_root / "notebook.ipynb").exists())
            self.assertTrue((candidate_root / "kernel-metadata.json").exists())

    def test_build_submission_is_idempotent_for_same_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            run_id = start_next(config, background=False)
            first_submission_id = build_submission(config, run_id=run_id)
            second_submission_id = build_submission(config, run_id=run_id)
            state = load_state(config)
            self.assertEqual(first_submission_id, second_submission_id)
            self.assertEqual(len(state.submissions), 1)
            self.assertEqual(state.submissions[0].dedupe_key, f"submission_candidate:{run_id}")

    def test_reprocessing_plan_done_run_does_not_duplicate_follow_up_experiment(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            start_next(config, background=False)
            state = load_state(config)
            run = state.runs[0]
            run.post_run_stage = "plan_done"
            run.post_run_error = ""
            save_state(config, state)
            tick(config, auto_start=False)
            refreshed = load_state(config)
            self.assertEqual(len(refreshed.experiments), 3)
            self.assertEqual(refreshed.runs[0].post_run_stage, "complete")
            follow_ups = [item for item in refreshed.experiments if item.source_decision_id]
            self.assertEqual(len(follow_ups), 1)
            self.assertTrue(follow_ups[0].dedupe_key)

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


if __name__ == "__main__":
    unittest.main()
