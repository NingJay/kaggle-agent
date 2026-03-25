from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle_agent.control.store import load_state, save_state
from kaggle_agent.service import build_submission, init_workspace, load_config, start_next
from tests.test_workspace import _copy_runtime, _write_workspace


EXTERNAL_ROOT = Path("/home/staff/jiayining/kaggle/kaggle_comp_seed/BirdCLEF-2026")


def _skip_external() -> bool:
    return os.environ.get("KAGGLE_AGENT_RUN_EXTERNAL_BIRDCLEF") != "1" or not EXTERNAL_ROOT.exists()


def _write_external_workspace(root: Path) -> None:
    _write_workspace(root)
    workspace_path = root / "workspace.toml"
    text = workspace_path.read_text(encoding="utf-8")
    text = text.replace(
        f'root = "{root / "BirdCLEF-2026-Codebase" / "birdclef-2026"}"',
        f'root = "{EXTERNAL_ROOT / "dataset"}"',
    )
    text = text.replace('perch_cache_dir = ""', f'perch_cache_dir = "{EXTERNAL_ROOT / "input" / "perch-meta"}"')
    text = text.replace('perch_model_dir = ""', f'perch_model_dir = "{EXTERNAL_ROOT / "models" / "google" / "bird-vocalization-classifier" / "tensorflow2" / "perch_v2_cpu" / "1"}"')
    text = text.replace('conda_env = ""', 'conda_env = "kaggle-agent"')
    text = text.replace('shell_init = ""', 'shell_init = "source /home/staff/jiayining/miniconda3/etc/profile.d/conda.sh"')
    workspace_path.write_text(text, encoding="utf-8")


@unittest.skipIf(_skip_external(), "Set KAGGLE_AGENT_RUN_EXTERNAL_BIRDCLEF=1 to run the external BirdCLEF integration test.")
class ExternalBirdCLEFIntegrationTests(unittest.TestCase):
    def test_reference_cached_probe_stage_chain(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_external_workspace(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            debug_item = next(item for item in state.work_items if item.id == "workitem-perch-debug-smoke")
            debug_item.status = "complete"
            save_state(config, state)

            run_id = start_next(config, background=False)
            self.assertTrue(run_id)

            state = load_state(config)
            run = next(item for item in state.runs if item.run_id == run_id)
            self.assertEqual(run.status, "succeeded")
            self.assertEqual(run.stage_cursor, "complete")
            self.assertEqual(run.stage_error, "")
            self.assertIn("prior_fusion_macro_roc_auc", run.secondary_metrics)

            stage_names = [item.stage_name for item in state.stage_runs if item.run_id == run_id]
            self.assertEqual(stage_names, ["evidence", "report", "research", "decision", "plan", "codegen", "critic", "validate", "submission"])

            result_payload = json.loads(Path(run.artifact_paths["result"]).read_text(encoding="utf-8"))
            self.assertEqual(result_payload["dataset_summary"]["fully_labeled_files"], 59)
            self.assertEqual(result_payload["dataset_summary"]["active_class_count"], 71)
            self.assertTrue(Path(result_payload["artifacts"]["probe_bundle"]).exists())
            self.assertTrue(Path(result_payload["artifacts"]["oof_predictions"]).exists())

            candidate_id = build_submission(config, run_id=run_id)
            candidate_root = root / "artifacts" / "submissions" / candidate_id
            self.assertTrue((candidate_root / "candidate_manifest.json").exists())
            self.assertTrue((candidate_root / "dry_run.json").exists())
