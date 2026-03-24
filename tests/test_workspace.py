from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
    shutil.copytree(REPO_ROOT / "schemas", root / "schemas")
    outputs = root / "BirdCLEF-2026-Codebase" / "outputs"
    if outputs.exists():
        shutil.rmtree(outputs)


def _adapter_commands() -> dict[str, str]:
    wrapper = f"PYTHONPATH={REPO_ROOT} {sys.executable} -m kaggle_agent.adapters.stage_wrapper"
    return {
        "evidence_command": "",
        "report_command": f"{wrapper} --provider claude",
        "research_command": f"{wrapper} --provider claude",
        "decision_command": f"{wrapper} --provider claude",
        "planner_command": f"{wrapper} --provider codex",
        "codegen_command": f"{wrapper} --provider codex",
        "critic_command": f"{wrapper} --provider critic",
        "submission_command": "",
    }


def _write_workspace(root: Path, *, adapter_commands: dict[str, str] | None = None) -> None:
    adapter_commands = adapter_commands or {
        "evidence_command": "",
        "report_command": "",
        "research_command": "",
        "decision_command": "",
        "planner_command": "",
        "codegen_command": "",
        "critic_command": "",
        "submission_command": "",
    }
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
evidence_command = {json.dumps(adapter_commands["evidence_command"])}
report_command = {json.dumps(adapter_commands["report_command"])}
research_command = {json.dumps(adapter_commands["research_command"])}
decision_command = {json.dumps(adapter_commands["decision_command"])}
planner_command = {json.dumps(adapter_commands["planner_command"])}
codegen_command = {json.dumps(adapter_commands["codegen_command"])}
critic_command = {json.dumps(adapter_commands["critic_command"])}
submission_command = {json.dumps(adapter_commands["submission_command"])}

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


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _install_fake_providers(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "claude",
        """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
if "--help" in args:
    print("Usage: claude [options]\\n  --output-format\\n  --json-schema\\n  --tools\\n  --no-session-persistence\\n  --append-system-prompt\\n")
    raise SystemExit(0)

stage = os.environ.get("KAGGLE_AGENT_STAGE", "")
if os.environ.get("FAKE_CLAUDE_INVALID_STAGE") == stage:
    print(json.dumps({"session_id": "claude-invalid", "structured_output": {"stage": stage}}))
    raise SystemExit(0)

payloads = {
    "report": {
        "stage": "report",
        "headline": "Adapter report headline",
        "focus": "submission calibration",
        "best_run_id": "run-best",
        "best_run_metric": 0.91,
        "primary_metric_value": 0.88,
        "root_cause": "healthy runtime",
        "verdict": "keep",
        "finding_titles": ["adapter finding"],
        "issue_titles": [],
        "markdown": "# Adapter Report\\n\\n- Healthy runtime."
    },
    "research": {
        "stage": "research",
        "root_cause": "healthy runtime",
        "queries": ["birdclef validation"],
        "adopt_now": ["promote cached probe baseline"],
        "consider": ["blend with priors"],
        "reject": [],
        "knowledge_files_seen": 1,
        "markdown": "# Adapter Research\\n\\n- Promote cached probe baseline."
    },
    "decision": {
        "stage": "decision",
        "decision_type": "promote_baseline",
        "next_action": "run_new_experiment",
        "submission_recommendation": "no",
        "root_cause": "healthy runtime",
        "why": "Promote the cached probe baseline.",
        "next_title": "Perch cached-probe baseline",
        "next_family": "perch_cached_probe",
        "next_config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
        "priority_delta": 10,
        "launch_mode": "background",
        "requires_human": False,
        "markdown": "# Adapter Decision\\n\\n- Promote baseline."
    },
    "critic": {
        "stage": "critic",
        "status": "approved",
        "concerns": [],
        "warnings": ["amp probe available"],
        "required_fixes": [],
        "markdown": "# Adapter Critic\\n\\n- Approved."
    },
}
payload = payloads[stage]
print(json.dumps({"session_id": f"claude-{stage}-session", "model": "claude-fake", "structured_output": payload}))
""",
    )
    _write_executable(
        bin_dir / "codex",
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if args[:2] == ["exec", "--help"] or args == ["exec", "--help"]:
    print("Run Codex non-interactively\\n  --output-schema\\n  --json\\n  --ephemeral\\n  --sandbox\\n  -o\\n")
    raise SystemExit(0)
if args[:3] == ["exec", "resume", "--help"] or args == ["exec", "resume", "--help"]:
    print("resume help")
    raise SystemExit(0)
if not args or args[0] != "exec":
    raise SystemExit(1)

response_path = None
for index, value in enumerate(args):
    if value in {"--output-last-message", "-o"} and index + 1 < len(args):
        response_path = Path(args[index + 1])
        break
if response_path is None:
    raise SystemExit(2)

stage = os.environ.get("KAGGLE_AGENT_STAGE", "")
workspace_root = Path(os.environ.get("KAGGLE_AGENT_WORKSPACE_ROOT", "."))
default_config = workspace_root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml"
default_yaml = default_config.read_text(encoding="utf-8")

if stage == "plan":
    payload = {
        "stage": "plan",
        "plan_status": "planned",
        "source_run_id": os.environ.get("KAGGLE_AGENT_RUN_ID", "run-unknown"),
        "reason": "Adapter planned the next cached probe run.",
        "title": "Adapter planned cached probe",
        "family": "perch_cached_probe",
        "hypothesis": "Use the stable cached probe baseline as the next validated branch.",
        "config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
        "priority": 70,
        "depends_on": ["workitem-perch-debug-smoke"],
        "tags": ["adapter", "planned"],
        "launch_mode": "background",
        "dedupe_key": "adapter:perch-cached-probe",
        "work_type": "experiment_iteration",
        "markdown": "# Adapter Plan\\n\\n- Queue cached probe baseline."
    }
else:
    payload = {
        "stage": "codegen",
        "status": "generated",
        "reason": "Generated a config copy and bundle payload.",
        "generated_config_yaml": default_yaml,
        "patch_diff": "diff --git a/generated.yaml b/generated.yaml\\n",
        "run_bundle": {
            "spec_type": "experiment",
            "title": "Adapter planned cached probe",
            "family": "perch_cached_probe",
            "config_path": "BirdCLEF-2026-Codebase/configs/default.yaml"
        },
        "markdown": "# Adapter Codegen\\n\\n- Materialized config and run bundle."
    }

response_path.write_text(json.dumps(payload), encoding="utf-8")
print(json.dumps({"event": "thread.started", "thread_id": f"codex-{stage}-thread"}))
print(json.dumps({"event": "turn.completed"}))
""",
    )
    _write_executable(
        bin_dir / "amp",
        """#!/usr/bin/env python3
import json
import sys

args = sys.argv[1:]
if "--help" in args:
    print("Amp CLI\\n  -x, --execute\\n  --stream-json\\nEnvironment variables:\\n  AMP_API_KEY\\n")
    raise SystemExit(0)
if args[:3] == ["threads", "continue", "--help"]:
    print("Usage: amp threads continue [threadId]\\n  --last\\n")
    raise SystemExit(0)
print(json.dumps({
    "type": "assistant",
    "threadId": "amp-thread-1",
    "message": {
        "content": [{"type": "text", "text": "Amp sidecar: no additional critic blocks."}]
    }
}))
""",
    )


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

    def test_adapter_wrappers_soft_skip_when_binaries_or_keys_are_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root, adapter_commands=_adapter_commands())
            _build_debug_dataset(root)

            empty_bin = root / "empty-bin"
            empty_bin.mkdir()
            env = {
                "PATH": str(empty_bin),
                "ANTHROPIC_API_KEY": "",
                "CODEX_API_KEY": "",
                "OPENAI_API_KEY": "",
                "AMP_API_KEY": "",
            }
            with patch.dict(os.environ, env, clear=False):
                config = load_config(root)
                init_workspace(config, archive_legacy=False, force=True)
                run_id = start_next(config, background=False)

            state = load_state(config)
            run = next(item for item in state.runs if item.run_id == run_id)
            self.assertEqual(run.stage_cursor, "complete")
            self.assertEqual(run.stage_error, "")
            stage_agent_statuses = {item.agent_role: item.status for item in state.agent_runs}
            self.assertEqual(stage_agent_statuses["report"], "skipped")
            self.assertEqual(stage_agent_statuses["plan"], "skipped")
            self.assertEqual(stage_agent_statuses["critic"], "skipped")

    def test_adapter_wrappers_populate_metadata_and_materialize_codegen_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root, adapter_commands=_adapter_commands())
            _build_debug_dataset(root)

            bin_dir = root / "fake-bin"
            bin_dir.mkdir()
            _install_fake_providers(bin_dir)
            env = {
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "CODEX_API_KEY": "test-codex-key",
                "AMP_API_KEY": "test-amp-key",
            }
            with patch.dict(os.environ, env, clear=False):
                config = load_config(root)
                init_workspace(config, archive_legacy=False, force=True)
                run_id = start_next(config, background=False)

            state = load_state(config)
            run = next(item for item in state.runs if item.run_id == run_id)
            self.assertEqual(run.stage_cursor, "complete")
            self.assertEqual(run.stage_error, "")

            agent_by_role = {item.agent_role: item for item in state.agent_runs}
            self.assertEqual(agent_by_role["report"].provider, "claude")
            self.assertEqual(agent_by_role["report"].model, "claude-fake")
            self.assertTrue(agent_by_role["report"].raw_stdout_path)
            self.assertEqual(agent_by_role["plan"].provider, "codex")
            self.assertTrue(agent_by_role["plan"].thread_id.startswith("codex-plan"))
            self.assertTrue(agent_by_role["critic"].provider_meta_path)

            plan_stage = next(item for item in state.stage_runs if item.run_id == run_id and item.stage_name == "plan")
            codegen_stage = next(item for item in state.stage_runs if item.run_id == run_id and item.stage_name == "codegen")
            self.assertTrue(plan_stage.spec_path)
            self.assertTrue(Path(plan_stage.spec_path).exists())

            codegen_payload = json.loads(Path(codegen_stage.output_json_path).read_text(encoding="utf-8"))
            self.assertEqual(codegen_payload["status"], "generated")
            self.assertTrue(Path(codegen_payload["generated_config_path"]).exists())
            self.assertTrue(Path(codegen_payload["run_bundle_path"]).exists())
            self.assertTrue(Path(codegen_payload["patch_path"]).exists())

            critic_meta = json.loads(Path(agent_by_role["critic"].provider_meta_path).read_text(encoding="utf-8"))
            self.assertIn("amp_probe_path", critic_meta)
            self.assertTrue(Path(critic_meta["amp_probe_path"]).exists())
            self.assertEqual(critic_meta["amp_probe_summary"], "Amp sidecar: no additional critic blocks.")

            validation = state.validations[-1]
            self.assertEqual(validation.status, "validated")
            ready_items = list_ready_work_items(config)
            self.assertIn("workitem-perch-baseline", ready_items)
            derived_items = [item for item in state.work_items if item.source_run_id == run_id]
            self.assertTrue(derived_items)
            self.assertTrue(all(item.latest_spec_id for item in derived_items))

    def test_adapter_wrapper_fails_on_schema_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root, adapter_commands=_adapter_commands())
            _build_debug_dataset(root)

            bin_dir = root / "fake-bin"
            bin_dir.mkdir()
            _install_fake_providers(bin_dir)
            env = {
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "CODEX_API_KEY": "test-codex-key",
                "AMP_API_KEY": "test-amp-key",
                "FAKE_CLAUDE_INVALID_STAGE": "report",
            }
            with patch.dict(os.environ, env, clear=False):
                config = load_config(root)
                init_workspace(config, archive_legacy=False, force=True)
                run_id = start_next(config, background=False)

            state = load_state(config)
            run = next(item for item in state.runs if item.run_id == run_id)
            self.assertEqual(run.stage_cursor, "report")
            self.assertIn("schema validation failed", run.stage_error)
            report_stage = next(item for item in state.stage_runs if item.run_id == run_id and item.stage_name == "report")
            self.assertEqual(report_stage.status, "failed")


if __name__ == "__main__":
    unittest.main()
