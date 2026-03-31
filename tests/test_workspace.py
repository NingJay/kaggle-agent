from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from kaggle_agent.adapters.command import CommandAdapterTimeout, run_stage_adapter
from kaggle_agent.adapters.providers.claude_code_exec import run_claude_code_exec
from kaggle_agent.adapters.providers.codex_exec import run_codex_exec
from kaggle_agent.adapters.stage_wrapper import CodegenWorkspace, StageContext, _build_prompt, _verify_codegen_workspace
from kaggle_agent.cli import main as cli_main
from kaggle_agent.control.executor import collect_finished_runs, start_run
from kaggle_agent.control.scheduler import choose_next_work_items
from kaggle_agent.control.monitor import _process_run_stage_chain, _run_validate_stage, process_completed_runs
from kaggle_agent.decision.codegen import build_codegen
from kaggle_agent.decision.planner import build_plan
from kaggle_agent.knowledge import render_retrieved_knowledge, retrieve_knowledge_bundle
from kaggle_agent.layout import run_label
from kaggle_agent.control.store import load_state
from kaggle_agent.schema import ExperimentSpec, RunRecord, RuntimeState, StageRun, WorkspaceState
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
primary = "val_soundscape_macro_roc_auc"
secondary = ["soundscape_macro_roc_auc", "padded_cmap"]

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
seed_notebook_path = "/tmp/ref_notebook/simplerun-perch-v2embedprobe-bayesian-0-912.ipynb"
allow_debug_preflight = false

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
from pathlib import Path

args = sys.argv[1:]
if "--help" in args:
    print("Usage: claude [options]\\n  --output-format\\n  --json-schema\\n  --tools\\n  --no-session-persistence\\n  --append-system-prompt\\n  --dangerously-skip-permissions\\n")
    raise SystemExit(0)

stage = os.environ.get("KAGGLE_AGENT_STAGE", "")
if os.environ.get("FAKE_CLAUDE_INVALID_STAGE") == stage:
    print(json.dumps({"session_id": "claude-invalid", "structured_output": {"stage": stage}}))
    raise SystemExit(0)

workspace_root = Path.cwd()
default_config = workspace_root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml"
default_yaml = default_config.read_text(encoding="utf-8") if default_config.exists() else ""

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
    "plan": {
        "stage": "plan",
        "plan_status": "planned",
        "source_run_id": os.environ.get("KAGGLE_AGENT_RUN_ID", "run-unknown"),
        "reason": "Claude Code planned the next cached probe run.",
        "title": "Claude planned cached probe",
        "family": "perch_cached_probe",
        "hypothesis": "Use the stable cached probe baseline as the next validated branch.",
        "config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
        "priority": 70,
        "depends_on": ["workitem-perch-baseline"],
        "tags": ["adapter", "planned"],
        "launch_mode": "background",
        "dedupe_key": "adapter:perch-cached-probe",
        "work_type": "experiment_iteration",
        "markdown": "# Adapter Plan\\n\\n- Queue cached probe baseline."
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
if stage == "codegen" and "--json-schema" not in args:
    generated_config = workspace_root / "BirdCLEF-2026-Codebase" / "configs" / "generated" / "adapter_codegen.yaml"
    generated_config.parent.mkdir(parents=True, exist_ok=True)
    generated_config.write_text(default_yaml + "\\n# claude-code agentic codegen\\n", encoding="utf-8")
    train_path = workspace_root / "train_sed.py"
    train_path.write_text(
        '''from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    run_dir = Path(os.environ["KAGGLE_AGENT_RUN_DIR"])
    (run_dir / "code_state_marker.txt").write_text("claude-worktree-active\\\\n", encoding="utf-8")
    payload = {
        "experiment_name": "claude_codegen_worktree",
        "config_path": os.environ.get("KAGGLE_AGENT_SPEC_ID", ""),
        "primary_metric_name": "soundscape_macro_roc_auc",
        "primary_metric_value": 0.42,
        "secondary_metrics": {"soundscape_macro_roc_auc": 0.42, "padded_cmap": 0.24},
        "all_metrics": {
            "soundscape_macro_roc_auc": 0.42,
            "val_soundscape_macro_roc_auc": 0.42,
            "padded_cmap": 0.24,
        },
        "root_cause": "worktree-applied",
        "verdict": "worktree-success",
        "artifacts": {},
        "dataset_summary": {},
        "summary_markdown": "Claude Code worktree path executed successfully.",
    }
    (run_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )
    print(json.dumps({"session_id": "claude-code-codegen-session", "model": "claude-code-fake", "result": "edited isolated stage workspace"}))
    raise SystemExit(0)

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
workspace_arg = None
for index, value in enumerate(args):
    if value in {"--output-last-message", "-o"} and index + 1 < len(args):
        response_path = Path(args[index + 1])
    if value == "-C" and index + 1 < len(args):
        workspace_arg = Path(args[index + 1])

stage = os.environ.get("KAGGLE_AGENT_STAGE", "")
workspace_root = workspace_arg or Path(os.environ.get("KAGGLE_AGENT_WORKSPACE_ROOT", "."))
default_config = workspace_root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml"
default_yaml = default_config.read_text(encoding="utf-8")

if stage == "plan":
    if response_path is None:
        raise SystemExit(2)
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
        "depends_on": ["workitem-perch-baseline"],
        "tags": ["adapter", "planned"],
        "launch_mode": "background",
        "dedupe_key": "adapter:perch-cached-probe",
        "work_type": "experiment_iteration",
        "markdown": "# Adapter Plan\\n\\n- Queue cached probe baseline."
    }
    response_path.write_text(json.dumps(payload), encoding="utf-8")
elif response_path is not None:
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
else:
    generated_config = workspace_root / "BirdCLEF-2026-Codebase" / "configs" / "generated" / "adapter_codegen.yaml"
    generated_config.parent.mkdir(parents=True, exist_ok=True)
    generated_config.write_text(default_yaml + "\\n# adapter agentic codegen\\n", encoding="utf-8")
    train_path = workspace_root / "train_sed.py"
    train_path.write_text(
        '''from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    run_dir = Path(os.environ["KAGGLE_AGENT_RUN_DIR"])
    (run_dir / "code_state_marker.txt").write_text("snapshot-active\\\\n", encoding="utf-8")
    payload = {
        "experiment_name": "agentic_codegen_snapshot",
        "config_path": os.environ.get("KAGGLE_AGENT_SPEC_ID", ""),
        "primary_metric_name": "soundscape_macro_roc_auc",
        "primary_metric_value": 0.42,
        "secondary_metrics": {"soundscape_macro_roc_auc": 0.42, "padded_cmap": 0.24},
        "all_metrics": {
            "soundscape_macro_roc_auc": 0.42,
            "val_soundscape_macro_roc_auc": 0.42,
            "padded_cmap": 0.24,
        },
        "root_cause": "snapshot-applied",
        "verdict": "snapshot-success",
        "artifacts": {},
        "dataset_summary": {},
        "summary_markdown": "Snapshot code path executed successfully.",
    }
    (run_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )
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
    def test_run_label_truncates_long_titles_with_stable_hash(self) -> None:
        title = (
            "Perch probe round 8 replace linear head with 1 hidden layer MLP embedding dim 256 "
            "num classes relu on run 0002 leader code state with extra calibration audit"
        )
        label = run_label("run-0008-0007", title)

        self.assertTrue(label.startswith("run-0008-0007__"))
        self.assertLessEqual(len(label), 120)
        self.assertRegex(label, r"run-0008-0007__.+-[0-9a-f]{10}$")

    def test_run_stage_adapter_times_out(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "adapter-output"
            output_dir.mkdir(parents=True)
            input_manifest_path = root / "input_manifest.json"
            input_manifest_path.write_text("{}", encoding="utf-8")

            with self.assertRaises(CommandAdapterTimeout):
                run_stage_adapter(
                    f'{sys.executable} -c "import time; time.sleep(5)"',
                    stage="critic",
                    workspace_root=root,
                    input_manifest_path=input_manifest_path,
                    output_dir=output_dir,
                    timeout_seconds=1,
                )

    def test_codegen_verify_timeout_returns_failed_status(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_root = root / "BirdCLEF-2026-Codebase"
            (runtime_root / "configs" / "generated").mkdir(parents=True)
            config_source = runtime_root / "configs" / "generated" / "verify-timeout.yaml"
            config_source.write_text("experiment_name: verify_timeout\n", encoding="utf-8")
            train_entrypoint = root / "train_sed.py"
            train_entrypoint.write_text(
                "from __future__ import annotations\n"
                "import time\n"
                "time.sleep(5)\n",
                encoding="utf-8",
            )
            codegen_workspace = CodegenWorkspace(
                snapshot_root=root / "state" / "snapshots" / "codegen",
                workspace_root=root,
                verify_root=root / "state" / "snapshots" / "codegen" / "verify_runtime",
                base_commit="abc123",
                expected_config_relpath="BirdCLEF-2026-Codebase/configs/generated/verify-timeout.yaml",
            )
            codegen_workspace.verify_root.mkdir(parents=True, exist_ok=True)

            with patch.dict(os.environ, {"KAGGLE_AGENT_VERIFY_TIMEOUT_SECONDS": "1"}, clear=False):
                status, verify_command, verify_summary = _verify_codegen_workspace(codegen_workspace, config_source)

            self.assertEqual(status, "failed")
            self.assertIn("train_sed.py", verify_command)
            self.assertIn("timed out", verify_summary.lower())

    def test_claude_code_exec_agentic_timeout_returns_salvage_response(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            _write_executable(
                bin_dir / "claude",
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "import sys\n"
                "import time\n"
                "if '--help' in sys.argv:\n"
                "    print('Usage: claude --output-format --no-session-persistence --disable-slash-commands --no-chrome --add-dir')\n"
                "    raise SystemExit(0)\n"
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                "time.sleep(5)\n",
            )
            started = time.monotonic()
            with patch.dict(os.environ, {"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", "KAGGLE_AGENT_PROVIDER_TIMEOUT_SECONDS": "1"}, clear=False):
                response = run_claude_code_exec(
                    prompt="touch a file",
                    schema_path=None,
                    workspace_root=root,
                    mode="agentic",
                )
            elapsed = time.monotonic() - started

            self.assertEqual(response.exit_code, 124)
            self.assertEqual(response.extra_meta.get("timed_out"), "1")
            self.assertIn("timed out", response.raw_stderr.lower())
            self.assertLess(elapsed, 10)

    def test_codex_exec_agentic_timeout_returns_salvage_response(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            output_dir = root / "output"
            bin_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_executable(
                bin_dir / "codex",
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "import sys\n"
                "import time\n"
                "if len(sys.argv) > 1 and sys.argv[1] == 'exec':\n"
                "    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                "    time.sleep(5)\n"
                "    raise SystemExit(0)\n"
                "raise SystemExit(0)\n",
            )
            started = time.monotonic()
            with patch.dict(os.environ, {"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", "KAGGLE_AGENT_PROVIDER_TIMEOUT_SECONDS": "1"}, clear=False):
                response = run_codex_exec(
                    prompt="edit files",
                    schema_path=None,
                    workspace_root=root,
                    output_dir=output_dir,
                    mode="agentic",
                )
            elapsed = time.monotonic() - started

            self.assertEqual(response.exit_code, 124)
            self.assertEqual(response.extra_meta.get("timed_out"), "1")
            self.assertIn("timed out", response.raw_stderr.lower())
            self.assertLess(elapsed, 10)

    def test_runtime_verify_mode_does_not_mirror_outputs_back_into_source_tree(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _build_debug_dataset(root)

            verify_root = root / "verify-runtime"
            env = os.environ.copy()
            env.update(
                {
                    "KAGGLE_AGENT_RUN_DIR": str(verify_root),
                    "KAGGLE_AGENT_DISABLE_OUTPUT_MIRROR": "1",
                    "KAGGLE_AGENT_VERIFY_MODE": "1",
                }
            )
            completed = subprocess.run(
                [sys.executable, "train_sed.py", "--config", "configs/debug.yaml"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
            self.assertTrue((verify_root / "result.json").exists())
            self.assertFalse((root / "BirdCLEF-2026-Codebase" / "outputs" / "perch_head_debug").exists())

    def test_retrieved_knowledge_bundle_surfaces_positive_negative_and_conditional_priors(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)
            knowledge_root = root / "knowledge"
            knowledge_root.mkdir(parents=True, exist_ok=True)
            (knowledge_root / "01_validated_findings.md").write_text(
                "# Findings\n\n"
                "## Soft pseudo labels are the biggest win\n\n"
                "Soft pseudo labels improved validation and remain the strongest positive prior.\n\n"
                "## Calibration-only sweep hurts\n\n"
                "Pure calibration-only sweeps regressed holdout validation and should be vetoed.\n",
                encoding="utf-8",
            )
            (knowledge_root / "03_next_experiment_priors.md").write_text(
                "# Priors\n\n"
                "## Class coverage before calibration\n\n"
                "If class imbalance is the blocking issue, expand coverage first and only then revisit calibration.\n",
                encoding="utf-8",
            )

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            bundle = retrieve_knowledge_bundle(
                config,
                {
                    "run": {"run_id": "run-knowledge", "primary_metric_name": "val_soundscape_macro_roc_auc"},
                    "experiment": {"family": "perch_cached_probe", "title": "Probe baseline"},
                    "report": {"root_cause": "class imbalance on holdout", "focus": "class coverage"},
                },
                stage="plan",
            )
            rendered = render_retrieved_knowledge(bundle)

            self.assertGreaterEqual(bundle["knowledge_files_seen"], 2)
            self.assertIn("Positive Priors", rendered)
            self.assertIn("Negative Vetoes", rendered)
            self.assertIn("Conditional Leads", rendered)
            self.assertTrue(bundle["knowledge_card_ids"])

    def test_build_plan_materializes_multi_branch_configs_from_fallback_search(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)
            knowledge_root = root / "knowledge"
            knowledge_root.mkdir(parents=True, exist_ok=True)
            (knowledge_root / "01_validated_findings.md").write_text(
                "# Findings\n\n"
                "## Coverage fixes help\n\n"
                "Long-tail coverage fixes improved validation.\n",
                encoding="utf-8",
            )
            (knowledge_root / "03_next_experiment_priors.md").write_text(
                "# Priors\n\n"
                "## Coverage before calibration\n\n"
                "Address coverage before calibration-only tuning when class imbalance is visible.\n",
                encoding="utf-8",
            )

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            work_item = next(item for item in state.work_items if item.id == "workitem-perch-baseline")
            experiment = ExperimentSpec(
                id="exp-plan-fallback",
                title="Perch cached-probe baseline",
                hypothesis="baseline",
                family=work_item.family,
                config_path=work_item.config_path,
                priority=work_item.priority,
                work_item_id=work_item.id,
                spec_id="spec-seed",
            )
            state.experiments.append(experiment)
            run = RunRecord(
                run_id="run-plan-fallback",
                experiment_id=experiment.id,
                work_item_id=work_item.id,
                spec_id="spec-seed",
                status="succeeded",
                command="",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-plan-fallback"),
                log_path=str(root / "artifacts" / "runs" / "run-plan-fallback" / "train.log"),
                primary_metric_name="val_soundscape_macro_roc_auc",
                primary_metric_value=0.61,
            )
            state.runs.append(run)

            def _latest_payload(_state, _run_id, stage_name):
                if stage_name == "decision":
                    return {
                        "stage": "decision",
                        "next_action": "run_new_experiment",
                        "why": "Keep searching higher-value branches.",
                        "next_title": "Perch cached-probe baseline follow-up",
                        "next_family": "perch_cached_probe",
                        "next_config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
                        "priority_delta": 10,
                        "launch_mode": "background",
                    }
                if stage_name == "research":
                    return {
                        "stage": "research",
                        "root_cause": "class imbalance on holdout",
                        "adopt_now": ["coverage-first branch"],
                        "consider": ["probe-head capacity branch"],
                        "reject": ["calibration-only sweep"],
                    }
                return {}

            with patch("kaggle_agent.decision.planner.latest_stage_payload", side_effect=_latest_payload), patch(
                "kaggle_agent.decision.planner.run_configured_stage_adapter",
                return_value=None,
            ):
                stage_run = build_plan(config, state, run.run_id)

            payload = json.loads(Path(stage_run.output_json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["plan_status"], "planned")
            self.assertGreaterEqual(len(payload["branch_plans"]), 2)
            self.assertEqual(payload["branch_plans"][0]["branch_role"], "primary")
            self.assertTrue(payload["portfolio_id"])
            for branch in payload["branch_plans"]:
                config_path = root / branch["config_path"]
                self.assertTrue(config_path.exists())
                self.assertTrue(branch["idea_class"])

    def test_build_plan_falls_back_when_decision_next_config_path_is_not_materialized(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            work_item = next(item for item in state.work_items if item.id == "workitem-perch-baseline")
            experiment = ExperimentSpec(
                id="exp-plan-missing-next-config",
                title="Perch cached-probe baseline",
                hypothesis="baseline",
                family=work_item.family,
                config_path=work_item.config_path,
                priority=work_item.priority,
                work_item_id=work_item.id,
                spec_id="spec-seed",
            )
            state.experiments.append(experiment)
            run = RunRecord(
                run_id="run-plan-missing-next-config",
                experiment_id=experiment.id,
                work_item_id=work_item.id,
                spec_id="spec-seed",
                status="succeeded",
                command="",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-plan-missing-next-config"),
                log_path=str(root / "artifacts" / "runs" / "run-plan-missing-next-config" / "train.log"),
                primary_metric_name="val_soundscape_macro_roc_auc",
                primary_metric_value=0.63,
            )
            state.runs.append(run)

            def _latest_payload(_state, _run_id, stage_name):
                if stage_name == "decision":
                    return {
                        "stage": "decision",
                        "next_action": "run_new_experiment",
                        "why": "Use the current config as the base even if the provider named a future config path.",
                        "next_title": "Perch cached-probe baseline follow-up",
                        "next_family": "perch_cached_probe",
                        "next_config_path": "BirdCLEF-2026-Codebase/configs/generated/plan-future-debug-branch.yaml",
                        "priority_delta": 10,
                        "launch_mode": "background",
                    }
                if stage_name == "research":
                    return {
                        "stage": "research",
                        "root_cause": "class imbalance on holdout",
                        "adopt_now": ["coverage-first branch"],
                        "consider": ["probe-head capacity branch"],
                        "reject": ["calibration-only sweep"],
                    }
                return {}

            with patch("kaggle_agent.decision.planner.latest_stage_payload", side_effect=_latest_payload), patch(
                "kaggle_agent.decision.planner.run_configured_stage_adapter",
                return_value=None,
            ):
                stage_run = build_plan(config, state, run.run_id)

            payload = json.loads(Path(stage_run.output_json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["plan_status"], "planned")
            self.assertGreaterEqual(len(payload["branch_plans"]), 1)
            for branch in payload["branch_plans"]:
                config_path = root / branch["config_path"]
                self.assertTrue(config_path.exists())

    def test_validate_stage_registers_multiple_branch_work_items(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            work_item = next(item for item in state.work_items if item.id == "workitem-perch-baseline")
            experiment = ExperimentSpec(
                id="exp-validate-branches",
                title="Perch cached-probe baseline",
                hypothesis="baseline",
                family=work_item.family,
                config_path=work_item.config_path,
                priority=work_item.priority,
                work_item_id=work_item.id,
                spec_id="spec-seed",
            )
            state.experiments.append(experiment)
            run = RunRecord(
                run_id="run-validate-branches",
                experiment_id=experiment.id,
                work_item_id=work_item.id,
                spec_id="spec-seed",
                status="succeeded",
                command="",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-validate-branches"),
                log_path=str(root / "artifacts" / "runs" / "run-validate-branches" / "train.log"),
                primary_metric_name="val_soundscape_macro_roc_auc",
                primary_metric_value=0.63,
            )
            state.runs.append(run)
            code_state_root = root / "state" / "snapshots" / "codegen" / "code-state"
            code_state_root.mkdir(parents=True, exist_ok=True)
            generated_root = root / "BirdCLEF-2026-Codebase" / "configs" / "generated"
            generated_root.mkdir(parents=True, exist_ok=True)
            branch_a = generated_root / "branch-a.yaml"
            branch_b = generated_root / "branch-b.yaml"
            branch_a.write_text((root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8")
            branch_b.write_text((root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8")

            def _latest_payload(_state, _run_id, stage_name):
                if stage_name == "plan":
                    return {
                        "stage": "plan",
                        "plan_status": "planned",
                        "source_run_id": run.run_id,
                        "reason": "Register sibling follow-up branches.",
                        "title": "Primary branch",
                        "family": "perch_cached_probe",
                        "hypothesis": "Primary branch hypothesis",
                        "config_path": str(branch_a.relative_to(root)),
                        "priority": 30,
                        "depends_on": [work_item.id],
                        "tags": ["planned", "branch-search"],
                        "launch_mode": "background",
                        "dedupe_key": "plan:branch-a",
                        "work_type": "experiment_iteration",
                        "portfolio_id": "portfolio-run-validate-branches",
                        "knowledge_card_ids": ["card-a"],
                        "branch_plans": [
                            {
                                "title": "Primary branch",
                                "family": "perch_cached_probe",
                                "hypothesis": "Primary branch hypothesis",
                                "reason": "Coverage-first branch",
                                "config_path": str(branch_a.relative_to(root)),
                                "priority": 30,
                                "depends_on": [work_item.id],
                                "tags": ["planned", "branch-search", "primary"],
                                "launch_mode": "background",
                                "dedupe_key": "plan:branch-a",
                                "work_type": "experiment_iteration",
                                "portfolio_id": "portfolio-run-validate-branches",
                                "idea_class": "class_coverage",
                                "branch_role": "primary",
                                "branch_rank": 0,
                                "knowledge_card_ids": ["card-a"],
                            },
                            {
                                "title": "Hedge branch",
                                "family": "perch_cached_probe",
                                "hypothesis": "Hedge branch hypothesis",
                                "reason": "Representation hedge",
                                "config_path": str(branch_b.relative_to(root)),
                                "priority": 32,
                                "depends_on": [work_item.id],
                                "tags": ["planned", "branch-search", "hedge"],
                                "launch_mode": "background",
                                "dedupe_key": "plan:branch-b",
                                "work_type": "experiment_iteration",
                                "portfolio_id": "portfolio-run-validate-branches",
                                "idea_class": "probe_head",
                                "branch_role": "hedge",
                                "branch_rank": 1,
                                "knowledge_card_ids": ["card-b"],
                            },
                        ],
                    }
                if stage_name == "codegen":
                    return {
                        "stage": "codegen",
                        "status": "generated",
                        "reason": "Generated code state.",
                        "generated_config_path": str(branch_a),
                        "run_bundle_path": "",
                        "patch_path": "",
                        "code_state_ref": str(code_state_root),
                        "verify_artifacts_ref": str(root / "state" / "verify"),
                        "verify_command": "python train_sed.py --config generated/branch-a.yaml",
                        "verify_status": "passed",
                        "verify_summary": "verify passed",
                        "worktree_path": str(code_state_root),
                        "base_commit": "abc123",
                        "head_commit": "def456",
                        "changed_files": ["train_sed.py"],
                        "provider_runtime": "claude_code mode:agentic",
                        "allowed_edit_roots": ["train_sed.py"],
                        "smoke_status": "passed",
                        "smoke_summary": "verify passed",
                    }
                if stage_name == "critic":
                    return {"stage": "critic", "status": "approved"}
                return {}

            with patch("kaggle_agent.control.monitor.latest_stage_payload", side_effect=_latest_payload):
                stage_run = _run_validate_stage(config, state, run.run_id)

            payload = json.loads(Path(stage_run.output_json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "validated")
            self.assertEqual(payload["branch_count"], 2)
            new_items = [item for item in state.work_items if item.id != "workitem-perch-baseline"]
            self.assertEqual(len(new_items), 2)
            self.assertEqual({item.portfolio_id for item in new_items}, {"portfolio-run-validate-branches"})
            self.assertEqual({item.branch_role for item in new_items}, {"primary", "hedge"})
            self.assertEqual({item.idea_class for item in new_items}, {"class_coverage", "probe_head"})
            self.assertTrue(all(item.latest_spec_id for item in new_items))

    def test_choose_next_work_items_returns_batch_up_to_capacity(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            config_path = str((root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml").relative_to(root))
            for index in range(2):
                state.work_items.append(
                    state.work_items[0].__class__(
                        id=f"workitem-extra-{index}",
                        title=f"Extra branch {index}",
                        work_type="experiment_iteration",
                        family="perch_cached_probe",
                        priority=25 + index,
                        status="queued",
                        config_path=config_path,
                        pipeline=list(state.work_items[0].pipeline),
                    )
                )
            state.runtime.active_run_ids = []
            config = config.__class__(
                root=config.root,
                competition=config.competition,
                metrics=config.metrics,
                data=config.data,
                paths=config.paths,
                automation=config.automation.__class__(**{**config.automation.__dict__, "max_active_runs": 2}),
                adapters=config.adapters,
                runtime=config.runtime,
                kaggle=config.kaggle,
                notes=config.notes,
            )

            selected = choose_next_work_items(config, state)
            self.assertEqual(len(selected), 2)

    def test_force_init_reset_clears_stale_workspace_state_and_seeds_attempt(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            (root / "artifacts" / "decision" / "stage-0004-decision").mkdir(parents=True)
            (root / "artifacts" / "decision" / "stage-0004-decision" / "decision.json").write_text("stale", encoding="utf-8")
            (root / "reports").mkdir()
            (root / "reports" / "old.html").write_text("legacy", encoding="utf-8")
            (root / "state" / "exports").mkdir(parents=True)
            (root / "state" / "exports" / "old.json").write_text("legacy", encoding="utf-8")
            for name in ["CHECKLIST.md", "JOURNAL.md", "FINDINGS.md", "ISSUES.md", "SUBMISSIONS.md"]:
                (root / name).write_text(f"stale {name}\n", encoding="utf-8")

            config = load_config(root)
            init_workspace(config, archive_legacy=True, force=True)

            state = load_state(config)
            legacy_root = root / "legacy"
            self.assertFalse(legacy_root.exists() and any(legacy_root.iterdir()))
            self.assertFalse((root / "artifacts" / "decision" / "stage-0004-decision" / "decision.json").exists())
            self.assertFalse((root / "reports" / "old.html").exists())
            self.assertFalse((root / "state" / "exports" / "old.json").exists())
            self.assertTrue((root / "state" / "ledger.db").exists())
            self.assertTrue((root / "reports" / "master_report.html").exists())
            self.assertTrue((root / "CHECKLIST.md").exists())
            self.assertTrue((root / "COMPETITION.md").exists())
            self.assertTrue((root / "prompts" / "report.md").exists())
            self.assertEqual(getattr(state.runtime, "current_attempt_slug", ""), "simplerun-perch-v2embedprobe-bayesian-0-912")
            self.assertNotIn("stale CHECKLIST.md", (root / "CHECKLIST.md").read_text(encoding="utf-8"))
            self.assertNotIn("stale JOURNAL.md", (root / "JOURNAL.md").read_text(encoding="utf-8"))

    def test_init_seeds_baseline_only_and_enqueue_preflight_requires_explicit_opt_in(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)

            work_item_ids = {item.id for item in state.work_items}
            self.assertNotIn("workitem-perch-debug-smoke", work_item_ids)
            self.assertIn("workitem-perch-baseline", work_item_ids)
            self.assertEqual(list_ready_work_items(config), ["workitem-perch-baseline"])

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(["--root", str(root), "enqueue-preflight"])
            self.assertEqual(exit_code, 2)

            state = load_state(config)
            self.assertFalse(any(item.work_type == "preflight_check" for item in state.work_items))

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(["--root", str(root), "enqueue-preflight", "--allow-debug"])
            self.assertEqual(exit_code, 0)

            state = load_state(config)
            preflight_items = [item for item in state.work_items if item.work_type == "preflight_check"]
            self.assertEqual(len(preflight_items), 1)
            self.assertIn("Preflight", preflight_items[0].title)
            self.assertTrue(preflight_items[0].config_path.endswith("configs/debug.yaml"))
            self.assertIn(preflight_items[0].id, list_ready_work_items(config))

    def test_sync_baseline_run_starts_from_baseline_without_default_preflight(self) -> None:
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
            self.assertEqual(run.work_item_id, "workitem-perch-baseline")
            self.assertEqual(run.stage_cursor, "complete")
            self.assertEqual(run.primary_metric_name, "val_soundscape_macro_roc_auc")

            stage_names = [item.stage_name for item in state.stage_runs if item.run_id == run_id]
            self.assertEqual(stage_names, ["evidence", "report", "research", "decision", "plan", "codegen", "critic", "validate", "submission"])
            baseline_item = next(item for item in state.work_items if item.id == "workitem-perch-baseline")
            self.assertEqual(baseline_item.latest_run_id, run_id)
            self.assertFalse(any(item.id == "workitem-perch-debug-smoke" for item in state.work_items))
            self.assertTrue((root / "reports" / "master_report.html").exists())
            self.assertTrue((root / "knowledge" / "experiment_conclusions.md").exists())
            self.assertTrue((root / "FINDINGS.md").exists())

    def test_collect_finished_runs_uses_terminal_artifacts_even_if_pid_still_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            cli_main(["--root", str(root), "enqueue-preflight", "--allow-debug"])
            state = load_state(config)
            preflight_item = next(item for item in state.work_items if item.work_type == "preflight_check")
            run = start_run(config, state, preflight_item.id, background=True)

            run_root = Path(run.run_dir)
            deadline = time.time() + 30
            while time.time() < deadline:
                result_path = run_root / "result.json"
                if (run_root / "exit_code.txt").exists() and result_path.exists():
                    try:
                        payload = json.loads(result_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict) and payload:
                        break
                time.sleep(0.2)
            else:
                self.fail("background baseline run did not finish within 30 seconds")

            run.pid = os.getpid()
            finished = collect_finished_runs(config, state)

            self.assertEqual([item.run_id for item in finished], [run.run_id])
            self.assertEqual(run.status, "succeeded")
            self.assertEqual(run.stage_cursor, "evidence")
            self.assertIsNone(run.pid)

    def test_artifact_layout_uses_attempt_centric_readable_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            cli_main(["--root", str(root), "enqueue-preflight", "--allow-debug"])
            run_id = start_next(config, background=False)

            state = load_state(config)
            run = next(item for item in state.runs if item.run_id == run_id)
            codegen_stage = next(item for item in state.stage_runs if item.run_id == run_id and item.stage_name == "codegen")

            attempt_root = root / "artifacts" / "attempts" / "simplerun-perch-v2embedprobe-bayesian-0-912"
            self.assertTrue(Path(run.run_dir).is_relative_to(attempt_root))
            self.assertIn("/runtime", run.run_dir)
            self.assertTrue(Path(codegen_stage.output_dir).is_relative_to(attempt_root))
            relative_stage = Path(codegen_stage.output_dir).relative_to(attempt_root / "runs")
            self.assertTrue(relative_stage.parts[0].startswith(f"{run_id}__"))
            self.assertEqual(relative_stage.parts[1], "stages")
            self.assertEqual(relative_stage.parts[2], "06-codegen__generated")
            self.assertNotIn("/artifacts/codegen/", codegen_stage.output_dir)

    def test_status_hides_debug_runs_by_default_and_can_include_them(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            cli_main(["--root", str(root), "enqueue-preflight", "--allow-debug"])
            run_id = start_next(config, background=False)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(["--root", str(root), "status"])
            self.assertEqual(exit_code, 0)
            status_text = stdout.getvalue()
            self.assertIn("Current attempt: simplerun-perch-v2embedprobe-bayesian-0-912", status_text)
            self.assertNotIn(f"{run_id}__", status_text)
            self.assertNotIn("06-codegen__generated", status_text)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(["--root", str(root), "status", "--include-debug"])
            self.assertEqual(exit_code, 0)
            debug_status_text = stdout.getvalue()
            self.assertIn(f"{run_id}__", debug_status_text)
            self.assertIn("06-codegen__generated", debug_status_text)

            checklist = (root / "CHECKLIST.md").read_text(encoding="utf-8")
            journal = (root / "JOURNAL.md").read_text(encoding="utf-8")
            self.assertIn("simplerun-perch-v2embedprobe-bayesian-0-912", checklist)
            self.assertIn("simplerun-perch-v2embedprobe-bayesian-0-912", journal)
            self.assertNotIn(f"{run_id}__", journal)
            self.assertNotIn("09-submission__skipped", journal)

    def test_status_and_journal_surfaces_show_attempt_and_readable_stage_labels(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            run_id = start_next(config, background=False)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(["--root", str(root), "status"])
            self.assertEqual(exit_code, 0)
            status_text = stdout.getvalue()
            self.assertIn("Current attempt: simplerun-perch-v2embedprobe-bayesian-0-912", status_text)
            self.assertIn(f"{run_id}__", status_text)
            self.assertIn("06-codegen__generated", status_text)

            checklist = (root / "CHECKLIST.md").read_text(encoding="utf-8")
            journal = (root / "JOURNAL.md").read_text(encoding="utf-8")
            self.assertIn("simplerun-perch-v2embedprobe-bayesian-0-912", checklist)
            self.assertIn("simplerun-perch-v2embedprobe-bayesian-0-912", journal)
            self.assertIn(f"{run_id}__", journal)
            self.assertIn("09-submission__skipped", journal)

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
            self.assertTrue(codegen_payload["code_state_ref"])
            self.assertTrue(Path(codegen_payload["code_state_ref"]).exists())
            self.assertEqual(codegen_payload["verify_status"], "passed")
            self.assertTrue(codegen_payload["verify_command"])
            self.assertTrue(codegen_payload["verify_artifacts_ref"])
            self.assertTrue(Path(codegen_payload["verify_artifacts_ref"]).exists())
            self.assertTrue((Path(codegen_payload["verify_artifacts_ref"]) / "result.json").exists())
            self.assertFalse((Path(codegen_payload["code_state_ref"]) / "result.json").exists())
            self.assertEqual(codegen_payload["smoke_status"], codegen_payload["verify_status"])
            self.assertEqual(codegen_payload["smoke_summary"], codegen_payload["verify_summary"])
            self.assertIn("train_sed.py", codegen_payload["allowed_edit_roots"])
            self.assertIn("BirdCLEF-2026-Codebase/configs", codegen_payload["allowed_edit_roots"])
            self.assertEqual(codegen_payload["provider_runtime"], "codex mode:agentic env:inherit")
            self.assertIn("train_sed.py", codegen_payload["changed_files"])
            patch_text = Path(codegen_payload["patch_path"]).read_text(encoding="utf-8")
            self.assertNotIn("GIT binary patch", patch_text)
            self.assertNotIn("BirdCLEF-2026-Codebase/outputs", patch_text)

            critic_meta = json.loads(Path(agent_by_role["critic"].provider_meta_path).read_text(encoding="utf-8"))
            self.assertIn("amp_probe_path", critic_meta)
            self.assertTrue(Path(critic_meta["amp_probe_path"]).exists())
            self.assertEqual(critic_meta["amp_probe_summary"], "Amp sidecar: no additional critic blocks.")

            codegen_meta = json.loads(Path(agent_by_role["codegen"].provider_meta_path).read_text(encoding="utf-8"))
            self.assertEqual(codegen_meta["provider_runtime"], codegen_payload["provider_runtime"])
            self.assertNotIn("isolated_home", codegen_meta)
            self.assertNotIn("codex_home", codegen_meta)
            self.assertTrue(Path(codegen_meta["stage_workspace_root"]).exists())
            self.assertEqual(codegen_meta["stage_workspace_mode"], "snapshot-repo")

            validation = state.validations[-1]
            self.assertEqual(validation.status, "validated")
            validated_spec = next(item for item in state.specs if item.spec_id == validation.spec_id)
            self.assertEqual(validated_spec.code_state_ref, codegen_payload["code_state_ref"])
            derived_items = [item for item in state.work_items if item.source_run_id == run_id]
            self.assertTrue(derived_items)
            self.assertTrue(all(item.latest_spec_id for item in derived_items))

            follow_up_run = start_run(config, state, derived_items[0].id, background=False)
            self.assertEqual(follow_up_run.status, "succeeded")
            self.assertTrue((Path(follow_up_run.run_dir) / "code_state_marker.txt").exists())

    def test_validate_stage_requires_passed_verify_for_code_state_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)

            output_dir = root / "artifacts" / "validate-output"
            output_dir.mkdir(parents=True)
            code_state_root = root / "state" / "snapshots" / "codegen" / "code-state"
            code_state_root.mkdir(parents=True)
            generated_config = root / "BirdCLEF-2026-Codebase" / "configs" / "generated" / "verify-gate.yaml"
            generated_config.parent.mkdir(parents=True, exist_ok=True)
            generated_config.write_text((root / "BirdCLEF-2026-Codebase" / "configs" / "debug.yaml").read_text(encoding="utf-8"), encoding="utf-8")

            run = RunRecord(
                run_id="run-validate-verify-gate",
                experiment_id="exp-validate-verify-gate",
                work_item_id="workitem-validate-verify-gate",
                spec_id="",
                status="succeeded",
                command="python train_sed.py",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-validate-verify-gate"),
                log_path=str(root / "artifacts" / "runs" / "run-validate-verify-gate" / "train.log"),
            )
            state = WorkspaceState(
                work_items=[],
                experiments=[],
                runs=[run],
                stage_runs=[],
                agent_runs=[],
                specs=[],
                validations=[],
                metrics=[],
                findings=[],
                issues=[],
                research_notes=[],
                submissions=[],
                submission_results=[],
                runtime=RuntimeState(initialized_at="2026-03-28T00:00:00Z", next_validation_number=1),
            )
            stage_run = StageRun(
                stage_run_id="stage-validate-verify-gate",
                run_id=run.run_id,
                work_item_id=run.work_item_id,
                stage_name="validate",
                status="running",
                input_ref=run.run_id,
                output_dir=str(output_dir),
                output_json_path=str(output_dir / "validate.json"),
                output_md_path=str(output_dir / "validate.md"),
            )
            captured: dict[str, object] = {}

            def _payload_for_stage(_state: WorkspaceState, _run_id: str, stage_name: str) -> dict[str, object]:
                if stage_name == "plan":
                    return {
                        "stage": "plan",
                        "plan_status": "planned",
                        "title": "Verify gate candidate",
                        "family": "perch_head_debug",
                        "config_path": "BirdCLEF-2026-Codebase/configs/generated/verify-gate.yaml",
                        "launch_mode": "background",
                        "dedupe_key": "verify-gate:test",
                    }
                if stage_name == "codegen":
                    return {
                        "stage": "codegen",
                        "status": "generated",
                        "generated_config_path": str(generated_config),
                        "run_bundle_path": str(output_dir / "missing-run-bundle.json"),
                        "patch_path": str(output_dir / "patch.diff"),
                        "code_state_ref": str(code_state_root),
                        "verify_status": "failed",
                        "verify_summary": "deterministic verify command failed",
                    }
                return {"stage": "critic", "status": "approved"}

            with patch("kaggle_agent.control.monitor.latest_stage_payload", side_effect=_payload_for_stage), patch(
                "kaggle_agent.control.monitor.begin_stage_run",
                return_value=(stage_run, output_dir / "input_manifest.json"),
            ), patch(
                "kaggle_agent.control.monitor.write_input_manifest",
                return_value=None,
            ), patch(
                "kaggle_agent.control.monitor.complete_stage_run",
                side_effect=lambda _stage_run, **kwargs: captured.update(kwargs),
            ), patch(
                "kaggle_agent.control.monitor.register_work_item",
            ) as register_mock:
                _run_validate_stage(config, state, run.run_id)

            self.assertEqual(captured["payload"]["status"], "failed")
            self.assertIn("verify", captured["payload"]["summary"])
            self.assertEqual(state.validations[-1].status, "failed")
            register_mock.assert_not_called()

    def test_process_completed_runs_reconciles_stale_active_run_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)

            run = RunRecord(
                run_id="run-stale-active-id",
                experiment_id="exp-stale-active-id",
                work_item_id="workitem-stale-active-id",
                spec_id="",
                status="succeeded",
                command="python train_sed.py",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-stale-active-id"),
                log_path=str(root / "artifacts" / "runs" / "run-stale-active-id" / "train.log"),
                stage_cursor="complete",
            )
            state = WorkspaceState(
                work_items=[],
                experiments=[],
                runs=[run],
                stage_runs=[],
                agent_runs=[],
                specs=[],
                validations=[],
                metrics=[],
                findings=[],
                issues=[],
                research_notes=[],
                submissions=[],
                submission_results=[],
                runtime=RuntimeState(
                    initialized_at="2026-03-31T00:00:00Z",
                    next_validation_number=1,
                    active_run_ids=[run.run_id],
                ),
            )

            process_completed_runs(config, state)

            self.assertEqual(state.runtime.active_run_ids, [])

    def test_process_run_stage_chain_returns_to_codegen_after_critic_reject(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)

            run = RunRecord(
                run_id="run-critic-repair-loop",
                experiment_id="exp-critic-repair-loop",
                work_item_id="workitem-critic-repair-loop",
                spec_id="",
                status="succeeded",
                command="python train_sed.py",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-critic-repair-loop"),
                log_path=str(root / "artifacts" / "runs" / "run-critic-repair-loop" / "train.log"),
                stage_cursor="critic",
            )
            prior_codegen = StageRun(
                stage_run_id="stage-codegen-0001",
                run_id=run.run_id,
                work_item_id=run.work_item_id,
                stage_name="codegen",
                status="completed",
                input_ref=run.run_id,
                output_dir=str(root / "artifacts" / "codegen"),
                output_json_path=str(root / "artifacts" / "codegen" / "codegen.json"),
                output_md_path=str(root / "artifacts" / "codegen" / "codegen.md"),
            )
            state = WorkspaceState(
                work_items=[],
                experiments=[],
                runs=[run],
                stage_runs=[prior_codegen],
                agent_runs=[],
                specs=[],
                validations=[],
                metrics=[],
                findings=[],
                issues=[],
                research_notes=[],
                submissions=[],
                submission_results=[],
                runtime=RuntimeState(initialized_at="2026-03-31T00:00:00Z", next_validation_number=1),
            )

            def _payload_for_stage(_state: WorkspaceState, _run_id: str, stage_name: str) -> dict[str, object]:
                if stage_name == "plan":
                    return {"stage": "plan", "plan_status": "planned"}
                if stage_name == "critic":
                    return {"stage": "critic", "status": "rejected"}
                return {}

            with patch("kaggle_agent.control.monitor.build_critic", return_value=None) as critic_mock, patch(
                "kaggle_agent.control.monitor.latest_stage_payload",
                side_effect=_payload_for_stage,
            ), patch(
                "kaggle_agent.control.monitor._run_validate_stage",
            ) as validate_mock, patch(
                "kaggle_agent.control.monitor.build_codegen",
            ) as codegen_mock:
                _process_run_stage_chain(config, state, run.run_id)

            critic_mock.assert_called_once()
            validate_mock.assert_not_called()
            codegen_mock.assert_not_called()
            self.assertEqual(run.stage_cursor, "codegen")

    def test_process_run_stage_chain_retries_retryable_stage_errors(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)

            run = RunRecord(
                run_id="run-critic-timeout-retry",
                experiment_id="exp-critic-timeout-retry",
                work_item_id="workitem-critic-timeout-retry",
                spec_id="",
                status="succeeded",
                command="python train_sed.py",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-critic-timeout-retry"),
                log_path=str(root / "artifacts" / "runs" / "run-critic-timeout-retry" / "train.log"),
                stage_cursor="critic",
                stage_error="critic adapter timed out after 1080s",
            )
            running_critic_stage = StageRun(
                stage_run_id="stage-critic-running",
                run_id=run.run_id,
                work_item_id=run.work_item_id,
                stage_name="critic",
                status="running",
                input_ref=run.run_id,
                output_dir=str(root / "artifacts" / "critic"),
                output_json_path=str(root / "artifacts" / "critic" / "critic.json"),
                output_md_path=str(root / "artifacts" / "critic" / "critic.md"),
            )
            state = WorkspaceState(
                work_items=[],
                experiments=[],
                runs=[run],
                stage_runs=[running_critic_stage],
                agent_runs=[],
                specs=[],
                validations=[],
                metrics=[],
                findings=[],
                issues=[],
                research_notes=[],
                submissions=[],
                submission_results=[],
                runtime=RuntimeState(initialized_at="2026-03-31T00:00:00Z", next_validation_number=1),
            )

            with patch(
                "kaggle_agent.control.monitor.build_critic",
                side_effect=CommandAdapterTimeout("critic adapter timed out after 1080s"),
            ) as critic_mock:
                _process_run_stage_chain(config, state, run.run_id)

            critic_mock.assert_called_once()
            self.assertEqual(run.stage_cursor, "critic")
            self.assertEqual(run.stage_error, "")
            self.assertEqual(state.stage_runs[-1].status, "failed")
            self.assertIn("timed out", state.stage_runs[-1].error)

    def test_build_codegen_retries_failed_verify_with_previous_attempt_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            config = load_config(root)

            output_dir = root / "artifacts" / "codegen-retry-output"
            output_dir.mkdir(parents=True)
            input_manifest_path = output_dir / "input_manifest.json"
            run = RunRecord(
                run_id="run-codegen-retry",
                experiment_id="exp-codegen-retry",
                work_item_id="workitem-codegen-retry",
                spec_id="",
                status="succeeded",
                command="python train_sed.py",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-codegen-retry"),
                log_path=str(root / "artifacts" / "runs" / "run-codegen-retry" / "train.log"),
            )
            state = WorkspaceState(
                work_items=[],
                experiments=[],
                runs=[run],
                stage_runs=[],
                agent_runs=[],
                specs=[],
                validations=[],
                metrics=[],
                findings=[],
                issues=[],
                research_notes=[],
                submissions=[],
                submission_results=[],
                runtime=RuntimeState(initialized_at="2026-03-31T00:00:00Z", next_validation_number=1),
            )
            stage_run = StageRun(
                stage_run_id="stage-codegen-retry",
                run_id=run.run_id,
                work_item_id=run.work_item_id,
                stage_name="codegen",
                status="running",
                input_ref=run.run_id,
                output_dir=str(output_dir),
                output_json_path=str(output_dir / "codegen.json"),
                output_md_path=str(output_dir / "codegen.md"),
            )
            plan_payload = {
                "stage": "plan",
                "plan_status": "planned",
                "title": "Retry verify failure",
                "family": "perch_cached_probe",
                "config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
                "launch_mode": "background",
                "dedupe_key": "codegen-retry:test",
            }
            failed_payload = {
                "stage": "codegen",
                "status": "generated",
                "reason": "First attempt broke cached probe calibration.",
                "generated_config_path": str(output_dir / "generated_config.yaml"),
                "run_bundle_path": str(output_dir / "run_bundle.json"),
                "patch_path": str(output_dir / "patch.diff"),
                "code_state_ref": str(root / "state" / "code-state"),
                "verify_artifacts_ref": str(root / "state" / "verify"),
                "verify_command": "python ./train_sed.py --config configs/default.yaml",
                "verify_status": "failed",
                "verify_summary": "Runtime failed: could not broadcast input array from shape (564,) into shape (708,)",
                "worktree_path": str(root / "state" / "code-state"),
                "base_commit": "abc123",
                "head_commit": "def456",
                "changed_files": [
                    "BirdCLEF-2026-Codebase/configs/default.yaml",
                    "BirdCLEF-2026-Codebase/src/birdclef_runtime/cached_probe.py",
                ],
                "provider_runtime": "claude_code mode:agentic",
                "allowed_edit_roots": [
                    "train_sed.py",
                    "BirdCLEF-2026-Codebase/configs",
                    "BirdCLEF-2026-Codebase/src",
                ],
                "smoke_status": "failed",
                "smoke_summary": "Runtime failed: could not broadcast input array from shape (564,) into shape (708,)",
            }
            repaired_payload = dict(failed_payload)
            repaired_payload.update(
                {
                    "reason": "Second attempt repaired the failing probe edit.",
                    "verify_status": "passed",
                    "verify_summary": "Verify run completed with val_soundscape_macro_roc_auc=0.681 and verdict=continue-iterating.",
                    "smoke_status": "passed",
                    "smoke_summary": "Verify run completed with val_soundscape_macro_roc_auc=0.681 and verdict=continue-iterating.",
                    "changed_files": ["BirdCLEF-2026-Codebase/configs/default.yaml"],
                }
            )

            captured_manifests: list[dict[str, object]] = []
            completed: dict[str, object] = {}
            adapter_payloads = [(failed_payload, "first"), (repaired_payload, "second")]

            def _capture_manifest(_path: Path, payload: dict[str, object]) -> None:
                captured_manifests.append(payload)

            with patch("kaggle_agent.decision.codegen.latest_stage_payload", return_value=plan_payload), patch(
                "kaggle_agent.decision.codegen.begin_stage_run",
                return_value=(stage_run, input_manifest_path),
            ), patch(
                "kaggle_agent.decision.codegen.write_input_manifest",
                side_effect=_capture_manifest,
            ), patch(
                "kaggle_agent.decision.codegen.run_configured_stage_adapter",
                side_effect=adapter_payloads,
            ) as adapter_mock, patch(
                "kaggle_agent.decision.codegen.complete_stage_run",
                side_effect=lambda _stage_run, **kwargs: completed.update(kwargs),
            ):
                build_codegen(config, state, run.run_id)

            self.assertEqual(adapter_mock.call_count, 2)
            self.assertEqual(len(captured_manifests), 2)
            self.assertEqual(captured_manifests[0]["codegen_attempt_number"], 1)
            self.assertNotIn("previous_codegen_attempt", captured_manifests[0])
            self.assertEqual(captured_manifests[1]["codegen_attempt_number"], 2)
            previous_attempt = captured_manifests[1]["previous_codegen_attempt"]
            self.assertEqual(previous_attempt["attempt_number"], 1)
            self.assertEqual(previous_attempt["verify_status"], "failed")
            self.assertIn("shape (564,)", previous_attempt["verify_summary"])
            self.assertIn("cached_probe.py", previous_attempt["changed_files"][1])
            self.assertEqual(completed["payload"]["verify_status"], "passed")
            self.assertEqual(completed["markdown"], "second")

    def test_codegen_prompt_surfaces_retry_context_and_repair_rules(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_path = root / "schemas" / "codegen.schema.json"
            schema_path.parent.mkdir(parents=True)
            schema_path.write_text("{}", encoding="utf-8")
            output_dir = root / "artifacts" / "codegen-output"
            output_dir.mkdir(parents=True)
            ctx = StageContext(
                stage="codegen",
                workspace_root=root,
                input_manifest_path=output_dir / "input_manifest.json",
                output_dir=output_dir,
                prompt_path=None,
                schema_path=schema_path,
                input_manifest={
                    "run": {"run_id": "run-codegen-prompt"},
                    "plan": {"config_path": "BirdCLEF-2026-Codebase/configs/default.yaml"},
                    "codegen_attempt_number": 2,
                    "previous_codegen_attempt": {
                        "attempt_number": 1,
                        "status": "generated",
                        "verify_status": "failed",
                        "verify_summary": "Runtime failed: shape mismatch in cached probe logits.",
                        "changed_files": [
                            "BirdCLEF-2026-Codebase/configs/default.yaml",
                            "BirdCLEF-2026-Codebase/src/birdclef_runtime/cached_probe.py",
                        ],
                    },
                    "previous_critic_attempt": {
                        "status": "rejected",
                        "concerns": ["Verify score regressed versus the leader baseline."],
                        "warnings": ["Do not keep the regressed runtime source edit."],
                        "required_fixes": ["Repair the bundle so critic can approve it."],
                    },
                },
            )
            codegen_workspace = CodegenWorkspace(
                snapshot_root=root / "state" / "worktrees" / "codegen" / "stage",
                workspace_root=root,
                verify_root=root / "state" / "worktrees" / "codegen" / "stage" / "verify_runtime",
                base_commit="abc123",
                expected_config_relpath="BirdCLEF-2026-Codebase/configs/default.yaml",
            )

            prompt = _build_prompt(ctx, codegen_workspace)

            self.assertIn("Retry Context", prompt)
            self.assertIn("shape mismatch in cached probe logits", prompt)
            self.assertIn("First repair the previous verify failure", prompt)
            self.assertIn("revert or narrow those source edits", prompt)
            self.assertIn("cached_probe.py", prompt)
            self.assertIn("Critic Feedback", prompt)
            self.assertIn("Verify score regressed versus the leader baseline", prompt)
            self.assertIn("Repair the bundle so critic can approve it", prompt)

    def test_build_codegen_includes_previous_critic_feedback_in_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            config = load_config(root)

            output_dir = root / "artifacts" / "codegen-critic-output"
            output_dir.mkdir(parents=True)
            input_manifest_path = output_dir / "input_manifest.json"
            run = RunRecord(
                run_id="run-codegen-critic-feedback",
                experiment_id="exp-codegen-critic-feedback",
                work_item_id="workitem-codegen-critic-feedback",
                spec_id="",
                status="succeeded",
                command="python train_sed.py",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-codegen-critic-feedback"),
                log_path=str(root / "artifacts" / "runs" / "run-codegen-critic-feedback" / "train.log"),
            )
            state = WorkspaceState(
                work_items=[],
                experiments=[],
                runs=[run],
                stage_runs=[],
                agent_runs=[],
                specs=[],
                validations=[],
                metrics=[],
                findings=[],
                issues=[],
                research_notes=[],
                submissions=[],
                submission_results=[],
                runtime=RuntimeState(initialized_at="2026-03-31T00:00:00Z", next_validation_number=1),
            )
            stage_run = StageRun(
                stage_run_id="stage-codegen-critic-feedback",
                run_id=run.run_id,
                work_item_id=run.work_item_id,
                stage_name="codegen",
                status="running",
                input_ref=run.run_id,
                output_dir=str(output_dir),
                output_json_path=str(output_dir / "codegen.json"),
                output_md_path=str(output_dir / "codegen.md"),
            )
            plan_payload = {
                "stage": "plan",
                "plan_status": "planned",
                "title": "Repair critic rejection",
                "family": "perch_cached_probe",
                "config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
                "launch_mode": "background",
                "dedupe_key": "codegen-critic-feedback:test",
            }
            critic_payload = {
                "stage": "critic",
                "status": "rejected",
                "concerns": ["The generated bundle regressed below the leader baseline."],
                "warnings": ["Revert the unsafe cached_probe.py edit unless it is strictly required."],
                "required_fixes": ["Repair the bundle so verify beats the current leader."],
                "amp_probe_summary": "No extra sidecar blocks.",
            }
            generated_payload = {
                "stage": "codegen",
                "status": "generated",
                "reason": "Generated a repaired bundle.",
                "generated_config_path": str(output_dir / "generated_config.yaml"),
                "run_bundle_path": str(output_dir / "run_bundle.json"),
                "patch_path": str(output_dir / "patch.diff"),
                "code_state_ref": str(root / "state" / "code-state"),
                "verify_artifacts_ref": str(root / "state" / "verify"),
                "verify_command": "python ./train_sed.py --config configs/default.yaml",
                "verify_status": "passed",
                "verify_summary": "Verify run completed with val_soundscape_macro_roc_auc=0.681 and verdict=continue-iterating.",
                "worktree_path": str(root / "state" / "code-state"),
                "base_commit": "abc123",
                "head_commit": "def456",
                "changed_files": ["BirdCLEF-2026-Codebase/configs/default.yaml"],
                "provider_runtime": "claude_code mode:agentic",
                "allowed_edit_roots": ["BirdCLEF-2026-Codebase/configs"],
                "smoke_status": "passed",
                "smoke_summary": "Verify run completed with val_soundscape_macro_roc_auc=0.681 and verdict=continue-iterating.",
            }

            captured_manifests: list[dict[str, object]] = []

            def _latest_payload(_state: WorkspaceState, _run_id: str, stage_name: str) -> dict[str, object]:
                if stage_name == "plan":
                    return plan_payload
                if stage_name == "critic":
                    return critic_payload
                return {}

            with patch("kaggle_agent.decision.codegen.latest_stage_payload", side_effect=_latest_payload), patch(
                "kaggle_agent.decision.codegen.begin_stage_run",
                return_value=(stage_run, input_manifest_path),
            ), patch(
                "kaggle_agent.decision.codegen.write_input_manifest",
                side_effect=lambda _path, payload: captured_manifests.append(payload),
            ), patch(
                "kaggle_agent.decision.codegen.run_configured_stage_adapter",
                return_value=(generated_payload, "generated"),
            ), patch(
                "kaggle_agent.decision.codegen.complete_stage_run",
                return_value=None,
            ):
                build_codegen(config, state, run.run_id)

            self.assertEqual(len(captured_manifests), 1)
            critic_attempt = captured_manifests[0]["previous_critic_attempt"]
            self.assertEqual(critic_attempt["status"], "rejected")
            self.assertIn("leader baseline", critic_attempt["concerns"][0])
            self.assertIn("Repair the bundle", critic_attempt["required_fixes"][0])

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
