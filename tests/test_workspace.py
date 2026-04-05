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
from kaggle_agent.branch_typing import compile_proposal_typing, compile_realized_typing, estimate_info_gain
from kaggle_agent.cli import main as cli_main
from kaggle_agent.control.lifecycle import resolve_lifecycle_template
from kaggle_agent.control.executor import collect_finished_runs, launch_execute_stage, start_run
from kaggle_agent.control.reporting import write_reports
from kaggle_agent.control.scheduler import choose_next_work_items
from kaggle_agent.control.monitor import _process_run_stage_chain, _run_validate_stage, process_completed_runs
from kaggle_agent.decision.codegen import build_codegen
from kaggle_agent.decision.critic import _apply_typing_contract, build_critic
from kaggle_agent.decision.planner import _prune_branch_candidates, build_plan
from kaggle_agent.knowledge import (
    build_problem_frame,
    ensure_knowledge_layout,
    render_retrieved_knowledge,
    retrieve_knowledge_bundle,
    retrieve_knowledge_bundle_from_root,
    synchronize_branch_memory,
)
from kaggle_agent.knowledge_reducer import active_search_envelope, record_search_envelope, synchronize_claims
from kaggle_agent.layout import run_label
from kaggle_agent.control.store import load_state, save_state
from kaggle_agent.schema import (
    BranchMemoryRecord,
    EvidenceLinkRecord,
    ExperimentSpec,
    RealizedTypingRecord,
    RunRecord,
    RuntimeState,
    SpecRecord,
    StageRun,
    ValidationRecord,
    WorkItem,
    WorkspaceState,
)
from kaggle_agent.service import (
    build_submission,
    doctor_checks,
    enqueue_config,
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
        "planner_command": f"{wrapper} --provider claude_code",
        "codegen_command": f"{wrapper} --provider claude_code",
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


def _write_ogg(path: Path, *, seconds: float, sample_rate: int = 32000, frequency: float = 440.0) -> None:
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(seconds * sample_rate)
    timeline = np.linspace(0, seconds, samples, endpoint=False, dtype=np.float32)
    waveform = (0.1 * np.sin(2.0 * np.pi * frequency * timeline)).astype(np.float32)
    sf.write(str(path), waveform, sample_rate)


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


def _empty_workspace_state() -> WorkspaceState:
    return WorkspaceState(
        runtime=RuntimeState(initialized_at="2026-04-02T00:00:00Z"),
        work_items=[],
        experiments=[],
        runs=[],
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
    )


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
print(json.dumps({"event": "thread.started", "thread_id": f"claude-code-{stage}-thread"}))
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
    def test_submission_like_branch_is_coerced_to_submission_lifecycle(self) -> None:
        payload = {
            "title": "Ship raw-probe CPU submission bundle",
            "branch_role": "submission",
            "idea_class": "baseline_submission",
            "work_type": "experiment_iteration",
            "tags": ["submission"],
        }
        self.assertEqual(resolve_lifecycle_template(payload), "submission_from_target_run")

    def test_submission_context_does_not_coerce_improvement_branch(self) -> None:
        payload = {
            "title": "Expand class coverage",
            "branch_role": "improvement",
            "idea_class": "class_coverage",
            "work_type": "experiment_iteration",
            "reason": "Deferred improvement branch after submission.",
            "hypothesis": "Coverage expansion should improve macro ROC-AUC after leaderboard anchoring.",
            "tags": ["planned", "branch-search", "improvement"],
        }
        self.assertEqual(resolve_lifecycle_template(payload), "recursive_experiment")

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
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
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

    def test_init_workspace_force_reseeds_legacy_knowledge_and_drops_stale_workspace_knowledge(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            stale_path = root / "knowledge" / "stale.md"
            stale_path.parent.mkdir(parents=True, exist_ok=True)
            stale_path.write_text("# stale\n", encoding="utf-8")

            legacy_root = Path(tmp) / "legacy_repo" / "knowledge"
            (legacy_root / "research").mkdir(parents=True, exist_ok=True)
            (legacy_root / "index").mkdir(parents=True, exist_ok=True)
            (legacy_root / "01_validated_findings.md").write_text(
                "# Legacy Findings\n\n## Coverage first\n\nCoverage should be expanded before calibration.\n",
                encoding="utf-8",
            )
            (legacy_root / "research" / "run-legacy.md").write_text(
                "# Legacy Run\n\n## Why it mattered\n\nThis prior branch established the class-coverage frontier.\n",
                encoding="utf-8",
            )
            (legacy_root / "index" / "cards.json").write_text("[]", encoding="utf-8")

            config = load_config(root)
            with patch.dict(os.environ, {"KAGGLE_AGENT_KNOWLEDGE_SEED_ROOTS": str(legacy_root)}, clear=False):
                init_workspace(config, archive_legacy=False, force=True)

            self.assertFalse(stale_path.exists())
            imported_root = root / "knowledge" / "imports" / "legacy-repo"
            self.assertTrue((imported_root / "01_validated_findings.md").exists())
            self.assertTrue((imported_root / "research" / "run-legacy.md").exists())
            self.assertFalse((imported_root / "index" / "cards.json").exists())

    def test_compile_knowledge_index_from_root_syncs_seeded_legacy_imports_with_source_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            legacy_root = Path(tmp) / "legacy_repo" / "knowledge"
            (legacy_root / "research").mkdir(parents=True, exist_ok=True)
            (legacy_root / "03_next_experiment_priors.md").write_text(
                "# Legacy Priors\n\n## Probe training change\n\nPrefer probe-surface changes before calibration-only sweeps.\n",
                encoding="utf-8",
            )

            config = load_config(root)
            with patch.dict(os.environ, {"KAGGLE_AGENT_KNOWLEDGE_SEED_ROOTS": str(legacy_root)}, clear=False):
                init_workspace(config, archive_legacy=False, force=True)
                imported_path = root / "knowledge" / "imports" / "legacy-repo" / "03_next_experiment_priors.md"
                self.assertTrue(imported_path.exists())
                imported_path.write_text("# Drifted Copy\n\n## Drift\n\nLocal imported mirror diverged.\n", encoding="utf-8")
                legacy_root.joinpath("03_next_experiment_priors.md").write_text(
                    "# Legacy Priors\n\n## Probe training change\n\nPrefer probe-surface changes before calibration-only sweeps.\n\n## Updated note\n\nKeep imported knowledge synchronized with the canonical source root.\n",
                    encoding="utf-8",
                )
                ensure_knowledge_layout(config)
                cards = retrieve_knowledge_bundle_from_root(
                    root,
                    {
                        "run": {"run_id": "run-knowledge-index"},
                        "experiment": {"family": "perch_cached_probe", "title": "Probe baseline"},
                    },
                    stage="research",
                )

            self.assertIn("Updated note", imported_path.read_text(encoding="utf-8"))
            self.assertNotIn("Drifted Copy", imported_path.read_text(encoding="utf-8"))
            self.assertTrue(any(item.get("source_label") == "legacy-repo" for item in cards.get("knowledge_sources", [])))
            imported_cards = [
                card
                for card in cards["cards"]
                if str(card.get("source_path", "")).startswith("imports/legacy-repo/03_next_experiment_priors.md")
            ]
            self.assertTrue(imported_cards)
            self.assertTrue(all(card.get("source_kind") == "imported" for card in imported_cards))
            self.assertTrue(all(card.get("source_label") == "legacy-repo" for card in imported_cards))
            self.assertTrue(all(card.get("comparison_path") == "03_next_experiment_priors.md" for card in imported_cards))

    def test_compile_knowledge_index_from_root_skips_generated_research_run_notes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            knowledge_root = root / "knowledge"
            (knowledge_root / "research").mkdir(parents=True, exist_ok=True)
            (knowledge_root / "01_validated_findings.md").write_text(
                "# Findings\n\n## Coverage first\n\nCoverage improvements are validated.\n",
                encoding="utf-8",
            )
            (knowledge_root / "research" / "run-0010-noisy.md").write_text(
                "# Research: run-0010\n\n## Key Metrics\n\nA table dump that should stay out of knowledge cards.\n",
                encoding="utf-8",
            )

            bundle = retrieve_knowledge_bundle_from_root(
                root,
                {
                    "run": {"run_id": "run-knowledge-index"},
                    "experiment": {"family": "perch_cached_probe", "title": "Probe baseline"},
                },
                stage="research",
            )

            source_paths = {str(card.get("source_path", "")) for card in bundle["cards"]}
            self.assertIn("01_validated_findings.md", source_paths)
            self.assertNotIn("research/run-0010-noisy.md", source_paths)

    def test_compile_knowledge_index_from_root_skips_imported_generated_research_run_notes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            legacy_root = Path(tmp) / "legacy_repo" / "knowledge"
            (legacy_root / "research").mkdir(parents=True, exist_ok=True)
            (legacy_root / "01_validated_findings.md").write_text(
                "# Legacy Findings\n\n## Coverage first\n\nCoverage should be expanded before calibration.\n",
                encoding="utf-8",
            )
            (legacy_root / "research" / "run-legacy.md").write_text(
                "# Legacy Run\n\n## Key Metrics\n\nA noisy per-run research note that should not become a card.\n",
                encoding="utf-8",
            )

            config = load_config(root)
            with patch.dict(os.environ, {"KAGGLE_AGENT_KNOWLEDGE_SEED_ROOTS": str(legacy_root)}, clear=False):
                init_workspace(config, archive_legacy=False, force=True)
                bundle = retrieve_knowledge_bundle_from_root(
                    root,
                    {
                        "run": {"run_id": "run-imported-index"},
                        "experiment": {"family": "perch_cached_probe", "title": "Probe baseline"},
                    },
                    stage="research",
                )

            source_paths = {str(card.get("source_path", "")) for card in bundle["cards"]}
            self.assertIn("imports/legacy-repo/01_validated_findings.md", source_paths)
            self.assertNotIn("imports/legacy-repo/research/run-legacy.md", source_paths)

    def test_compile_knowledge_index_from_root_skips_run_titled_sections_inside_curated_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            knowledge_root = root / "knowledge"
            knowledge_root.mkdir(parents=True, exist_ok=True)
            (knowledge_root / "experiment_conclusions.md").write_text(
                "# Experiment Conclusions\n\n"
                "## run-0008-perch-probe-leader\n\n"
                "A per-run ledger summary that should stay out of cards.\n\n"
                "## Structural takeaway\n\n"
                "Coverage expansion remains the strongest validated lever.\n",
                encoding="utf-8",
            )

            bundle = retrieve_knowledge_bundle_from_root(
                root,
                {
                    "run": {"run_id": "run-conclusions-index"},
                    "experiment": {"family": "perch_cached_probe", "title": "Probe baseline"},
                },
                stage="research",
            )

            titles = {str(card.get("title", "")) for card in bundle["cards"]}
            self.assertIn("Structural takeaway", titles)
            self.assertNotIn("run-0008-perch-probe-leader", titles)

    def test_init_workspace_imports_structured_and_curated_legacy_knowledge(self) -> None:
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            root = project_root / "worktrees" / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            structured_root = project_root / "knowledge_seek"
            (structured_root / "00_command_center").mkdir(parents=True, exist_ok=True)
            (structured_root / "02_canonical_knowledge").mkdir(parents=True, exist_ok=True)
            (structured_root / "00_command_center" / "00_CURRENT.md").write_text(
                "# Current\n\n## Frontier\n\nRebuild the v5-like SED chain.\n",
                encoding="utf-8",
            )
            (structured_root / "02_canonical_knowledge" / "confirmed_findings.md").write_text(
                "# Findings\n\n## Dual BCE\n\nClip-level plus frame-level BCE is the stable baseline.\n",
                encoding="utf-8",
            )

            legacy_root = project_root / "kaggle_agent" / "knowledge"
            legacy_root.mkdir(parents=True, exist_ok=True)
            (legacy_root / "01_validated_findings.md").write_text(
                "# Legacy Findings\n\n## Old note\n\nKeep validated legacy findings comparable with the structured knowledge base.\n",
                encoding="utf-8",
            )

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            manifest = json.loads((root / "knowledge" / "index" / "source_imports.json").read_text(encoding="utf-8"))

            self.assertEqual(len(manifest), 2)
            self.assertEqual(Path(manifest[0]["source_root"]).resolve(), structured_root.resolve())
            self.assertEqual(Path(manifest[1]["source_root"]).resolve(), legacy_root.resolve())
            imported_structured = root / "knowledge" / "imports" / "knowledge-seek"
            self.assertTrue((imported_structured / "00_command_center" / "00_CURRENT.md").exists())
            imported_legacy = root / "knowledge" / "imports" / "kaggle-agent"
            self.assertTrue((imported_legacy / "01_validated_findings.md").exists())

    def test_run_training_supports_tensorflow_sed_v5_backend(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            data_root = root / "BirdCLEF-2026-Codebase" / "birdclef-2026"
            for label, filename, frequency in [
                ("sp1", "a.ogg", 440.0),
                ("sp1", "b.ogg", 480.0),
                ("sp1", "c.ogg", 520.0),
                ("sp2", "d.ogg", 660.0),
                ("sp2", "e.ogg", 700.0),
                ("sp2", "f.ogg", 740.0),
            ]:
                _write_ogg(data_root / "train_audio" / label / filename, seconds=6.0, frequency=frequency)
            _write_ogg(data_root / "train_soundscapes" / "sound1.ogg", seconds=10.0, frequency=440.0)
            _write_ogg(data_root / "train_soundscapes" / "sound2.ogg", seconds=10.0, frequency=660.0)
            _write_csv(
                data_root / "train_soundscapes_labels.csv",
                [
                    ["filename", "start", "end", "primary_label"],
                    ["sound1.ogg", "00:00:00", "00:00:05", "sp1"],
                    ["sound1.ogg", "00:00:05", "00:00:10", "sp1"],
                    ["sound2.ogg", "00:00:00", "00:00:05", "sp2"],
                    ["sound2.ogg", "00:00:05", "00:00:10", "sp2"],
                ],
            )

            config_path = root / "BirdCLEF-2026-Codebase" / "configs" / "sed_v5_like.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  name: sed_v5_test",
                        "  seed: 7",
                        "paths:",
                        f"  data_root: {data_root}",
                        "  output_root: ./outputs",
                        "data:",
                        "  train_csv: train.csv",
                        "  taxonomy_csv: taxonomy.csv",
                        "  sample_submission_csv: sample_submission.csv",
                        "  train_audio_dir: train_audio",
                        "  train_soundscapes_dir: train_soundscapes",
                        "  train_soundscapes_labels_csv: train_soundscapes_labels.csv",
                        "  test_soundscapes_dir: test_soundscapes",
                        "  require_full_soundscapes: false",
                        "  max_train_rows: 4",
                        "  max_val_rows: 4",
                        "model:",
                        "  backbone_name: tiny_conv",
                        "  image_size: 64",
                        "  n_mels: 32",
                        "  n_fft: 256",
                        "  hop_length: 64",
                        "  dropout: 0.1",
                        "training:",
                        "  backend: tensorflow_sed_v5",
                        "  sample_rate: 32000",
                        "  chunk_duration: 5.0",
                        "  epochs: 1",
                        "  steps_per_epoch: 1",
                        "  batch_size: 2",
                        "  eval_batch_size: 1",
                        "  learning_rate: 0.001",
                        "  mixup_alpha: 0.0",
                        "  specaugment_num_masks: 0",
                        "  gain_db_range: 0.0",
                        "  gaussian_noise_std: 0.0",
                        "metrics:",
                        "  primary: val_soundscape_macro_roc_auc",
                        "  secondary:",
                        "    - soundscape_macro_roc_auc",
                        "    - padded_cmap",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, str(root / "BirdCLEF-2026-Codebase" / "train.py"), "--config", str(config_path)],
                cwd=root / "BirdCLEF-2026-Codebase",
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
            run_dir = root / "BirdCLEF-2026-Codebase" / "outputs" / "sed_v5_test"
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["verdict"], "baseline-ready")
            self.assertIn("val_soundscape_macro_roc_auc", result["all_metrics"])
            self.assertTrue((run_dir / "best_sed_v5.weights.h5").exists())

    def test_sed_v5_backbone_weights_hint_is_forwarded_to_keras(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            runtime_root = root / "BirdCLEF-2026-Codebase"
            src_root = runtime_root / "src"
            sys.path.insert(0, str(src_root))
            try:
                from birdclef_runtime.sed_v5 import SEDSettings, _build_backbone

                class DummyBackbone:
                    trainable = False

                with patch("tensorflow.keras.applications.EfficientNetB0", return_value=DummyBackbone()) as factory:
                    backbone = _build_backbone(
                        SEDSettings(
                            image_size=64,
                            backbone_name="tf_efficientnet_b0.ns_jft_in1k",
                            backbone_weights="imagenet",
                            backbone_trainable=False,
                        )
                    )

                self.assertIsInstance(backbone, DummyBackbone)
                self.assertEqual(factory.call_args.kwargs["weights"], "imagenet")
            finally:
                for module_name in list(sys.modules):
                    if module_name == "birdclef_runtime" or module_name.startswith("birdclef_runtime."):
                        sys.modules.pop(module_name, None)
                if str(src_root) in sys.path:
                    sys.path.remove(str(src_root))

    def test_run_training_supports_tensorflow_sed_v5_asl_and_soft_pseudo(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            data_root = root / "BirdCLEF-2026-Codebase" / "birdclef-2026"
            for label, filename, frequency in [
                ("sp1", "a.ogg", 440.0),
                ("sp1", "b.ogg", 480.0),
                ("sp1", "c.ogg", 520.0),
                ("sp2", "d.ogg", 700.0),
            ]:
                _write_ogg(data_root / "train_audio" / label / filename, seconds=6.0, frequency=frequency)
            for filename, frequency in [
                ("sound1.ogg", 440.0),
                ("sound2.ogg", 460.0),
                ("sound3.ogg", 660.0),
                ("sound4.ogg", 700.0),
            ]:
                _write_ogg(data_root / "train_soundscapes" / filename, seconds=10.0, frequency=frequency)
            _write_csv(
                data_root / "train_soundscapes_labels.csv",
                [
                    ["filename", "start", "end", "primary_label"],
                    ["sound1.ogg", "00:00:00", "00:00:05", "sp1"],
                    ["sound1.ogg", "00:00:05", "00:00:10", "sp1"],
                    ["sound2.ogg", "00:00:00", "00:00:05", "sp1"],
                    ["sound2.ogg", "00:00:05", "00:00:10", "sp1"],
                    ["sound3.ogg", "00:00:00", "00:00:05", "sp2"],
                    ["sound3.ogg", "00:00:05", "00:00:10", "sp2"],
                    ["sound4.ogg", "00:00:00", "00:00:05", "sp2"],
                    ["sound4.ogg", "00:00:05", "00:00:10", "sp2"],
                ],
            )
            pseudo_path = data_root / "soft_pseudo.csv"
            _write_csv(
                pseudo_path,
                [
                    ["row_id", "sp1", "sp2"],
                    ["sound1_5", "0.90", "0.10"],
                    ["sound1_10", "0.88", "0.12"],
                    ["sound2_5", "0.85", "0.15"],
                    ["sound2_10", "0.83", "0.17"],
                ],
            )

            config_path = root / "BirdCLEF-2026-Codebase" / "configs" / "sed_v24_like_test.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  name: sed_v24_like_test",
                        "  seed: 11",
                        "paths:",
                        f"  data_root: {data_root}",
                        "  output_root: ./outputs",
                        "data:",
                        "  train_csv: train.csv",
                        "  taxonomy_csv: taxonomy.csv",
                        "  sample_submission_csv: sample_submission.csv",
                        "  train_audio_dir: train_audio",
                        "  train_soundscapes_dir: train_soundscapes",
                        "  train_soundscapes_labels_csv: train_soundscapes_labels.csv",
                        "  test_soundscapes_dir: test_soundscapes",
                        "  require_full_soundscapes: false",
                        "  max_train_rows: 4",
                        "  max_val_rows: 4",
                        "  max_val_files: 2",
                        "  val_file_offset: 2",
                        f"  pseudo_source_paths:\n    - {pseudo_path}",
                        "  exclude_validation_soundscapes_from_pseudo: true",
                        "  pseudo_sampling_weight: 2.0",
                        "model:",
                        "  backbone_name: tiny_conv",
                        "  image_size: 64",
                        "  n_mels: 32",
                        "  n_fft: 256",
                        "  hop_length: 64",
                        "  dropout: 0.1",
                        "training:",
                        "  backend: tensorflow_sed_v5",
                        "  sample_rate: 32000",
                        "  chunk_duration: 5.0",
                        "  epochs: 1",
                        "  steps_per_epoch: 1",
                        "  batch_size: 2",
                        "  eval_batch_size: 1",
                        "  learning_rate: 0.001",
                        "  clip_loss_name: asl",
                        "  frame_loss_name: asl",
                        "  asl_gamma_neg: 4.0",
                        "  asl_gamma_pos: 0.0",
                        "  asl_clip: 0.05",
                        "  pseudo_loss_weight: 0.75",
                        "  mixup_alpha: 0.0",
                        "  specaugment_num_masks: 0",
                        "  gain_db_range: 0.0",
                        "  gaussian_noise_std: 0.0",
                        "metrics:",
                        "  primary: val_soundscape_macro_roc_auc",
                        "  secondary:",
                        "    - soundscape_macro_roc_auc",
                        "    - padded_cmap",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, str(root / "BirdCLEF-2026-Codebase" / "train.py"), "--config", str(config_path)],
                cwd=root / "BirdCLEF-2026-Codebase",
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
            run_dir = root / "BirdCLEF-2026-Codebase" / "outputs" / "sed_v24_like_test"
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["verdict"], "baseline-ready")
            self.assertEqual(result["dataset_summary"]["pseudo_example_count"], 4)
            self.assertIn("soft_pseudo", result["dataset_summary"]["train_sources"])
            self.assertIn("ASL", result["root_cause"])
            self.assertIn("soft pseudo", result["root_cause"])
            self.assertTrue((run_dir / "best_sed_v5.weights.h5").exists())

    def test_run_training_supports_tensorflow_sed_v5_inference_backend(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            runtime_root = root / "BirdCLEF-2026-Codebase"
            src_root = runtime_root / "src"
            sys.path.insert(0, str(src_root))
            try:
                from birdclef_runtime.training import run_training

                def _fake_inference(config: dict[str, object], _runtime_root: Path, output_dir: Path) -> dict[str, object]:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    submission_csv = output_dir / "sed_soundscape_predictions.csv"
                    submission_csv.write_text("row_id,sp1,sp2\ntest1_5,0.1,0.9\n", encoding="utf-8")
                    return {
                        "rows": 1,
                        "soundscapes": 1,
                        "submission_csv": str(submission_csv),
                    }

                config = {
                    "experiment": {"name": "sed_v5_infer_test"},
                    "_config_path": str(runtime_root / "configs" / "sed_v5_like_infer.yaml"),
                    "paths": {"data_root": str(runtime_root / "birdclef-2026"), "output_root": "./outputs"},
                    "data": {
                        "train_csv": "train.csv",
                        "taxonomy_csv": "taxonomy.csv",
                        "sample_submission_csv": "sample_submission.csv",
                        "train_audio_dir": "train_audio",
                        "train_soundscapes_dir": "train_soundscapes",
                        "train_soundscapes_labels_csv": "train_soundscapes_labels.csv",
                        "test_soundscapes_dir": "test_soundscapes",
                        "max_infer_files": 1,
                    },
                    "model": {
                        "backbone_name": "tiny_conv",
                        "image_size": 64,
                        "n_mels": 32,
                        "n_fft": 256,
                        "hop_length": 64,
                        "dropout": 0.1,
                        "checkpoint_path": "./outputs/sed_v5_like_baseline/best_sed_v5.weights.h5",
                    },
                    "training": {
                        "backend": "tensorflow_sed_v5_infer",
                        "sample_rate": 32000,
                        "chunk_duration": 5.0,
                    },
                    "metrics": {
                        "primary": "soundscapes_processed",
                        "secondary": ["inference_rows"],
                    },
                }

                with patch("birdclef_runtime.sed_v5.run_sed_soundscape_inference", side_effect=_fake_inference):
                    result = run_training(config, runtime_root)
            finally:
                if str(src_root) in sys.path:
                    sys.path.remove(str(src_root))

            run_dir = runtime_root / "outputs" / "sed_v5_infer_test"
            persisted = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["verdict"], "inference-ready")
            self.assertEqual(result["primary_metric_name"], "soundscapes_processed")
            self.assertEqual(persisted["verdict"], "inference-ready")
            self.assertEqual(persisted["primary_metric_name"], "soundscapes_processed")
            self.assertTrue((run_dir / "inference" / "sed_soundscape_predictions.csv").exists())

    def test_ensure_knowledge_layout_removes_stale_imported_seed_files_when_source_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            legacy_root = Path(tmp) / "legacy_repo" / "knowledge"
            (legacy_root / "research").mkdir(parents=True, exist_ok=True)
            (legacy_root / "research" / "run-legacy.md").write_text(
                "# Legacy Run\n\n## Why it mattered\n\nThis prior branch established the frontier.\n",
                encoding="utf-8",
            )

            config = load_config(root)
            with patch.dict(os.environ, {"KAGGLE_AGENT_KNOWLEDGE_SEED_ROOTS": str(legacy_root)}, clear=False):
                init_workspace(config, archive_legacy=False, force=True)
                imported_path = root / "knowledge" / "imports" / "legacy-repo" / "research" / "run-legacy.md"
                self.assertTrue(imported_path.exists())
                legacy_root.joinpath("research", "run-legacy.md").unlink()
                ensure_knowledge_layout(config)

            self.assertFalse(imported_path.exists())

    def test_save_state_dedupes_duplicate_realized_typing_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            state.realized_typings = [
                RealizedTypingRecord(
                    realized_typing_id="realized-0001",
                    run_id="run-0001",
                    stage_run_id="stage-0001",
                    title="first",
                    family="perch_cached_probe",
                    config_path="BirdCLEF-2026-Codebase/configs/default.yaml",
                    typing_signature="sig-a",
                    typing_payload={"axis_tags": ["coverage"]},
                    created_at="2026-04-03T00:00:00+00:00",
                ),
                RealizedTypingRecord(
                    realized_typing_id="realized-0001",
                    run_id="run-0001",
                    stage_run_id="stage-0002",
                    title="second",
                    family="perch_cached_probe",
                    config_path="BirdCLEF-2026-Codebase/configs/default.yaml",
                    typing_signature="sig-b",
                    typing_payload={"axis_tags": ["coverage", "pseudo"]},
                    created_at="2026-04-03T00:00:01+00:00",
                ),
            ]

            save_state(config, state)
            reloaded = load_state(config)

            self.assertEqual(len(reloaded.realized_typings), 1)
            self.assertEqual(reloaded.realized_typings[0].stage_run_id, "stage-0002")
            self.assertEqual(reloaded.realized_typings[0].typing_signature, "sig-b")

    def test_ensure_knowledge_layout_preserves_existing_source_import_manifest_without_seed_candidates(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir(parents=True, exist_ok=True)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            legacy_root = Path(tmp) / "legacy_repo" / "knowledge"
            legacy_root.mkdir(parents=True, exist_ok=True)
            (legacy_root / "01_validated_findings.md").write_text(
                "# Legacy Findings\n\n## Coverage first\n\nCoverage should be expanded before calibration.\n",
                encoding="utf-8",
            )

            config = load_config(root)
            with patch.dict(os.environ, {"KAGGLE_AGENT_KNOWLEDGE_SEED_ROOTS": str(legacy_root)}, clear=False):
                init_workspace(config, archive_legacy=False, force=True)

            manifest_path = root / "knowledge" / "index" / "source_imports.json"
            original_manifest = manifest_path.read_text(encoding="utf-8")

            with patch.dict(os.environ, {"KAGGLE_AGENT_KNOWLEDGE_SEED_ROOTS": ""}, clear=False):
                ensure_knowledge_layout(config)

            self.assertEqual(original_manifest, manifest_path.read_text(encoding="utf-8"))

    def test_retrieved_knowledge_bundle_includes_branch_memories_and_policy_contradictions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            knowledge_root = root / "knowledge"
            knowledge_root.mkdir(parents=True, exist_ok=True)
            (knowledge_root / "01_validated_findings.md").write_text(
                "# Findings\n\n"
                "## Coverage fixes are validated\n\n"
                "Coverage-first branches improved validation and should be preferred.\n\n"
                "## Calibration-only sweep hurts\n\n"
                "Calibration-only sweeps regressed holdout validation and should be vetoed.\n",
                encoding="utf-8",
            )
            state = load_state(config)
            state.branch_memories.append(
                BranchMemoryRecord(
                    memory_id="memory-0001",
                    run_id="run-calibration-win",
                    work_item_id="workitem-0001",
                    experiment_id="exp-0001",
                    family="perch_cached_probe",
                    idea_class="prior_calibration",
                    branch_role="hedge",
                    outcome="improved",
                    summary="A prior-calibration hedge unexpectedly improved holdout ROC-AUC.",
                    signal_score=2.0,
                    created_at="2026-04-01T00:00:00Z",
                )
            )

            bundle = retrieve_knowledge_bundle(
                config,
                {
                    "run": {"run_id": "run-knowledge-memory", "primary_metric_name": "val_soundscape_macro_roc_auc"},
                    "experiment": {"family": "perch_cached_probe", "title": "Probe baseline"},
                    "report": {"root_cause": "class imbalance on holdout", "focus": "calibration"},
                },
                stage="plan",
                state=state,
            )

            self.assertTrue(bundle["policy_cards"])
            self.assertTrue(bundle["branch_memories"])
            contradiction_types = {item["type"] for item in bundle["contradictions"]}
            self.assertIn("negative-policy-overridden-by-result", contradiction_types)

    def test_build_plan_materializes_multi_branch_configs_from_fallback_search(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
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

    def test_build_plan_persists_pruned_branches_and_policy_trace(self) -> None:
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
                id="exp-plan-policy-prune",
                title="Perch cached-probe baseline",
                hypothesis="baseline",
                family=work_item.family,
                config_path=work_item.config_path,
                priority=work_item.priority,
                work_item_id=work_item.id,
                spec_id="spec-seed",
            )
            state.experiments.append(experiment)
            state.branch_memories.append(
                BranchMemoryRecord(
                    memory_id="memory-0002",
                    run_id="run-old-calibration-regression",
                    work_item_id="workitem-old",
                    experiment_id="exp-old",
                    family="perch_cached_probe",
                    idea_class="prior_calibration",
                    branch_role="hedge",
                    outcome="regressed",
                    summary="Calibration-only branch regressed validation.",
                    signal_score=-2.0,
                    created_at="2026-04-01T00:10:00Z",
                )
            )
            run = RunRecord(
                run_id="run-plan-policy-prune",
                experiment_id=experiment.id,
                work_item_id=work_item.id,
                spec_id="spec-seed",
                status="succeeded",
                command="",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-plan-policy-prune"),
                log_path=str(root / "artifacts" / "runs" / "run-plan-policy-prune" / "train.log"),
                primary_metric_name="val_soundscape_macro_roc_auc",
                primary_metric_value=0.64,
            )
            state.runs.append(run)

            def _latest_payload(_state, _run_id, stage_name):
                if stage_name == "decision":
                    return {
                        "stage": "decision",
                        "next_action": "run_new_experiment",
                        "why": "Keep a compact frontier portfolio.",
                        "next_title": "Perch cached-probe baseline follow-up",
                        "next_family": "perch_cached_probe",
                        "next_config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
                        "priority_delta": 10,
                        "launch_mode": "background",
                        "portfolio_policy": {
                            "target_branch_count": 2,
                            "per_portfolio_cap": 1,
                            "per_idea_class_cap": 1,
                            "dispatch_strategy": "prefer-diverse-frontier-before-follow-on-support",
                            "deprioritized_axes": ["prior_calibration"],
                        },
                        "branch_portfolio": [
                            {
                                "title": "Coverage branch",
                                "family": "perch_cached_probe",
                                "hypothesis": "Fix coverage first.",
                                "rationale": "Coverage remains the main bottleneck.",
                                "branch_role": "primary",
                                "idea_class": "class_coverage",
                                "target_component": "class_coverage",
                                "priority_delta": 10,
                                "launch_mode": "background",
                                "knowledge_card_ids": ["card-coverage"],
                            },
                            {
                                "title": "Calibration sweep",
                                "family": "perch_cached_probe",
                                "hypothesis": "Only retune calibration.",
                                "rationale": "Cheap hedge branch.",
                                "branch_role": "hedge",
                                "idea_class": "prior_calibration",
                                "target_component": "prior_calibration",
                                "priority_delta": 12,
                                "launch_mode": "background",
                                "knowledge_card_ids": ["card-calibration"],
                            },
                            {
                                "title": "Probe branch",
                                "family": "perch_cached_probe",
                                "hypothesis": "Adjust probe capacity.",
                                "rationale": "Representation hedge.",
                                "branch_role": "explore",
                                "idea_class": "probe_head",
                                "target_component": "probe_head",
                                "priority_delta": 14,
                                "launch_mode": "background",
                                "knowledge_card_ids": ["card-probe"],
                            },
                        ],
                    }
                if stage_name == "research":
                    return {
                        "stage": "research",
                        "root_cause": "class imbalance on holdout",
                        "adopt_now": ["coverage-first branch"],
                        "consider": ["probe-head capacity branch"],
                        "reject": ["calibration-only sweep"],
                        "knowledge_card_ids": ["card-coverage", "card-calibration", "card-probe"],
                        "policy_rules": [
                            {
                                "rule_id": "rule-coverage",
                                "component": "class_coverage",
                                "policy_type": "require",
                                "confidence": 0.91,
                            },
                            {
                                "rule_id": "rule-calibration",
                                "component": "prior_calibration",
                                "policy_type": "veto",
                                "confidence": 0.88,
                            },
                        ],
                        "branch_memories": [
                            {
                                "memory_id": "memory-0002",
                                "idea_class": "prior_calibration",
                                "outcome": "regressed",
                                "signal_score": -2.0,
                            }
                        ],
                        "branch_memory_ids": ["memory-0002"],
                    }
                return {}

            with patch("kaggle_agent.decision.planner.latest_stage_payload", side_effect=_latest_payload), patch(
                "kaggle_agent.decision.planner.run_configured_stage_adapter",
                return_value=None,
            ):
                stage_run = build_plan(config, state, run.run_id)

            payload = json.loads(Path(stage_run.output_json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["plan_status"], "planned")
            self.assertEqual(len(payload["branch_plans"]), 2)
            self.assertTrue(payload["pruned_branches"])
            pruned_titles = {item["title"] for item in payload["pruned_branches"]}
            self.assertIn("Calibration sweep", pruned_titles)
            self.assertTrue(payload["policy_trace"])
            self.assertTrue(payload["scheduler_hints"])
            self.assertTrue(all(branch["policy_trace"] for branch in payload["branch_plans"]))
            self.assertTrue(all(isinstance(branch["scheduler_hints"], dict) for branch in payload["branch_plans"]))
            markdown = Path(stage_run.output_md_path).read_text(encoding="utf-8")
            self.assertIn("Pruned Branches", markdown)
            self.assertIn("Policy Trace", markdown)

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
            self.assertTrue(all(item.lifecycle_template == "branch_experiment" for item in new_items))
            self.assertTrue(
                all(
                    item.pipeline
                    == ["codegen", "critic", "validate", "execute", "evidence", "report", "research", "decision", "plan", "submission"]
                    for item in new_items
                )
            )

    def test_validate_stage_dispatches_deferred_branches_for_submission_candidate(self) -> None:
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
                id="exp-validate-submission-deferred",
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
                run_id="run-validate-submission-deferred",
                experiment_id=experiment.id,
                work_item_id=work_item.id,
                spec_id="spec-seed",
                status="succeeded",
                command="",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-validate-submission-deferred"),
                log_path=str(root / "artifacts" / "runs" / "run-validate-submission-deferred" / "train.log"),
                primary_metric_name="val_soundscape_macro_roc_auc",
                primary_metric_value=0.665,
            )
            state.runs.append(run)
            generated_root = root / "BirdCLEF-2026-Codebase" / "configs" / "generated"
            generated_root.mkdir(parents=True, exist_ok=True)
            branch_cfg = generated_root / "post-submit-branch.yaml"
            branch_cfg.write_text((root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8")

            def _latest_payload(_state, _run_id, stage_name):
                if stage_name == "plan":
                    return {
                        "stage": "plan",
                        "plan_status": "submission_candidate",
                        "source_run_id": run.run_id,
                        "reason": "Ship the baseline, then fan out post-submission improvements.",
                        "title": "Build submission bundle",
                        "family": "perch_cached_probe",
                        "hypothesis": "Submit now, then expand class coverage.",
                        "config_path": str((root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml").relative_to(root)),
                        "priority": 20,
                        "depends_on": [work_item.id],
                        "tags": ["submission_candidate"],
                        "work_type": "submission",
                        "lifecycle_template": "submission_from_target_run",
                        "target_run_id": run.run_id,
                        "portfolio_id": "portfolio-run-validate-submission-deferred",
                        "branch_plans": [
                            {
                                "title": "Expand class coverage",
                                "family": "perch_cached_probe",
                                "hypothesis": "Coverage expansion should improve macro ROC-AUC after leaderboard anchoring.",
                                "reason": "Deferred improvement branch after submission.",
                                "config_path": str(branch_cfg.relative_to(root)),
                                "priority": 28,
                                "depends_on": [work_item.id],
                                "tags": ["planned", "branch-search", "improvement"],
                                "launch_mode": "background",
                                "dedupe_key": "plan:post-submit-branch",
                                "work_type": "experiment_iteration",
                                "portfolio_id": "portfolio-run-validate-submission-deferred",
                                "idea_class": "class_coverage",
                                "branch_role": "improvement",
                                "branch_rank": 0,
                                "knowledge_card_ids": ["card-post-submit"],
                            }
                        ],
                    }
                if stage_name == "codegen":
                    return {
                        "stage": "codegen",
                        "status": "noop",
                        "verify_status": "skipped",
                        "verify_summary": "Codegen provider was not invoked because the plan is `submission_candidate`.",
                    }
                if stage_name == "critic":
                    return {
                        "stage": "critic",
                        "status": "approved",
                        "warnings": ["Critic provider was not invoked because the plan is `submission_candidate`."],
                    }
                return {}

            with patch("kaggle_agent.control.monitor.latest_stage_payload", side_effect=_latest_payload):
                stage_run = _run_validate_stage(config, state, run.run_id)

            payload = json.loads(Path(stage_run.output_json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "validated")
            self.assertEqual(payload["plan_status"], "submission_candidate")
            self.assertEqual(payload["branch_count"], 1)
            self.assertIn("proceeding to submission", payload["summary"])
            derived = next(item for item in state.work_items if item.id != "workitem-perch-baseline")
            self.assertEqual(derived.branch_role, "improvement")
            self.assertEqual(derived.idea_class, "class_coverage")
            self.assertEqual(derived.lifecycle_template, "branch_experiment")

    def test_validate_stage_assigns_submission_lifecycle_and_target_run(self) -> None:
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
                id="exp-validate-submission",
                title="Perch cached-probe baseline",
                hypothesis="baseline",
                family=work_item.family,
                config_path=work_item.config_path,
                priority=work_item.priority,
                work_item_id=work_item.id,
                spec_id="spec-seed",
            )
            state.experiments.append(experiment)
            parent_run = RunRecord(
                run_id="run-validate-submission",
                experiment_id=experiment.id,
                work_item_id=work_item.id,
                spec_id="spec-seed",
                status="succeeded",
                command="",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-validate-submission"),
                log_path=str(root / "artifacts" / "runs" / "run-validate-submission" / "train.log"),
                primary_metric_name="val_soundscape_macro_roc_auc",
                primary_metric_value=0.68,
            )
            target_run = RunRecord(
                run_id="run-leader-target",
                experiment_id=experiment.id,
                work_item_id=work_item.id,
                spec_id="spec-seed",
                status="succeeded",
                command="",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-leader-target"),
                log_path=str(root / "artifacts" / "runs" / "run-leader-target" / "train.log"),
                primary_metric_name="val_soundscape_macro_roc_auc",
                primary_metric_value=0.72,
            )
            state.runs.extend([parent_run, target_run])
            code_state_root = root / "state" / "snapshots" / "codegen" / "code-state"
            code_state_root.mkdir(parents=True, exist_ok=True)
            generated_root = root / "BirdCLEF-2026-Codebase" / "configs" / "generated"
            generated_root.mkdir(parents=True, exist_ok=True)
            branch_cfg = generated_root / "submission-branch.yaml"
            branch_cfg.write_text((root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8")

            def _latest_payload(_state, _run_id, stage_name):
                if stage_name == "plan":
                    return {
                        "stage": "plan",
                        "plan_status": "planned",
                        "source_run_id": parent_run.run_id,
                        "reason": "Package the current leader directly.",
                        "title": "Submission branch",
                        "family": "perch_cached_probe",
                        "hypothesis": "Submit the best validated run without retraining.",
                        "config_path": str(branch_cfg.relative_to(root)),
                        "priority": 20,
                        "depends_on": [work_item.id],
                        "tags": ["planned", "submission"],
                        "launch_mode": "background",
                        "dedupe_key": "plan:submission-branch",
                        "work_type": "submission",
                        "portfolio_id": "portfolio-run-validate-submission",
                        "knowledge_card_ids": ["card-submit"],
                        "branch_plans": [
                            {
                                "title": "Submission branch",
                                "family": "perch_cached_probe",
                                "hypothesis": "Submit the best validated run without retraining.",
                                "reason": "Use the validated leader as-is.",
                                "config_path": str(branch_cfg.relative_to(root)),
                                "priority": 20,
                                "depends_on": [work_item.id],
                                "tags": ["planned", "submission"],
                                "launch_mode": "background",
                                "dedupe_key": "plan:submission-branch",
                                "work_type": "submission",
                                "target_run_id": target_run.run_id,
                                "portfolio_id": "portfolio-run-validate-submission",
                                "idea_class": "submission",
                                "branch_role": "submission",
                                "branch_rank": 0,
                                "knowledge_card_ids": ["card-submit"],
                            }
                        ],
                    }
                if stage_name == "codegen":
                    return {
                        "stage": "codegen",
                        "status": "generated",
                        "reason": "Generated submission config pointer.",
                        "generated_config_path": str(branch_cfg),
                        "run_bundle_path": "",
                        "patch_path": "",
                        "code_state_ref": str(code_state_root),
                        "verify_artifacts_ref": str(root / "state" / "verify"),
                        "verify_command": "python train_sed.py --config generated/submission-branch.yaml",
                        "verify_status": "passed",
                        "verify_summary": "verify passed",
                        "worktree_path": str(code_state_root),
                        "base_commit": "abc123",
                        "head_commit": "def456",
                        "changed_files": ["BirdCLEF-2026-Codebase/configs/generated/submission-branch.yaml"],
                        "provider_runtime": "claude_code mode:agentic",
                        "allowed_edit_roots": ["BirdCLEF-2026-Codebase/configs/generated"],
                        "smoke_status": "passed",
                        "smoke_summary": "verify passed",
                    }
                if stage_name == "critic":
                    return {"stage": "critic", "status": "approved"}
                return {}

            with patch("kaggle_agent.control.monitor.latest_stage_payload", side_effect=_latest_payload):
                stage_run = _run_validate_stage(config, state, parent_run.run_id)

            payload = json.loads(Path(stage_run.output_json_path).read_text(encoding="utf-8"))
            dispatched = payload["dispatch_summary"][0]
            derived = next(item for item in state.work_items if item.id != "workitem-perch-baseline")
            self.assertEqual(derived.lifecycle_template, "submission_from_target_run")
            self.assertEqual(derived.pipeline, ["submission"])
            self.assertEqual(derived.target_run_id, target_run.run_id)
            self.assertEqual(dispatched["lifecycle_template"], "submission_from_target_run")
            self.assertEqual(dispatched["stage_plan"], ["submission"])
            self.assertEqual(dispatched["target_run_id"], target_run.run_id)

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

    def test_choose_next_work_items_prefers_portfolio_diversity(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            config_path = str((root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml").relative_to(root))
            state.work_items = [
                WorkItem(
                    id="workitem-a",
                    title="Coverage frontier",
                    work_type="experiment_iteration",
                    family="perch_cached_probe",
                    priority=20,
                    status="queued",
                    config_path=config_path,
                    pipeline=list(state.work_items[0].pipeline),
                    portfolio_id="portfolio-a",
                    idea_class="class_coverage",
                    branch_role="primary",
                    scheduler_hints={"portfolio_cap": 1, "idea_class_cap": 1, "dispatch_priority": 8.0},
                ),
                WorkItem(
                    id="workitem-b",
                    title="Probe hedge same portfolio",
                    work_type="experiment_iteration",
                    family="perch_cached_probe",
                    priority=21,
                    status="queued",
                    config_path=config_path,
                    pipeline=list(state.work_items[0].pipeline),
                    portfolio_id="portfolio-a",
                    idea_class="probe_head",
                    branch_role="hedge",
                    scheduler_hints={"portfolio_cap": 1, "idea_class_cap": 1, "dispatch_priority": 7.0},
                ),
                WorkItem(
                    id="workitem-c",
                    title="Pseudo-label frontier",
                    work_type="experiment_iteration",
                    family="perch_cached_probe",
                    priority=22,
                    status="queued",
                    config_path=config_path,
                    pipeline=list(state.work_items[0].pipeline),
                    portfolio_id="portfolio-b",
                    idea_class="pseudo_label",
                    branch_role="primary",
                    scheduler_hints={"portfolio_cap": 1, "idea_class_cap": 1, "dispatch_priority": 6.0},
                ),
            ]
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

            self.assertEqual([item.id for item in selected], ["workitem-a", "workitem-c"])

    def test_synchronize_branch_memory_records_outcome_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            work_item = next(item for item in state.work_items if item.id == "workitem-perch-baseline")
            work_item.branch_role = "primary"
            work_item.idea_class = "class_coverage"
            experiment = ExperimentSpec(
                id="exp-memory-sync",
                title="Perch cached-probe baseline",
                hypothesis="baseline",
                family=work_item.family,
                config_path=work_item.config_path,
                priority=work_item.priority,
                work_item_id=work_item.id,
                spec_id="spec-seed",
            )
            state.experiments.append(experiment)
            state.runs.append(
                RunRecord(
                    run_id="run-memory-parent",
                    experiment_id=experiment.id,
                    work_item_id=work_item.id,
                    spec_id="spec-seed",
                    status="succeeded",
                    command="",
                    cwd=str(root),
                    run_dir=str(root / "artifacts" / "runs" / "run-memory-parent"),
                    log_path=str(root / "artifacts" / "runs" / "run-memory-parent" / "train.log"),
                    primary_metric_name="val_soundscape_macro_roc_auc",
                    primary_metric_value=0.61,
                )
            )
            work_item.source_run_id = "run-memory-parent"
            run = RunRecord(
                run_id="run-memory-sync",
                experiment_id=experiment.id,
                work_item_id=work_item.id,
                spec_id="spec-seed",
                status="succeeded",
                command="",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-memory-sync"),
                log_path=str(root / "artifacts" / "runs" / "run-memory-sync" / "train.log"),
                primary_metric_name="val_soundscape_macro_roc_auc",
                primary_metric_value=0.64,
            )
            state.runs.append(run)
            critic_dir = root / "artifacts" / "critic"
            critic_dir.mkdir(parents=True)
            (critic_dir / "critic.json").write_text(json.dumps({"stage": "critic", "status": "approved"}), encoding="utf-8")
            validate_dir = root / "artifacts" / "validate"
            validate_dir.mkdir(parents=True)
            (validate_dir / "validate.json").write_text(json.dumps({"stage": "validate", "status": "validated"}), encoding="utf-8")
            state.stage_runs.extend(
                [
                    StageRun(
                        stage_run_id="stage-critic-sync",
                        run_id=run.run_id,
                        work_item_id=work_item.id,
                        stage_name="critic",
                        status="completed",
                        input_ref=run.run_id,
                        output_dir=str(critic_dir),
                        output_json_path=str(critic_dir / "critic.json"),
                        output_md_path=str(critic_dir / "critic.md"),
                    ),
                    StageRun(
                        stage_run_id="stage-validate-sync",
                        run_id=run.run_id,
                        work_item_id=work_item.id,
                        stage_name="validate",
                        status="completed",
                        input_ref=run.run_id,
                        output_dir=str(validate_dir),
                        output_json_path=str(validate_dir / "validate.json"),
                        output_md_path=str(validate_dir / "validate.md"),
                    ),
                ]
            )
            run.latest_stage_run_id = "stage-validate-sync"

            memory = synchronize_branch_memory(state, run.run_id)

            self.assertIsNotNone(memory)
            assert memory is not None
            self.assertEqual(memory.outcome, "leader")
            self.assertIn("delta=+0.030000", memory.summary)
            self.assertEqual(state.branch_memories[-1].run_id, run.run_id)

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

    def test_enqueue_config_allows_terminal_milestone_lifecycle_and_notes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = enqueue_config(
                config,
                "BirdCLEF-2026-Codebase/configs/debug.yaml",
                title="Milestone v5 seed",
                family="sed_v5_like",
                priority=7,
                work_type="ablation_terminal",
                notes=[
                    "Target the v5-like SED baseline directly.",
                    "Do not expand into unrelated Perch full-scale planning.",
                ],
            )

            work_item = next(item for item in state.work_items if item.title == "Milestone v5 seed")
            self.assertEqual(work_item.work_type, "ablation_terminal")
            self.assertEqual(work_item.lifecycle_template, "terminal_experiment")
            self.assertEqual(work_item.pipeline, ["execute", "evidence", "report", "validate"])
            self.assertIn("Target the v5-like SED baseline directly.", work_item.notes)

            frame = build_problem_frame({"work_item": work_item.to_dict(), "experiment": {"family": work_item.family}}, stage="plan")
            self.assertEqual(frame["lifecycle_template"], "terminal_experiment")
            self.assertIn("v5-like", " ".join(frame["notes"]))

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
            self.assertEqual(agent_by_role["plan"].provider, "claude_code")
            self.assertTrue(agent_by_role["plan"].thread_id)
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
            self.assertTrue(codegen_payload["provider_runtime"].startswith("claude_code mode:agentic"))
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

            follow_up_item = derived_items[0]
            follow_up_run = start_run(config, state, follow_up_item.id, background=False)
            self.assertEqual(follow_up_run.command, "synthetic:codegen")
            follow_up_experiment = next(item for item in state.experiments if item.id == follow_up_run.experiment_id)
            follow_up_spec = next(item for item in state.specs if item.spec_id == follow_up_run.spec_id)
            launch_execute_stage(
                config,
                state,
                follow_up_item,
                follow_up_spec,
                follow_up_experiment,
                follow_up_run,
                background=False,
            )
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

    def test_process_completed_runs_finalizes_stage_error_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)

            work_item = WorkItem(
                id="workitem-stage-error",
                title="Stage error terminal run",
                work_type="ablation_terminal",
                family="perch_cached_probe",
                priority=40,
                config_path="BirdCLEF-2026-Codebase/configs/default.yaml",
                lifecycle_template="terminal_experiment",
                pipeline=["execute", "evidence", "report", "validate"],
                status="reviewing",
            )
            run = RunRecord(
                run_id="run-stage-error",
                experiment_id="exp-stage-error",
                work_item_id=work_item.id,
                spec_id="spec-stage-error",
                status="succeeded",
                command="python train_sed.py",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-stage-error"),
                log_path=str(root / "artifacts" / "runs" / "run-stage-error" / "train.log"),
                stage_cursor="report",
                stage_error="report adapter failed: no output",
                lifecycle_template="terminal_experiment",
                stage_plan=["execute", "evidence", "report", "validate"],
            )
            state = WorkspaceState(
                work_items=[work_item],
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

            process_completed_runs(config, state)

            self.assertEqual(work_item.status, "failed")
            self.assertEqual(run.stage_cursor, "complete")
            self.assertEqual(run.stage_error, "report adapter failed: no output")

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

    def test_process_run_stage_chain_stops_terminal_experiment_after_validate(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)

            work_item = WorkItem(
                id="workitem-terminal",
                title="Terminal ablation",
                work_type="ablation_terminal",
                family="perch_cached_probe",
                priority=40,
                config_path="BirdCLEF-2026-Codebase/configs/default.yaml",
                lifecycle_template="terminal_experiment",
                pipeline=["execute", "evidence", "report", "validate"],
                status="reviewing",
            )
            run = RunRecord(
                run_id="run-terminal",
                experiment_id="exp-terminal",
                work_item_id=work_item.id,
                spec_id="spec-terminal",
                status="succeeded",
                command="python train_sed.py",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-terminal"),
                log_path=str(root / "artifacts" / "runs" / "run-terminal" / "train.log"),
                stage_cursor="evidence",
                lifecycle_template="terminal_experiment",
                stage_plan=["execute", "evidence", "report", "validate"],
            )
            state = WorkspaceState(
                work_items=[work_item],
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

            def _payload_for_stage(_state: WorkspaceState, _run_id: str, stage_name: str) -> dict[str, object]:
                if stage_name == "validate":
                    return {"stage": "validate", "status": "validated"}
                return {}

            with patch("kaggle_agent.control.monitor.build_evidence", return_value=None) as evidence_mock, patch(
                "kaggle_agent.control.monitor.build_report",
                return_value=None,
            ) as report_mock, patch(
                "kaggle_agent.control.monitor._run_validate_stage",
                return_value=None,
            ) as validate_mock, patch(
                "kaggle_agent.control.monitor.build_research",
            ) as research_mock, patch(
                "kaggle_agent.control.monitor.build_decision",
            ) as decision_mock, patch(
                "kaggle_agent.control.monitor.build_plan",
            ) as plan_mock, patch(
                "kaggle_agent.control.monitor.build_codegen",
            ) as codegen_mock, patch(
                "kaggle_agent.control.monitor.build_critic",
            ) as critic_mock, patch(
                "kaggle_agent.control.monitor._run_submission_stage",
            ) as submission_mock, patch(
                "kaggle_agent.control.monitor.latest_stage_payload",
                side_effect=_payload_for_stage,
            ):
                _process_run_stage_chain(config, state, run.run_id)

            evidence_mock.assert_called_once()
            report_mock.assert_called_once()
            validate_mock.assert_called_once()
            research_mock.assert_not_called()
            decision_mock.assert_not_called()
            plan_mock.assert_not_called()
            codegen_mock.assert_not_called()
            critic_mock.assert_not_called()
            submission_mock.assert_not_called()
            self.assertEqual(run.stage_cursor, "complete")
            self.assertEqual(work_item.status, "complete")

    def test_submission_lifecycle_creates_synthetic_run_and_packages_target_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            baseline = next(item for item in state.work_items if item.id == "workitem-perch-baseline")

            target_experiment = ExperimentSpec(
                id="exp-target-run",
                title="Validated leader",
                hypothesis="leader",
                family=baseline.family,
                config_path=baseline.config_path,
                priority=baseline.priority,
                work_item_id=baseline.id,
                spec_id="spec-target-run",
            )
            state.experiments.append(target_experiment)
            target_run = RunRecord(
                run_id="run-target-run",
                experiment_id=target_experiment.id,
                work_item_id=baseline.id,
                spec_id="spec-target-run",
                status="succeeded",
                command="python train_sed.py --config default.yaml",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-target-run"),
                log_path=str(root / "artifacts" / "runs" / "run-target-run" / "train.log"),
                primary_metric_name="val_soundscape_macro_roc_auc",
                primary_metric_value=0.74,
                stage_cursor="complete",
            )
            state.runs.append(target_run)

            submission_item = WorkItem(
                id="workitem-submission-target",
                title="Package target run",
                work_type="submission",
                family=baseline.family,
                priority=10,
                config_path=baseline.config_path,
                lifecycle_template="submission_from_target_run",
                target_run_id=target_run.run_id,
                pipeline=["submission"],
                status="queued",
                dedupe_key="manual:submission-target",
            )
            state.work_items.append(submission_item)

            synthetic_run = start_run(config, state, submission_item.id, background=False)
            self.assertEqual(synthetic_run.status, "succeeded")
            self.assertEqual(synthetic_run.command, "synthetic:submission")
            self.assertEqual(synthetic_run.stage_cursor, "submission")
            self.assertFalse((Path(synthetic_run.run_dir) / "launch.sh").exists())

            process_completed_runs(config, state)

            submission_stage = next(item for item in state.stage_runs if item.run_id == synthetic_run.run_id and item.stage_name == "submission")
            submission_payload = json.loads(Path(submission_stage.output_json_path).read_text(encoding="utf-8"))
            self.assertEqual(submission_payload["status"], "candidate_created")
            self.assertEqual(submission_payload["target_run_id"], target_run.run_id)
            self.assertEqual(state.submissions[-1].source_run_id, target_run.run_id)
            self.assertEqual(submission_item.status, "submitted")
            self.assertEqual(synthetic_run.stage_cursor, "complete")

    def test_start_run_for_branch_experiment_enters_codegen_first(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            baseline = next(item for item in state.work_items if item.id == "workitem-perch-baseline")
            branch_item = WorkItem(
                id="workitem-branch-entry",
                title="Coverage branch",
                work_type="experiment_iteration",
                family=baseline.family,
                priority=25,
                config_path=baseline.config_path,
                lifecycle_template="branch_experiment",
                pipeline=["codegen", "critic", "validate", "execute", "evidence", "report", "research", "decision", "plan", "submission"],
                status="queued",
            )
            state.work_items.append(branch_item)

            run = start_run(config, state, branch_item.id, background=False)

            self.assertEqual(run.status, "succeeded")
            self.assertEqual(run.command, "synthetic:codegen")
            self.assertEqual(run.stage_cursor, "codegen")
            self.assertEqual(branch_item.status, "running")

    def test_process_run_stage_chain_branch_experiment_launches_execute_after_validate(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            baseline = next(item for item in state.work_items if item.id == "workitem-perch-baseline")
            branch_item = WorkItem(
                id="workitem-branch-exec",
                title="Branch execute transition",
                work_type="experiment_iteration",
                family=baseline.family,
                priority=25,
                config_path=baseline.config_path,
                lifecycle_template="branch_experiment",
                pipeline=["codegen", "critic", "validate", "execute", "evidence", "report", "research", "decision", "plan", "submission"],
                status="running",
            )
            state.work_items.append(branch_item)
            spec = ExperimentSpec(
                id="exp-branch-exec",
                title=branch_item.title,
                hypothesis="branch",
                family=branch_item.family,
                config_path=branch_item.config_path,
                priority=branch_item.priority,
                work_item_id=branch_item.id,
                spec_id="spec-branch-exec",
            )
            state.experiments.append(spec)
            state.specs.append(
                SpecRecord(
                    spec_id="spec-branch-exec",
                    work_item_id=branch_item.id,
                    source_stage_run_id="stage-upstream",
                    spec_type="experiment",
                    title=branch_item.title,
                    family=branch_item.family,
                    config_path=branch_item.config_path,
                    payload_path=branch_item.config_path,
                    launch_mode="background",
                    code_state_ref="",
                    status="validated",
                    dedupe_key="spec:branch-exec",
                )
            )
            run = RunRecord(
                run_id="run-branch-exec",
                experiment_id=spec.id,
                work_item_id=branch_item.id,
                spec_id="spec-branch-exec",
                status="succeeded",
                command="synthetic:codegen",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-branch-exec"),
                log_path=str(root / "artifacts" / "runs" / "run-branch-exec" / "train.log"),
                stage_cursor="codegen",
                lifecycle_template="branch_experiment",
                stage_plan=["codegen", "critic", "validate", "execute", "evidence", "report", "research", "decision", "plan", "submission"],
            )
            state.runs.append(run)

            def _launch(*_args, **_kwargs):
                run.status = "running"
                run.stage_cursor = ""
                run.stage_updated_at = "2026-04-01T00:00:00Z"
                return run

            with patch("kaggle_agent.control.monitor.build_codegen", return_value=None) as codegen_mock, patch(
                "kaggle_agent.control.monitor.build_critic",
                return_value=None,
            ) as critic_mock, patch(
                "kaggle_agent.control.monitor._run_validate_stage",
                return_value=None,
            ) as validate_mock, patch(
                "kaggle_agent.control.monitor.launch_execute_stage",
                side_effect=_launch,
            ) as launch_mock, patch(
                "kaggle_agent.control.monitor.build_evidence",
            ) as evidence_mock:
                _process_run_stage_chain(config, state, run.run_id, execute_in_background=True)

            codegen_mock.assert_called_once()
            critic_mock.assert_called_once()
            validate_mock.assert_called_once()
            launch_mock.assert_called_once()
            evidence_mock.assert_not_called()
            self.assertEqual(run.status, "running")
            self.assertEqual(run.stage_cursor, "")

    def test_save_state_uses_validation_id_for_validation_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            state.validations = [
                ValidationRecord(
                    validation_id="validation-0001",
                    work_item_id="workitem-a",
                    source_stage_run_id="stage-a",
                    spec_id="spec-shared",
                    status="validated",
                    summary="first validation",
                    output_json_path="/tmp/validate-a.json",
                    output_md_path="/tmp/validate-a.md",
                    created_at="2026-03-31T00:00:00Z",
                ),
                ValidationRecord(
                    validation_id="validation-0002",
                    work_item_id="workitem-b",
                    source_stage_run_id="stage-b",
                    spec_id="spec-shared",
                    status="validated",
                    summary="second validation",
                    output_json_path="/tmp/validate-b.json",
                    output_md_path="/tmp/validate-b.md",
                    created_at="2026-03-31T00:01:00Z",
                ),
            ]

            save_state(config, state)
            reloaded = load_state(config)

            self.assertEqual([item.validation_id for item in reloaded.validations], ["validation-0001", "validation-0002"])
            self.assertEqual([item.spec_id for item in reloaded.validations], ["spec-shared", "spec-shared"])

    def test_write_reports_emits_branch_lifecycle_order_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            branch_item = WorkItem(
                id="workitem-branch-order",
                title="Branch order visibility",
                work_type="experiment_iteration",
                family="perch_cached_probe",
                priority=25,
                config_path="BirdCLEF-2026-Codebase/configs/generated/branch-order.yaml",
                lifecycle_template="branch_experiment",
                pipeline=["codegen", "critic", "validate", "execute", "evidence", "report", "research", "decision", "plan", "submission"],
                status="running",
            )
            state.work_items.append(branch_item)
            run = RunRecord(
                run_id="run-branch-order",
                experiment_id="exp-branch-order",
                work_item_id=branch_item.id,
                spec_id="spec-branch-order",
                status="running",
                command="python train_sed.py",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "attempts" / "demo" / "runs" / "run-branch-order" / "runtime"),
                log_path=str(root / "artifacts" / "attempts" / "demo" / "runs" / "run-branch-order" / "runtime" / "train.log"),
                started_at="2026-04-01T00:00:00Z",
                lifecycle_template="branch_experiment",
                stage_plan=["codegen", "critic", "validate", "execute", "evidence", "report", "research", "decision", "plan", "submission"],
                stage_cursor="report",
                latest_stage_run_id="stage-report",
            )
            state.runs.append(run)
            state.stage_runs.extend(
                [
                    StageRun(
                        stage_run_id="stage-codegen",
                        run_id=run.run_id,
                        work_item_id=branch_item.id,
                        stage_name="codegen",
                        status="completed",
                        input_ref=run.run_id,
                        output_dir=str(Path(run.run_dir).parent / "stages" / "06-codegen__noop"),
                        output_json_path=str(Path(run.run_dir).parent / "stages" / "06-codegen__noop" / "codegen.json"),
                        output_md_path=str(Path(run.run_dir).parent / "stages" / "06-codegen__noop" / "codegen.md"),
                        validator_status="noop",
                        created_at="2026-04-01T00:01:00Z",
                        updated_at="2026-04-01T00:01:00Z",
                    ),
                    StageRun(
                        stage_run_id="stage-critic",
                        run_id=run.run_id,
                        work_item_id=branch_item.id,
                        stage_name="critic",
                        status="completed",
                        input_ref="stage-codegen",
                        output_dir=str(Path(run.run_dir).parent / "stages" / "07-critic__approved"),
                        output_json_path=str(Path(run.run_dir).parent / "stages" / "07-critic__approved" / "critic.json"),
                        output_md_path=str(Path(run.run_dir).parent / "stages" / "07-critic__approved" / "critic.md"),
                        validator_status="approved",
                        created_at="2026-04-01T00:02:00Z",
                        updated_at="2026-04-01T00:02:00Z",
                    ),
                    StageRun(
                        stage_run_id="stage-validate",
                        run_id=run.run_id,
                        work_item_id=branch_item.id,
                        stage_name="validate",
                        status="completed",
                        input_ref="stage-critic",
                        output_dir=str(Path(run.run_dir).parent / "stages" / "08-validate__not-required"),
                        output_json_path=str(Path(run.run_dir).parent / "stages" / "08-validate__not-required" / "validate.json"),
                        output_md_path=str(Path(run.run_dir).parent / "stages" / "08-validate__not-required" / "validate.md"),
                        validator_status="not-required",
                        created_at="2026-04-01T00:03:00Z",
                        updated_at="2026-04-01T00:03:00Z",
                    ),
                    StageRun(
                        stage_run_id="stage-evidence",
                        run_id=run.run_id,
                        work_item_id=branch_item.id,
                        stage_name="evidence",
                        status="completed",
                        input_ref="run-branch-order",
                        output_dir=str(Path(run.run_dir).parent / "stages" / "01-evidence__succeeded"),
                        output_json_path=str(Path(run.run_dir).parent / "stages" / "01-evidence__succeeded" / "evidence.json"),
                        output_md_path=str(Path(run.run_dir).parent / "stages" / "01-evidence__succeeded" / "evidence.md"),
                        validator_status="succeeded",
                        created_at="2026-04-01T00:04:00Z",
                        updated_at="2026-04-01T00:04:00Z",
                    ),
                    StageRun(
                        stage_run_id="stage-report",
                        run_id=run.run_id,
                        work_item_id=branch_item.id,
                        stage_name="report",
                        status="running",
                        input_ref="stage-evidence",
                        output_dir=str(Path(run.run_dir).parent / "stages" / "02-report__running"),
                        output_json_path=str(Path(run.run_dir).parent / "stages" / "02-report__running" / "report.json"),
                        output_md_path=str(Path(run.run_dir).parent / "stages" / "02-report__running" / "report.md"),
                        created_at="2026-04-01T00:05:00Z",
                        updated_at="2026-04-01T00:05:00Z",
                    ),
                ]
            )

            write_reports(config, state)

            order_md = Path(run.run_dir).parent / "stages" / "ORDER.md"
            lifecycle_json = Path(run.run_dir).parent / "lifecycle.json"
            self.assertTrue(order_md.exists())
            self.assertTrue(lifecycle_json.exists())
            order_text = order_md.read_text(encoding="utf-8")
            self.assertIn("Stage directory prefixes keep canonical stage ids", order_text)
            self.assertIn("1. `codegen` | status=`noop` | canonical_dir=`06-codegen__*` | artifact=`06-codegen__noop`", order_text)
            self.assertIn("5. `evidence` | status=`succeeded` | canonical_dir=`01-evidence__*` | artifact=`01-evidence__succeeded`", order_text)
            self.assertIn("6. `report` | status=`running` | canonical_dir=`02-report__*` | artifact=`02-report__running`", order_text)

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

    def test_build_plan_keeps_followup_branch_portfolio_for_submission_candidate(self) -> None:
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
                id="exp-plan-submission-portfolio",
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
                run_id="run-plan-submission-portfolio",
                experiment_id=experiment.id,
                work_item_id=work_item.id,
                spec_id="spec-seed",
                status="succeeded",
                command="",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-plan-submission-portfolio"),
                log_path=str(root / "artifacts" / "runs" / "run-plan-submission-portfolio" / "train.log"),
                primary_metric_name="val_soundscape_macro_roc_auc",
                primary_metric_value=0.665,
            )
            state.runs.append(run)
            generated_root = root / "BirdCLEF-2026-Codebase" / "configs" / "generated"
            generated_root.mkdir(parents=True, exist_ok=True)
            branch_cfg = generated_root / "coverage-after-submit.yaml"
            branch_cfg.write_text((root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8")

            adapter_payload = {
                "stage": "plan",
                "plan_status": "submission_candidate",
                "title": "Build CPU-first submission bundle from run-0001",
                "family": "perch_cached_probe",
                "hypothesis": "Submit now, then continue coverage expansion in parallel.",
                "config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
                "priority": 20,
                "work_type": "submission",
                "target_run_id": run.run_id,
                "branch_plans": [
                    {
                        "title": "Submit baseline bundle",
                        "family": "perch_cached_probe",
                        "hypothesis": "Submit the current leader bundle.",
                        "reason": "Immediate submission route.",
                        "config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
                        "priority": 20,
                        "tags": ["submission"],
                        "work_type": "submission",
                        "branch_role": "submission",
                        "idea_class": "baseline_submission",
                        "target_run_id": run.run_id,
                    },
                    {
                        "title": "Expand class coverage after submission",
                        "family": "perch_cached_probe",
                        "hypothesis": "Coverage expansion should improve recall after the baseline is anchored.",
                        "reason": "Post-submission improvement branch.",
                        "config_path": str(branch_cfg.relative_to(root)),
                        "priority": 28,
                        "tags": ["improvement"],
                        "work_type": "experiment_iteration",
                        "branch_role": "improvement",
                        "idea_class": "class_coverage",
                        "knowledge_card_ids": ["card-coverage"],
                    },
                ],
            }

            def _latest_payload(_state, _run_id, stage_name):
                if stage_name == "decision":
                    return {
                        "stage": "decision",
                        "next_action": "submit_candidate",
                        "why": "Ship now, but keep the next improvement portfolio warm.",
                        "next_title": "Build CPU-first submission bundle from run-0001",
                        "next_family": "perch_cached_probe",
                    }
                if stage_name == "research":
                    return {
                        "stage": "research",
                        "root_cause": "Need leaderboard anchoring plus post-submission coverage work.",
                        "adopt_now": ["submit baseline bundle"],
                        "consider": ["coverage expansion after submission"],
                        "reject": [],
                    }
                return {}

            with patch("kaggle_agent.decision.planner.latest_stage_payload", side_effect=_latest_payload), patch(
                "kaggle_agent.decision.planner.run_configured_stage_adapter",
                return_value=(adapter_payload, "plan markdown"),
            ):
                stage_run = build_plan(config, state, run.run_id)

            payload = json.loads(Path(stage_run.output_json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["plan_status"], "submission_candidate")
            self.assertEqual(payload["lifecycle_template"], "submission_from_target_run")
            self.assertEqual(payload["stage_plan"], ["submission"])
            self.assertEqual(payload["target_run_id"], run.run_id)
            self.assertEqual(len(payload["branch_plans"]), 1)
            self.assertEqual(payload["branch_plans"][0]["branch_role"], "improvement")
            self.assertEqual(payload["branch_plans"][0]["idea_class"], "class_coverage")

    def test_build_codegen_skips_provider_for_submission_candidate_plan(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            config = load_config(root)

            output_dir = root / "artifacts" / "codegen-submission-output"
            output_dir.mkdir(parents=True)
            input_manifest_path = output_dir / "input_manifest.json"
            run = RunRecord(
                run_id="run-codegen-submission-skip",
                experiment_id="exp-codegen-submission-skip",
                work_item_id="workitem-codegen-submission-skip",
                spec_id="",
                status="succeeded",
                command="python train_sed.py",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-codegen-submission-skip"),
                log_path=str(root / "artifacts" / "runs" / "run-codegen-submission-skip" / "train.log"),
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
                stage_run_id="stage-codegen-submission-skip",
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
                "plan_status": "submission_candidate",
                "title": "Build submission bundle",
                "family": "perch_cached_probe",
                "config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
                "branch_role": "submission",
                "idea_class": "baseline_submission",
            }

            completed: dict[str, object] = {}

            with patch("kaggle_agent.decision.codegen.latest_stage_payload", side_effect=[plan_payload, {}]), patch(
                "kaggle_agent.decision.codegen.begin_stage_run",
                return_value=(stage_run, input_manifest_path),
            ), patch(
                "kaggle_agent.decision.codegen.run_configured_stage_adapter",
            ) as adapter_mock, patch(
                "kaggle_agent.decision.codegen.complete_stage_run",
                side_effect=lambda _stage_run, **kwargs: completed.update(kwargs),
            ):
                build_codegen(config, state, run.run_id)

            adapter_mock.assert_not_called()
            self.assertEqual(completed["payload"]["status"], "noop")
            self.assertEqual(completed["payload"]["reason"], "plan_status=submission_candidate")
            self.assertIn("was not invoked", completed["payload"]["verify_summary"])
            self.assertIn("Provider invocation: skipped", completed["markdown"])

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

    def test_build_prompt_persists_resolved_retrieved_knowledge_into_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_path = root / "schemas" / "decision.schema.json"
            schema_path.parent.mkdir(parents=True)
            schema_path.write_text("{}", encoding="utf-8")
            output_dir = root / "artifacts" / "decision-output"
            output_dir.mkdir(parents=True)
            input_manifest_path = output_dir / "input_manifest.json"
            ctx = StageContext(
                stage="decision",
                workspace_root=root,
                input_manifest_path=input_manifest_path,
                output_dir=output_dir,
                prompt_path=None,
                schema_path=schema_path,
                input_manifest={
                    "run": {"run_id": "run-decision-prompt"},
                    "research": {},
                    "report": {},
                },
            )
            bundle = {
                "problem_frame": {"run_id": "run-decision-prompt", "stage": "decision"},
                "knowledge_files_seen": 1,
                "knowledge_card_ids": ["card-1"],
                "cards": [{"card_id": "card-1", "summary": "Prior calibration remains conditional."}],
                "policy_rules": [{"rule_id": "policy-1", "component": "prior_calibration", "policy_type": "conditional"}],
                "claims": [{"claim_id": "claim-1", "summary": "Probe calibration is mixed."}],
                "branch_memories": [{"memory_id": "memory-1", "summary": "Previous probe branch was flat."}],
                "contradictions": [{"rule_id": "policy-1", "summary": "Mixed empirical evidence."}],
                "constraints": [{"constraint_id": "constraint-1", "summary": "Require probe training change."}],
                "semantic_memory_files": [],
                "capability_packs": [{"pack_id": "veto_checker"}],
                "capability_results": {"veto_checker": {"blocked_patterns": ["blend_only"]}},
                "session_memory": {"current_objective": "Keep branch search grounded."},
            }

            with patch("kaggle_agent.adapters.stage_wrapper.retrieve_knowledge_bundle_from_root", return_value=bundle), patch(
                "kaggle_agent.adapters.stage_wrapper.render_retrieved_knowledge",
                return_value="## Positive Priors\n- prior_calibration",
            ):
                prompt = _build_prompt(ctx)

            written_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
            self.assertIn("Knowledge Context", prompt)
            self.assertIn("## Positive Priors", prompt)
            self.assertEqual(written_manifest["retrieved_knowledge"]["policy_rules"][0]["rule_id"], "policy-1")
            self.assertEqual(
                written_manifest["retrieved_knowledge"]["session_memory"]["current_objective"],
                "Keep branch search grounded.",
            )

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

    def test_build_critic_skips_provider_for_submission_candidate_plan(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            config = load_config(root)

            output_dir = root / "artifacts" / "critic-submission-output"
            output_dir.mkdir(parents=True)
            input_manifest_path = output_dir / "input_manifest.json"
            run = RunRecord(
                run_id="run-critic-submission-skip",
                experiment_id="exp-critic-submission-skip",
                work_item_id="workitem-critic-submission-skip",
                spec_id="",
                status="succeeded",
                command="python train_sed.py",
                cwd=str(root),
                run_dir=str(root / "artifacts" / "runs" / "run-critic-submission-skip"),
                log_path=str(root / "artifacts" / "runs" / "run-critic-submission-skip" / "train.log"),
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
                stage_run_id="stage-critic-submission-skip",
                run_id=run.run_id,
                work_item_id=run.work_item_id,
                stage_name="critic",
                status="running",
                input_ref=run.run_id,
                output_dir=str(output_dir),
                output_json_path=str(output_dir / "critic.json"),
                output_md_path=str(output_dir / "critic.md"),
            )
            plan_payload = {
                "stage": "plan",
                "plan_status": "submission_candidate",
                "branch_role": "submission",
                "idea_class": "baseline_submission",
            }
            codegen_payload = {
                "stage": "codegen",
                "status": "noop",
                "verify_status": "skipped",
            }

            completed: dict[str, object] = {}

            def _latest_payload(_state: WorkspaceState, _run_id: str, stage_name: str) -> dict[str, object]:
                if stage_name == "plan":
                    return plan_payload
                if stage_name == "codegen":
                    return codegen_payload
                return {}

            with patch("kaggle_agent.decision.critic.latest_stage_payload", side_effect=_latest_payload), patch(
                "kaggle_agent.decision.critic.begin_stage_run",
                return_value=(stage_run, input_manifest_path),
            ), patch(
                "kaggle_agent.decision.critic.run_configured_stage_adapter",
            ) as adapter_mock, patch(
                "kaggle_agent.decision.critic.complete_stage_run",
                side_effect=lambda _stage_run, **kwargs: completed.update(kwargs),
            ):
                build_critic(config, state, run.run_id)

            adapter_mock.assert_not_called()
            self.assertEqual(completed["payload"]["status"], "approved")
            self.assertIn("was not invoked", completed["payload"]["warnings"][0])
            self.assertIn("Provider invocation: skipped", completed["markdown"])

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


class BranchSearchKernelTests(unittest.TestCase):
    def test_claim_identity_merges_same_scope(self) -> None:
        state = _empty_workspace_state()
        cards = [
            {
                "card_id": "card-a",
                "component": "pseudo_label",
                "stance": "positive",
                "summary": "soft pseudo labels remain the biggest win",
                "title": "Pseudo Labels",
                "confidence": 0.9,
                "source_path": "memory/policies/pseudo_kd.md",
            },
            {
                "card_id": "card-b",
                "component": "pseudo_label",
                "stance": "positive",
                "summary": "teacher distillation still dominates the frontier",
                "title": "Pseudo Labels Followup",
                "confidence": 0.82,
                "source_path": "memory/playbooks/pseudo_kd.md",
            },
            {
                "card_id": "card-c",
                "component": "backbone",
                "stance": "positive",
                "summary": "B0 remains the safer leader",
                "title": "Backbone",
                "confidence": 0.75,
                "source_path": "memory/families/b0.md",
            },
        ]

        synchronize_claims(state, cards)

        pseudo_claims = [item for item in state.claims if item.component == "pseudo_label" and item.stance == "positive"]
        backbone_claims = [item for item in state.claims if item.component == "backbone" and item.stance == "positive"]
        self.assertEqual(len(pseudo_claims), 1)
        self.assertEqual(len(backbone_claims), 1)
        self.assertEqual(sorted(pseudo_claims[0].support_ids), ["card-a", "card-b"])
        self.assertEqual(pseudo_claims[0].seed_support_count, 2)

    def test_search_envelope_persists_latest_turn(self) -> None:
        state = _empty_workspace_state()
        record_search_envelope(
            state,
            run_id="run-0001",
            stage_run_id="stage-decision-1",
            family="birdclef",
            envelope_payload={"turn_id": "turn-1", "grounded_branch_slots": 2, "novel_branch_slots": 0},
        )
        record_search_envelope(
            state,
            run_id="run-0001",
            stage_run_id="stage-decision-2",
            family="birdclef",
            envelope_payload={"turn_id": "turn-2", "grounded_branch_slots": 1, "novel_branch_slots": 1},
        )

        active = active_search_envelope(state, run_id="run-0001", family="birdclef")
        self.assertIsNotNone(active)
        self.assertEqual(active.stage_run_id, "stage-decision-2")
        self.assertEqual(active.envelope["novel_branch_slots"], 1)

    def test_save_state_uses_link_id_for_evidence_links(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            state.evidence_links = [
                EvidenceLinkRecord(
                    link_id="evidence-link-a",
                    claim_id="claim-shared",
                    run_id="run-1",
                    stage_name="research",
                    source_type="curated_seed",
                    source_ref="card-a",
                    polarity="positive",
                ),
                EvidenceLinkRecord(
                    link_id="evidence-link-b",
                    claim_id="claim-shared",
                    run_id="run-1",
                    stage_name="research",
                    source_type="curated_seed",
                    source_ref="card-b",
                    polarity="positive",
                ),
            ]

            save_state(config, state)
            reloaded = load_state(config)

            self.assertEqual([item.link_id for item in reloaded.evidence_links], ["evidence-link-a", "evidence-link-b"])

    def test_planner_prunes_low_information_and_keeps_grounded_novel_budget(self) -> None:
        state = _empty_workspace_state()
        source_config = {
            "training": {
                "probe_min_pos": 8,
                "probe_pca_dim": 32,
                "learning_rate": 0.001,
            }
        }
        search_envelope = {
            "slot_budget": 2,
            "grounded_branch_slots": 1,
            "novel_branch_slots": 1,
            "forbidden_patterns": ["blend_only", "calibration_only"],
            "required_patterns": ["coverage_first"],
            "minimum_information_gain_bar": 0.4,
            "per_portfolio_cap": 1,
            "per_idea_class_cap": 1,
            "novel_max_cost_tier": "medium",
        }
        decision = {"grounded_branch_slots": 1, "novel_branch_slots": 1}
        policy = {"target_branch_count": 2, "per_portfolio_cap": 1, "per_idea_class_cap": 1, "deprioritized_axes": []}
        branch_inputs = [
            {
                "title": "coverage-first branch",
                "branch_role": "primary",
                "idea_class": "class_coverage",
                "config_overrides": [{"path": "training.probe_min_pos", "value": 4}],
            },
            {
                "title": "novel probe branch",
                "branch_role": "explore",
                "idea_class": "probe_head",
                "grounding_mode": "novel",
                "unsupported_claims": ["new probe regularizer may help"],
                "required_evidence": ["smoke canary before full train"],
                "config_overrides": [{"path": "training.probe_pca_dim", "value": 96}],
            },
            {
                "title": "prior fusion blend sweep",
                "branch_role": "hedge",
                "idea_class": "prior_calibration",
                "config_overrides": [{"path": "prior.temperature", "value": 0.8}],
            },
        ]

        _scored, selected, pruned, _overridden, _trace = _prune_branch_candidates(
            branch_inputs,
            state=state,
            run_id="run-0001",
            stage_run_id="stage-plan-1",
            family="birdclef",
            source_config=source_config,
            policy_rules=[],
            branch_memories=[],
            policy=policy,
            search_envelope=search_envelope,
            decision=decision,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual({item["grounding_mode"] for item in selected}, {"grounded", "novel"})
        self.assertTrue(any(item["idea_class"] == "class_coverage" for item in selected))
        self.assertTrue(any(item["idea_class"] == "probe_head" for item in selected))
        self.assertTrue(any(item["title"] == "prior fusion blend sweep" and item["pruned_reason"] == "policy_veto" for item in pruned))

    def test_typing_drift_rejects_critic_gate(self) -> None:
        state = _empty_workspace_state()
        source_config = {"training": {"probe_min_pos": 8}}
        proposal_typing = compile_proposal_typing(
            state,
            run_id="run-0001",
            stage_run_id="stage-plan-1",
            family="birdclef",
            title="coverage branch",
            branch_input={
                "title": "coverage branch",
                "branch_role": "primary",
                "idea_class": "class_coverage",
                "config_overrides": [{"path": "training.probe_min_pos", "value": 4}],
            },
            source_config=source_config,
        )
        info_gain_estimate = estimate_info_gain(
            state,
            run_id="run-0001",
            stage_run_id="stage-plan-1",
            family="birdclef",
            title="coverage branch",
            branch_input={"branch_role": "primary"},
            proposal_typing=proposal_typing,
            search_envelope={"minimum_information_gain_bar": 0.4, "required_patterns": ["coverage_first"]},
        )
        realized_typing = compile_realized_typing(
            run_id="run-0001",
            stage_run_id="stage-codegen-1",
            family="birdclef",
            title="coverage branch",
            branch_input={"title": "coverage branch"},
            proposal_typing=proposal_typing,
            source_config_path="",
            fallback_source_config=source_config,
            generated_config_path="",
            changed_files=["prior.temperature"],
        )
        plan = {
            "family": "birdclef",
            "search_envelope": {
                "forbidden_patterns": ["calibration_only", "blend_only"],
                "required_patterns": ["coverage_first"],
                "minimum_information_gain_bar": 0.4,
                "novel_max_cost_tier": "medium",
            },
            "branch_plans": [
                {
                    "title": "coverage branch",
                    "family": "birdclef",
                    "proposal_typing": proposal_typing,
                    "proposal_typing_id": proposal_typing["proposal_typing_id"],
                    "info_gain_estimate": info_gain_estimate,
                    "override_reason": "",
                }
            ],
        }
        codegen = {
            "proposal_typing": proposal_typing,
            "realized_typing": realized_typing,
            "info_gain_estimate": info_gain_estimate,
            "branch_role": "primary",
            "idea_class": "class_coverage",
            "verify_status": "passed",
        }
        payload = {
            "stage": "critic",
            "status": "approved",
            "concerns": [],
            "warnings": [],
            "required_fixes": [],
            "branch_quality": {},
            "branch_memory_ids": [],
            "reusable_judgments": [],
        }

        enforced = _apply_typing_contract(state, "run-0001", plan, codegen, payload)

        self.assertEqual(enforced["status"], "rejected")
        self.assertTrue(enforced["typing_drift"]["severe"])
        self.assertTrue(enforced["envelope_violations"])

    def test_retrieve_knowledge_bundle_uses_reducer_backed_truth(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            knowledge_file = config.knowledge_path("01_validated_findings.md")
            knowledge_file.write_text(
                "# Findings\n\n## Soft Pseudo Labels\nSoft pseudo labels are the biggest win.\n\n## No PCEN\nNo PCEN hurts validation badly.\n",
                encoding="utf-8",
            )
            state = load_state(config)

            bundle = retrieve_knowledge_bundle(
                config,
                {"run": {"run_id": "run-knowledge"}, "experiment": {"family": "birdclef"}},
                stage="plan",
                state=state,
            )

            self.assertTrue(bundle["policy_rules"])
            self.assertTrue(bundle["claims"])
            self.assertIn("session_memory", bundle)
            self.assertIn("policy_cards", bundle)
            policy_types = {item.get("policy_type") for item in bundle["policy_rules"]}
            self.assertTrue({"prefer", "avoid"} & policy_types)

    def test_retrieve_knowledge_bundle_from_root_prefers_live_reducer_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            knowledge_file = config.knowledge_path("01_validated_findings.md")
            knowledge_file.write_text(
                "# Findings\n\n## Soft Pseudo Labels\nSoft pseudo labels are the biggest win.\n",
                encoding="utf-8",
            )
            state = load_state(config)
            state.branch_memories.append(
                BranchMemoryRecord(
                    memory_id="memory-live-0001",
                    run_id="run-old",
                    work_item_id="workitem-old",
                    experiment_id="exp-old",
                    family="birdclef",
                    idea_class="pseudo_label",
                    branch_role="primary",
                    outcome="improved",
                    summary="Pseudo-label branch improved holdout.",
                    signal_score=2.4,
                    created_at="2026-04-02T00:00:00Z",
                )
            )
            save_state(config, state)

            bundle = retrieve_knowledge_bundle_from_root(
                root,
                {"run": {"run_id": "run-live"}, "experiment": {"family": "birdclef"}},
                stage="plan",
            )

            self.assertTrue(bundle["policy_rules"])
            self.assertTrue(bundle["branch_memories"])
            self.assertTrue(bundle["capability_results"])
            self.assertIn("ledger_miner", bundle["capability_results"])

    def test_session_memory_is_rebuilt_instead_of_merged(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)
            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            config.session_memory_json_path().write_text(
                json.dumps(
                    {
                        "current_objective": "stale",
                        "top_positive_priors": ["stale prior"],
                        "top_negative_vetoes": ["stale veto"],
                        "unresolved_questions": ["stale question"],
                    }
                ),
                encoding="utf-8",
            )
            knowledge_file = config.knowledge_path("01_validated_findings.md")
            knowledge_file.write_text(
                "# Findings\n\n## Coverage\nCoverage fixes remain preferred.\n",
                encoding="utf-8",
            )
            state = load_state(config)

            bundle = retrieve_knowledge_bundle(
                config,
                {"run": {"run_id": "run-memory"}, "experiment": {"family": "birdclef"}, "report": {"root_cause": "coverage"}},
                stage="plan",
                state=state,
            )

            self.assertNotIn("stale prior", bundle["session_memory"].get("top_positive_priors", []))
            self.assertNotIn("stale veto", bundle["session_memory"].get("top_negative_vetoes", []))
            self.assertNotIn("stale question", bundle["session_memory"].get("unresolved_questions", []))

    def test_choose_next_work_items_respects_cost_budget_and_novel_share(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_runtime(root)
            _write_workspace(root)
            _build_debug_dataset(root)

            config = load_config(root)
            init_workspace(config, archive_legacy=False, force=True)
            state = load_state(config)
            config_path = str((root / "BirdCLEF-2026-Codebase" / "configs" / "default.yaml").relative_to(root))
            state.work_items = [
                WorkItem(
                    id="workitem-grounded",
                    title="Grounded branch",
                    work_type="experiment_iteration",
                    family="perch_cached_probe",
                    priority=20,
                    status="queued",
                    config_path=config_path,
                    pipeline=list(state.work_items[0].pipeline),
                    portfolio_id="portfolio-a",
                    idea_class="class_coverage",
                    branch_role="primary",
                    scheduler_hints={
                        "portfolio_cap": 2,
                        "idea_class_cap": 1,
                        "dispatch_priority": 8.0,
                        "grounding_mode": "grounded",
                        "cost_tier": "low",
                        "cost_units": 1.0,
                        "cost_budget": 4.0,
                        "max_budget_share": 0.35,
                        "cost_caps": {"low": 2.0, "medium": 1.0, "high": 0.0},
                    },
                ),
                WorkItem(
                    id="workitem-novel-high",
                    title="Novel high-cost branch",
                    work_type="experiment_iteration",
                    family="perch_cached_probe",
                    priority=21,
                    status="queued",
                    config_path=config_path,
                    pipeline=list(state.work_items[0].pipeline),
                    portfolio_id="portfolio-a",
                    idea_class="backbone",
                    branch_role="explore",
                    scheduler_hints={
                        "portfolio_cap": 2,
                        "idea_class_cap": 1,
                        "dispatch_priority": 9.0,
                        "grounding_mode": "novel",
                        "cost_tier": "high",
                        "cost_units": 4.0,
                        "cost_budget": 4.0,
                        "max_budget_share": 0.35,
                        "cost_caps": {"low": 2.0, "medium": 1.0, "high": 0.0},
                    },
                ),
            ]
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

            self.assertEqual([item.id for item in selected], ["workitem-grounded"])


if __name__ == "__main__":
    unittest.main()
