from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from kaggle_agent.adapters.providers import ProviderResponse
from kaggle_agent.adapters.providers.claude_code_exec import run_claude_code_exec
from kaggle_agent.adapters.providers.claude_headless import run_claude_headless
from kaggle_agent.adapters.providers.codex_exec import run_codex_exec
from kaggle_agent.adapters.schema_validation import SchemaValidationError
from kaggle_agent.decision.helpers import _archive_conflicting_stage_dir, _relocate_stage_output, _rewrite_text_file_paths
from kaggle_agent.schema import StageRun
from kaggle_agent.adapters.stage_wrapper import (
    CodegenWorkspace,
    StageContext,
    _allow_codegen_paths,
    _build_prompt,
    _materialize_codegen,
    _prepare_stage_workspace,
    main as stage_wrapper_main,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _skip_live() -> bool:
    return os.environ.get("KAGGLE_AGENT_RUN_LIVE_PROVIDER_TESTS") != "1"


def _skip_amp() -> bool:
    return os.environ.get("KAGGLE_AGENT_RUN_AMP_SMOKE") != "1"


class StageWrapperCompatibilityTests(unittest.TestCase):
    def test_archive_conflicting_stage_dir_moves_stale_directory_aside(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            running_dir = root / "01-evidence__running"
            running_dir.mkdir(parents=True)
            (running_dir / "input_manifest.json").write_text("{}", encoding="utf-8")

            archived = _archive_conflicting_stage_dir(running_dir)

            self.assertIsNotNone(archived)
            assert archived is not None
            self.assertFalse(running_dir.exists())
            self.assertTrue(archived.exists())
            self.assertIn("__stale_", archived.name)
            self.assertTrue((archived / "input_manifest.json").exists())

    def test_relocate_stage_output_archives_existing_target_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_dir = root / "01-evidence__running"
            old_dir.mkdir(parents=True)
            (old_dir / "evidence.json").write_text('{"status":"running"}\n', encoding="utf-8")
            (old_dir / "evidence.md").write_text("running\n", encoding="utf-8")
            stale_target = root / "01-evidence__succeeded"
            stale_target.mkdir(parents=True)
            (stale_target / "stale.txt").write_text("stale\n", encoding="utf-8")
            stage_run = StageRun(
                stage_run_id="stage-9999-evidence",
                run_id="run-9999",
                work_item_id="workitem-9999",
                stage_name="evidence",
                status="running",
                input_ref="run-9999",
                output_dir=str(old_dir),
                output_json_path=str(old_dir / "evidence.json"),
                output_md_path=str(old_dir / "evidence.md"),
            )

            _relocate_stage_output(None, stage_run, payload={"status": "succeeded"}, status="completed")

            target_dir = root / "01-evidence__succeeded"
            archived_targets = list(root.glob("01-evidence__succeeded__stale_*"))
            self.assertTrue(target_dir.exists())
            self.assertTrue((target_dir / "evidence.json").exists())
            self.assertEqual(len(archived_targets), 1)
            self.assertTrue((archived_targets[0] / "stale.txt").exists())

    def test_rewrite_text_file_paths_rebases_markdown_links(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_dir = root / "06-codegen__running"
            new_dir = root / "06-codegen__generated"
            new_dir.mkdir(parents=True)
            markdown_path = new_dir / "codegen.md"
            markdown_path.write_text(f"Patch path: {old_dir / 'patch.diff'}\n", encoding="utf-8")

            _rewrite_text_file_paths(markdown_path, old_dir, new_dir)

            self.assertEqual(
                markdown_path.read_text(encoding="utf-8"),
                f"Patch path: {new_dir / 'patch.diff'}\n",
            )

    def test_prepare_stage_workspace_excludes_runtime_outputs_from_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_outputs = root / "BirdCLEF-2026-Codebase" / "outputs" / "baseline"
            runtime_outputs.mkdir(parents=True, exist_ok=True)
            (runtime_outputs / "result.json").write_text("{}", encoding="utf-8")
            input_manifest_path = root / "input_manifest.json"
            input_manifest_path.write_text("{}", encoding="utf-8")
            output_dir = root / "output"
            output_dir.mkdir()
            schema_path = root / "codegen.schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            ctx = StageContext(
                stage="codegen",
                workspace_root=root,
                input_manifest_path=input_manifest_path,
                output_dir=output_dir,
                prompt_path=None,
                schema_path=schema_path,
                input_manifest={},
            )

            stage_workspace = _prepare_stage_workspace(ctx)

            self.assertFalse((stage_workspace.workspace_root / "BirdCLEF-2026-Codebase" / "outputs").exists())

    def test_stage_wrapper_accepts_legacy_bypass_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()
            schema_path = root / "codegen.schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            ctx = StageContext(
                stage="codegen",
                workspace_root=root,
                input_manifest_path=root / "input_manifest.json",
                output_dir=output_dir,
                prompt_path=None,
                schema_path=schema_path,
                input_manifest={},
            )
            response = ProviderResponse(
                provider="codex",
                payload={},
            )
            codegen_workspace = CodegenWorkspace(
                snapshot_root=root / "snapshot",
                workspace_root=root / "snapshot" / "workspace",
                verify_root=root / "snapshot" / "verify_runtime",
                base_commit="abc123",
                expected_config_relpath="BirdCLEF-2026-Codebase/configs/generated/test.yaml",
            )
            with patch("kaggle_agent.adapters.stage_wrapper.StageContext.from_env", return_value=ctx), patch(
                "kaggle_agent.adapters.stage_wrapper._prepare_codegen_workspace",
                return_value=codegen_workspace,
            ), patch(
                "kaggle_agent.adapters.stage_wrapper._build_prompt",
                return_value="prompt",
            ), patch(
                "kaggle_agent.adapters.stage_wrapper._run_provider",
                return_value=(response, None),
            ), patch(
                "kaggle_agent.adapters.stage_wrapper._materialize_codegen",
                return_value={
                    "stage": "codegen",
                    "status": "noop",
                    "reason": "legacy flag accepted",
                    "generated_config_path": "",
                    "run_bundle_path": "",
                    "patch_path": "",
                    "code_state_ref": "",
                    "verify_artifacts_ref": "",
                    "verify_command": "",
                    "verify_status": "skipped",
                    "verify_summary": "legacy path",
                    "worktree_path": "",
                    "base_commit": "abc123",
                    "head_commit": "",
                    "changed_files": [],
                    "provider_runtime": "codex mode:agentic env:inherit",
                    "allowed_edit_roots": ["train_sed.py"],
                    "smoke_status": "skipped",
                    "smoke_summary": "legacy path",
                },
            ), patch(
                "kaggle_agent.adapters.stage_wrapper.validate_payload",
                return_value=None,
            ):
                exit_code = stage_wrapper_main(["--provider", "codex", "--dangerously-bypass-approvals-and-sandbox"])

            self.assertEqual(exit_code, 0)
            payload = json.loads((output_dir / "codegen.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "noop")
            self.assertEqual(payload["reason"], "legacy flag accepted")


class CodegenGuardrailTests(unittest.TestCase):
    def test_codegen_path_allowlist_rejects_runtime_outputs_and_binary_artifacts(self) -> None:
        _allow_codegen_paths(
            [
                "train_sed.py",
                "BirdCLEF-2026-Codebase/configs/generated/adapter_codegen.yaml",
                "BirdCLEF-2026-Codebase/src/birdclef_runtime/training.py",
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "outputs"):
            _allow_codegen_paths(["BirdCLEF-2026-Codebase/outputs/oof_predictions.npz"])
        with self.assertRaisesRegex(RuntimeError, "notebook"):
            _allow_codegen_paths(["BirdCLEF-2026-Codebase/configs/generated/analysis.ipynb"])
        with self.assertRaisesRegex(RuntimeError, "artifact"):
            _allow_codegen_paths(["BirdCLEF-2026-Codebase/src/birdclef_runtime/probe_bundle.pkl"])

    def test_codex_exec_inherits_user_environment_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()
            completed = subprocess.CompletedProcess(
                args=["codex", "exec"],
                returncode=0,
                stdout='{"event":"thread.started","thread_id":"agentic-thread"}\n',
                stderr="",
            )
            with patch.dict(os.environ, {"HOME": "/tmp/codex-home"}, clear=True), patch(
                "kaggle_agent.adapters.providers.codex_exec.shutil.which",
                return_value="/usr/bin/codex",
            ), patch(
                "kaggle_agent.adapters.providers.codex_exec.subprocess.run",
                return_value=completed,
            ) as run_mock:
                response = run_codex_exec(
                    prompt="edit files",
                    schema_path=None,
                    workspace_root=root,
                    output_dir=output_dir,
                    mode="agentic",
                    extra_env={"KAGGLE_AGENT_RUN_DIR": "/tmp/verify-root", "PYTHONDONTWRITEBYTECODE": "1"},
                )

        called_args = run_mock.call_args.args[0]
        called_env = run_mock.call_args.kwargs["env"]
        self.assertNotIn("--profile", called_args)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", called_args)
        self.assertNotIn("--full-auto", called_args)
        self.assertEqual(called_env["HOME"], "/tmp/codex-home")
        self.assertEqual(called_env["KAGGLE_AGENT_RUN_DIR"], "/tmp/verify-root")
        self.assertEqual(called_env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertNotIn("CODEX_HOME", called_env)
        self.assertNotIn("XDG_CONFIG_HOME", called_env)
        self.assertEqual(response.provider, "codex")
        self.assertEqual(response.extra_meta["provider_runtime"], "codex mode:agentic env:inherit")

    def test_claude_code_exec_uses_claude_cli_for_agentic(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = subprocess.CompletedProcess(
                args=["claude", "-p"],
                returncode=0,
                stdout='{"session_id":"claude-code-agentic","result":"edited files"}\n',
                stderr="",
            )
            with patch.dict(os.environ, {"CLAUDE_CODE_SSE_PORT": "31741"}, clear=False), patch(
                "kaggle_agent.adapters.providers.claude_runtime.load_user_claude_settings",
                return_value={"env": {"ANTHROPIC_AUTH_TOKEN": "token"}, "model": "opus"},
            ), patch(
                "kaggle_agent.adapters.providers.claude_code_exec.shutil.which",
                return_value="/usr/bin/claude",
            ), patch(
                "kaggle_agent.adapters.providers.claude_code_exec._supports_flag",
                return_value=True,
            ), patch(
                "kaggle_agent.adapters.providers.claude_code_exec.subprocess.run",
                return_value=completed,
            ) as run_mock:
                response = run_claude_code_exec(
                    prompt="edit files",
                    schema_path=None,
                    workspace_root=root,
                    mode="agentic",
                    extra_env={
                        "KAGGLE_AGENT_RUN_DIR": "/tmp/verify-root",
                        "KAGGLE_AGENT_DISABLE_OUTPUT_MIRROR": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )

        called_args = run_mock.call_args.args[0]
        called_env = run_mock.call_args.kwargs["env"]
        self.assertEqual(run_mock.call_args.kwargs["cwd"], root)
        self.assertIn("--dangerously-skip-permissions", called_args)
        self.assertIn("--output-format", called_args)
        self.assertIn("json", called_args)
        self.assertIn("--add-dir", called_args)
        self.assertIn(str(root), called_args)
        self.assertIn("--no-chrome", called_args)
        self.assertNotIn("CLAUDE_CODE_SSE_PORT", called_env)
        self.assertEqual(called_env["ANTHROPIC_AUTH_TOKEN"], "token")
        self.assertEqual(called_env["KAGGLE_AGENT_RUN_DIR"], "/tmp/verify-root")
        self.assertEqual(called_env["KAGGLE_AGENT_DISABLE_OUTPUT_MIRROR"], "1")
        self.assertEqual(called_env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertNotEqual(called_env["HOME"], os.environ.get("HOME"))
        self.assertEqual(response.provider, "claude_code")
        self.assertEqual(response.session_id, "claude-code-agentic")
        self.assertEqual(response.extra_meta["provider_runtime"], "claude_code mode:agentic")
        self.assertEqual(response.extra_meta["isolated_home"], "1")

    def test_claude_code_exec_can_disable_home_isolation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = subprocess.CompletedProcess(
                args=["claude", "-p"],
                returncode=0,
                stdout='{"session_id":"claude-code-agentic","result":"edited files"}\n',
                stderr="",
            )
            with patch.dict(
                os.environ,
                {
                    "KAGGLE_AGENT_CLAUDE_CODE_ISOLATE_HOME": "0",
                    "HOME": "/tmp/original-home",
                },
                clear=False,
            ), patch(
                "kaggle_agent.adapters.providers.claude_runtime.load_user_claude_settings",
                return_value={"env": {"ANTHROPIC_AUTH_TOKEN": "token"}, "model": "opus"},
            ), patch(
                "kaggle_agent.adapters.providers.claude_code_exec.shutil.which",
                return_value="/usr/bin/claude",
            ), patch(
                "kaggle_agent.adapters.providers.claude_code_exec._supports_flag",
                return_value=True,
            ), patch(
                "kaggle_agent.adapters.providers.claude_code_exec.subprocess.run",
                return_value=completed,
            ) as run_mock:
                response = run_claude_code_exec(
                    prompt="edit files",
                    schema_path=None,
                    workspace_root=root,
                    mode="agentic",
                    extra_env={"KAGGLE_AGENT_RUN_DIR": "/tmp/verify-root"},
                )

        called_args = run_mock.call_args.args[0]
        called_env = run_mock.call_args.kwargs["env"]
        self.assertIn("--no-chrome", called_args)
        self.assertEqual(called_env["HOME"], "/tmp/original-home")
        self.assertEqual(called_env["KAGGLE_AGENT_RUN_DIR"], "/tmp/verify-root")
        self.assertEqual(response.extra_meta["isolated_home"], "0")

    def test_claude_headless_uses_isolated_home_and_parses_wrapped_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = subprocess.CompletedProcess(
                args=["claude", "-p"],
                returncode=0,
                stdout='{"session_id":"claude-headless","result":"{\\"headline\\":\\"ok\\"}"}\n',
                stderr="",
            )
            with patch.dict(os.environ, {"CLAUDE_CODE_SSE_PORT": "31741"}, clear=False), patch(
                "kaggle_agent.adapters.providers.claude_runtime.load_user_claude_settings",
                return_value={"env": {"ANTHROPIC_AUTH_TOKEN": "token"}, "model": "opus"},
            ), patch(
                "kaggle_agent.adapters.providers.claude_headless.shutil.which",
                return_value="/usr/bin/claude",
            ), patch(
                "kaggle_agent.adapters.providers.claude_headless._supports_flag",
                return_value=True,
            ), patch(
                "kaggle_agent.adapters.providers.claude_headless.subprocess.run",
                return_value=completed,
            ) as run_mock:
                response = run_claude_headless(
                    prompt="report",
                    schema={"type": "object"},
                    workspace_root=root,
                )

        called_args = run_mock.call_args.args[0]
        called_env = run_mock.call_args.kwargs["env"]
        self.assertIn("--no-chrome", called_args)
        self.assertNotIn("CLAUDE_CODE_SSE_PORT", called_env)
        self.assertEqual(called_env["ANTHROPIC_AUTH_TOKEN"], "token")
        self.assertNotEqual(called_env["HOME"], os.environ.get("HOME"))
        self.assertEqual(response.provider, "claude")
        self.assertEqual(response.payload["headline"], "ok")
        self.assertEqual(response.extra_meta["isolated_home"], "1")

    def test_plan_prompt_includes_curated_knowledge_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge_root = root / "knowledge"
            knowledge_root.mkdir(parents=True)
            (knowledge_root / "00_experiment_rules.md").write_text(
                "Primary metric is val_soundscape_macro_roc_auc.\nTreat soundscape_macro_roc_auc as diagnostic only.\n",
                encoding="utf-8",
            )
            (knowledge_root / "03_next_experiment_priors.md").write_text(
                "Handle imbalance by expanding class coverage before calibration-only tuning.\n",
                encoding="utf-8",
            )
            schema_path = root / "plan.schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            ctx = StageContext(
                stage="plan",
                workspace_root=root,
                input_manifest_path=root / "input_manifest.json",
                output_dir=root / "output",
                prompt_path=None,
                schema_path=schema_path,
                input_manifest={"run": {"run_id": "run-knowledge-plan"}},
            )

            prompt = _build_prompt(ctx, provider_workspace_root=root)

        self.assertIn("# Knowledge Context", prompt)
        self.assertIn("Primary metric is val_soundscape_macro_roc_auc.", prompt)
        self.assertIn("Handle imbalance by expanding class coverage before calibration-only tuning.", prompt)

    def test_stage_wrapper_retries_structured_payload_after_schema_failure(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()
            schema_path = root / "report.schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            ctx = StageContext(
                stage="report",
                workspace_root=root,
                input_manifest_path=root / "input_manifest.json",
                output_dir=output_dir,
                prompt_path=None,
                schema_path=schema_path,
                input_manifest={},
            )
            invalid = ProviderResponse(provider="claude", payload={"markdown": "# Report\n\n- incomplete"})
            repaired = ProviderResponse(
                provider="claude",
                payload={
                    "stage": "report",
                    "headline": "Fixed",
                    "focus": "repair",
                    "best_run_id": "run-1",
                    "best_run_metric": 0.5,
                    "primary_metric_value": 0.5,
                    "root_cause": "fixed schema",
                    "verdict": "iterate",
                    "finding_titles": [],
                    "issue_titles": [],
                    "markdown": "# Report\n\n- repaired",
                },
            )
            with patch("kaggle_agent.adapters.stage_wrapper.StageContext.from_env", return_value=ctx), patch(
                "kaggle_agent.adapters.stage_wrapper._build_prompt",
                return_value="prompt",
            ), patch(
                "kaggle_agent.adapters.stage_wrapper._run_provider",
                side_effect=[(invalid, None), (repaired, None)],
            ), patch(
                "kaggle_agent.adapters.stage_wrapper.validate_payload",
                side_effect=[SchemaValidationError("$.headline is required"), None],
            ):
                exit_code = stage_wrapper_main(["--provider", "claude"])

            self.assertEqual(exit_code, 0)
            payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
            meta = json.loads((output_dir / "provider_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["headline"], "Fixed")
            self.assertEqual(meta["provider"], "claude")

    def test_stage_wrapper_writes_raw_captures_for_generic_codegen_failure(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()
            schema_path = root / "codegen.schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            ctx = StageContext(
                stage="codegen",
                workspace_root=root,
                input_manifest_path=root / "input_manifest.json",
                output_dir=output_dir,
                prompt_path=None,
                schema_path=schema_path,
                input_manifest={},
            )
            response = ProviderResponse(
                provider="claude_code",
                payload={},
                raw_stdout='{"type":"result","result":"edited"}\n',
                raw_stderr="provider stderr\n",
                event_log_text='{"type":"result","result":"edited"}\n',
            )
            codegen_workspace = CodegenWorkspace(
                snapshot_root=root / "snapshot",
                workspace_root=root / "snapshot" / "workspace",
                verify_root=root / "snapshot" / "verify_runtime",
                base_commit="abc123",
                expected_config_relpath="BirdCLEF-2026-Codebase/configs/generated/test.yaml",
            )
            with patch("kaggle_agent.adapters.stage_wrapper.StageContext.from_env", return_value=ctx), patch(
                "kaggle_agent.adapters.stage_wrapper._prepare_codegen_workspace",
                return_value=codegen_workspace,
            ), patch(
                "kaggle_agent.adapters.stage_wrapper._build_prompt",
                return_value="prompt",
            ), patch(
                "kaggle_agent.adapters.stage_wrapper._run_provider",
                return_value=(response, None),
            ), patch(
                "kaggle_agent.adapters.stage_wrapper._materialize_codegen",
                side_effect=RuntimeError("boom"),
            ):
                exit_code = stage_wrapper_main(["--provider", "claude_code"])

            self.assertEqual(exit_code, 1)
            self.assertEqual((output_dir / "raw_stdout.txt").read_text(encoding="utf-8"), response.raw_stdout)
            self.assertEqual((output_dir / "raw_stderr.txt").read_text(encoding="utf-8"), response.raw_stderr)
            self.assertEqual((output_dir / "events.jsonl").read_text(encoding="utf-8"), response.event_log_text)

    def test_materialize_codegen_accepts_provider_committed_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace_root = root / "snapshot" / "workspace"
            verify_root = root / "snapshot" / "verify_runtime"
            output_dir = root / "output"
            workspace_root.mkdir(parents=True)
            verify_root.mkdir(parents=True)
            output_dir.mkdir()

            runtime_root = workspace_root / "BirdCLEF-2026-Codebase"
            config_path = runtime_root / "configs" / "default.yaml"
            src_dir = runtime_root / "src" / "birdclef_runtime"
            src_dir.mkdir(parents=True, exist_ok=True)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("experiment:\n  name: baseline\n", encoding="utf-8")
            (workspace_root / "train_sed.py").write_text(
                "\n".join(
                    [
                        "from __future__ import annotations",
                        "",
                        "import json",
                        "import os",
                        "from pathlib import Path",
                        "",
                        "run_dir = Path(os.environ['KAGGLE_AGENT_RUN_DIR'])",
                        "run_dir.mkdir(parents=True, exist_ok=True)",
                        "payload = {",
                        "    'primary_metric_name': 'val_soundscape_macro_roc_auc',",
                        "    'primary_metric_value': 0.77,",
                        "    'verdict': 'keep',",
                        "}",
                        "(run_dir / 'result.json').write_text(json.dumps(payload), encoding='utf-8')",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(["git", "init", "-q"], cwd=workspace_root, check=True)
            subprocess.run(["git", "config", "user.email", "kaggle-agent@local"], cwd=workspace_root, check=True)
            subprocess.run(["git", "config", "user.name", "kaggle-agent"], cwd=workspace_root, check=True)
            subprocess.run(["git", "add", "."], cwd=workspace_root, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "Baseline codegen snapshot"], cwd=workspace_root, check=True)
            base_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            config_path.write_text("experiment:\n  name: expanded\n", encoding="utf-8")
            pycache_dir = src_dir / "__pycache__"
            pycache_dir.mkdir(parents=True, exist_ok=True)
            (pycache_dir / "cached_probe.cpython-313.pyc").write_bytes(b"noise")
            outputs_dir = runtime_root / "outputs" / "perch_coverage_expansion_r2"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            (outputs_dir / "result.json").write_text('{"verdict":"noise"}', encoding="utf-8")
            subprocess.run(["git", "add", str(config_path.relative_to(workspace_root))], cwd=workspace_root, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "Provider committed config change"], cwd=workspace_root, check=True)

            input_manifest_path = root / "input_manifest.json"
            input_manifest_path.write_text("{}", encoding="utf-8")
            schema_path = root / "codegen.schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            ctx = StageContext(
                stage="codegen",
                workspace_root=root,
                input_manifest_path=input_manifest_path,
                output_dir=output_dir,
                prompt_path=None,
                schema_path=schema_path,
                input_manifest={
                    "plan": {
                        "title": "Committed codegen test",
                        "family": "perch_cached_probe",
                        "config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
                        "launch_mode": "background",
                        "dedupe_key": "adapter:committed-codegen",
                    }
                },
            )
            codegen_workspace = CodegenWorkspace(
                snapshot_root=root / "snapshot",
                workspace_root=workspace_root,
                verify_root=verify_root,
                base_commit=base_commit,
                expected_config_relpath="BirdCLEF-2026-Codebase/configs/default.yaml",
            )

            payload = _materialize_codegen(ctx, codegen_workspace, provider_runtime="claude_code mode:agentic")
            self.assertEqual(payload["status"], "generated")
            self.assertEqual(payload["verify_status"], "passed")
            self.assertIn("BirdCLEF-2026-Codebase/configs/default.yaml", payload["changed_files"])
            self.assertNotIn("BirdCLEF-2026-Codebase/src/birdclef_runtime/__pycache__/cached_probe.cpython-313.pyc", payload["changed_files"])
            self.assertNotIn("BirdCLEF-2026-Codebase/outputs/perch_coverage_expansion_r2/result.json", payload["changed_files"])
            self.assertTrue(payload["head_commit"])
            self.assertTrue(Path(payload["patch_path"]).exists())
            patch_text = Path(payload["patch_path"]).read_text(encoding="utf-8")
            self.assertIn("name: expanded", patch_text)
            self.assertFalse(outputs_dir.exists())


@unittest.skipIf(_skip_live(), "Set KAGGLE_AGENT_RUN_LIVE_PROVIDER_TESTS=1 to run live provider smoke tests.")
class LiveAdapterSmokeTests(unittest.TestCase):
    def _run_stage(self, *, tmp_path: Path, stage: str, provider: str, manifest: dict[str, object]) -> tuple[subprocess.CompletedProcess[str], Path]:
        output_dir = tmp_path / "output"
        manifest_path = tmp_path / "input_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "KAGGLE_AGENT_STAGE": stage,
                "KAGGLE_AGENT_WORKSPACE_ROOT": str(REPO_ROOT),
                "KAGGLE_AGENT_INPUT_MANIFEST": str(manifest_path),
                "KAGGLE_AGENT_OUTPUT_DIR": str(output_dir),
                "KAGGLE_AGENT_PROMPT_FILE": str(REPO_ROOT / "prompts" / f"{stage}.md"),
            }
        )
        completed = subprocess.run(
            [sys.executable, "-m", "kaggle_agent.adapters.stage_wrapper", "--provider", provider],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"{stage}/{provider} failed with code {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        payload = json.loads((output_dir / f"{stage}.json").read_text(encoding="utf-8"))
        meta = json.loads((output_dir / "provider_meta.json").read_text(encoding="utf-8"))
        expected_provider = "claude" if provider == "critic" else provider
        self.assertEqual(meta["provider"], expected_provider)
        self.assertEqual(payload["stage"], stage)
        return completed, output_dir

    def test_claude_report_stage(self) -> None:
        with TemporaryDirectory() as tmp:
            _completed, output_dir = self._run_stage(
                tmp_path=Path(tmp),
                stage="report",
                provider="claude",
                manifest={
                    "run": {"run_id": "run-live-report", "status": "succeeded", "primary_metric_value": 0.52},
                    "experiment": {"id": "exp-live-report", "family": "perch_cached_probe"},
                    "work_item": {"id": "workitem-live-report", "title": "live report"},
                    "evidence": {"root_cause": "runtime completed", "verdict": "baseline-ready"},
                    "recent_findings": [],
                    "recent_issues": [],
                    "leader_run": {},
                },
            )
            payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["headline"])

    def test_codex_plan_stage(self) -> None:
        with TemporaryDirectory() as tmp:
            _completed, output_dir = self._run_stage(
                tmp_path=Path(tmp),
                stage="plan",
                provider="codex",
                manifest={
                    "run": {"run_id": "run-live-plan", "status": "succeeded"},
                    "experiment": {"id": "exp-live-plan", "family": "perch_cached_probe", "config_path": "BirdCLEF-2026-Codebase/configs/default.yaml"},
                    "decision": {
                        "stage": "decision",
                        "decision_type": "tune",
                        "next_action": "run_new_experiment",
                        "submission_recommendation": "no",
                        "root_cause": "runtime completed",
                        "why": "Test Codex plan generation.",
                        "next_title": "Live planned run",
                        "next_family": "perch_cached_probe",
                        "next_config_path": "BirdCLEF-2026-Codebase/configs/default.yaml",
                        "priority_delta": 10,
                        "launch_mode": "background",
                        "requires_human": False,
                    },
                    "knowledge_context": "Keep execution config-path oriented.",
                },
            )
            payload = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["plan_status"], "planned")
            self.assertTrue((output_dir / "spec.yaml").exists())

    @unittest.skipIf(_skip_amp(), "Set KAGGLE_AGENT_RUN_AMP_SMOKE=1 to include the optional Amp smoke test.")
    def test_amp_cli_stream_json(self) -> None:
        amp = shutil.which("amp")
        self.assertTrue(amp, "amp binary is not available on PATH")
        completed = subprocess.run(
            [amp, "--no-ide", "--no-jetbrains", "-x", "Reply with a short confirmation that Amp execute mode is working.", "--stream-json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(f"amp failed with code {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
        events = []
        for line in completed.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
        self.assertTrue(any(event.get("type") == "assistant" for event in events))
