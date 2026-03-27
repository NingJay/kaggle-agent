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
from kaggle_agent.adapters.stage_wrapper import StageContext, main as stage_wrapper_main


REPO_ROOT = Path(__file__).resolve().parents[1]


def _skip_live() -> bool:
    return os.environ.get("KAGGLE_AGENT_RUN_LIVE_PROVIDER_TESTS") != "1"


def _skip_amp() -> bool:
    return os.environ.get("KAGGLE_AGENT_RUN_AMP_SMOKE") != "1"


class StageWrapperCompatibilityTests(unittest.TestCase):
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
                payload={
                    "stage": "codegen",
                    "status": "noop",
                    "reason": "legacy flag accepted",
                    "markdown": "# Codegen\n\n- noop",
                },
            )
            with patch("kaggle_agent.adapters.stage_wrapper.StageContext.from_env", return_value=ctx), patch(
                "kaggle_agent.adapters.stage_wrapper._build_prompt",
                return_value="prompt",
            ), patch(
                "kaggle_agent.adapters.stage_wrapper._run_provider",
                return_value=(response, None),
            ), patch(
                "kaggle_agent.adapters.stage_wrapper.validate_payload",
                return_value=None,
            ):
                exit_code = stage_wrapper_main(["--provider", "codex", "--dangerously-bypass-approvals-and-sandbox"])

            self.assertEqual(exit_code, 0)
            payload = json.loads((output_dir / "codegen.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "noop")
            self.assertEqual(payload["reason"], "legacy flag accepted")


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
        self.assertEqual(meta["provider"], "claude" if provider == "critic" else provider)
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
