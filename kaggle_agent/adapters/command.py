from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from kaggle_agent.utils import atomic_write_text, truncate


class CommandAdapterError(RuntimeError):
    pass


def run_command_adapter(
    command: str,
    *,
    stage: str,
    workspace_root: Path,
    input_path: Path,
    output_path: Path,
    extra_env: dict[str, str] | None = None,
) -> str:
    env = os.environ.copy()
    env["KAGGLE_AGENT_STAGE"] = stage
    env["KAGGLE_AGENT_WORKSPACE_ROOT"] = str(workspace_root)
    env["KAGGLE_AGENT_INPUT_FILE"] = str(input_path)
    env["KAGGLE_AGENT_OUTPUT_FILE"] = str(output_path)
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        command,
        shell=True,
        cwd=workspace_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + ("\n" if completed.stdout and not completed.stdout.endswith("\n") else "") + completed.stderr
    if completed.returncode != 0:
        raise CommandAdapterError(f"{stage} adapter failed: {truncate(combined or 'no output')}")
    if output_path.exists():
        return output_path.read_text(encoding="utf-8")
    if combined.strip():
        atomic_write_text(output_path, combined)
        return combined
    raise CommandAdapterError(f"{stage} adapter returned no output.")


def parse_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)
