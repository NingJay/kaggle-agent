from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from kaggle_agent.utils import atomic_write_text, ensure_directory, truncate


class CommandAdapterError(RuntimeError):
    pass


class CommandAdapterUnavailable(CommandAdapterError):
    pass


ADAPTER_UNAVAILABLE_EXIT_CODE = 41


def run_stage_adapter(
    command: str,
    *,
    stage: str,
    workspace_root: Path,
    input_manifest_path: Path,
    output_dir: Path,
    prompt_path: Path | None = None,
    extra_env: dict[str, str] | None = None,
    shell_init: str = "",
    conda_env: str = "",
) -> dict[str, Path]:
    ensure_directory(output_dir)
    env = os.environ.copy()
    env["KAGGLE_AGENT_STAGE"] = stage
    env["KAGGLE_AGENT_WORKSPACE_ROOT"] = str(workspace_root)
    env["KAGGLE_AGENT_INPUT_MANIFEST"] = str(input_manifest_path)
    env["KAGGLE_AGENT_OUTPUT_DIR"] = str(output_dir)
    if prompt_path is not None:
        env["KAGGLE_AGENT_PROMPT_FILE"] = str(prompt_path)
    if extra_env:
        env.update(extra_env)

    bootstrap_lines = ["set -e"]
    if shell_init.strip():
        bootstrap_lines.append(shell_init.strip())
    if conda_env.strip():
        bootstrap_lines.append(f"conda activate {shlex.quote(conda_env)}")
    bootstrap_lines.append(command)

    completed = subprocess.run(
        ["/usr/bin/bash", "-c", "\n".join(bootstrap_lines)],
        cwd=workspace_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + ("\n" if completed.stdout and not completed.stdout.endswith("\n") else "") + completed.stderr
    if completed.returncode == ADAPTER_UNAVAILABLE_EXIT_CODE:
        raise CommandAdapterUnavailable(truncate(combined or f"{stage} adapter unavailable"))
    if completed.returncode != 0:
        raise CommandAdapterError(f"{stage} adapter failed: {truncate(combined or 'no output')}")

    json_path = output_dir / f"{stage}.json"
    md_path = output_dir / f"{stage}.md"
    meta_path = output_dir / "provider_meta.json"
    spec_path = output_dir / "spec.yaml"
    if not json_path.exists():
        if completed.stdout.strip().startswith("{"):
            atomic_write_text(json_path, completed.stdout.strip() + "\n")
        else:
            raise CommandAdapterError(f"{stage} adapter did not write {json_path.name}.")
    if not md_path.exists():
        summary = combined.strip() or f"{stage} adapter completed without markdown output."
        atomic_write_text(md_path, summary + ("\n" if not summary.endswith("\n") else ""))
    result = {"json_path": json_path, "md_path": md_path}
    if meta_path.exists():
        result["meta_path"] = meta_path
    if spec_path.exists():
        result["spec_path"] = spec_path
    return result


def parse_json_payload(path_or_text: str | Path) -> dict[str, Any]:
    if isinstance(path_or_text, Path):
        text = path_or_text.read_text(encoding="utf-8")
    else:
        text = path_or_text
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)
