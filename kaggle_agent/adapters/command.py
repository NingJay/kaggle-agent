from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from kaggle_agent.utils import atomic_write_text, ensure_directory, truncate


class CommandAdapterError(RuntimeError):
    pass


class CommandAdapterUnavailable(CommandAdapterError):
    pass


class CommandAdapterTimeout(CommandAdapterError):
    pass


ADAPTER_UNAVAILABLE_EXIT_CODE = 41


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


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
    timeout_seconds: int | None = None,
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

    process = subprocess.Popen(
        ["/usr/bin/bash", "-c", "\n".join(bootstrap_lines)],
        cwd=workspace_root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        combined = (stdout or "") + ("\n" if stdout and not stdout.endswith("\n") else "") + (stderr or "")
        timeout_label = f"{timeout_seconds}s" if timeout_seconds is not None else "configured timeout"
        raise CommandAdapterTimeout(
            f"{stage} adapter timed out after {timeout_label}: {truncate(combined or 'no output')}"
        ) from error
    completed = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
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
