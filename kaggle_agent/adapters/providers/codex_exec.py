from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from kaggle_agent.adapters.command import parse_json_payload
from kaggle_agent.adapters.providers import ProviderResponse, ProviderUnavailable


CODEX_BINARY = "codex"
CODEX_PROFILE_ENV = "KAGGLE_AGENT_CODEX_PROFILE"
DEFAULT_AGENTIC_TIMEOUT_SECONDS = 300


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


def _run_process(
    args: list[str],
    *,
    prompt: str,
    workspace_root: Path,
    env: dict[str, str],
    timeout_seconds: int | None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace_root,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _terminate_process_group(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout = (error.stdout or "") if isinstance(error.stdout, str) else ""
            stderr = (error.stderr or "") if isinstance(error.stderr, str) else ""
            stderr = (stderr + "\nprovider subprocess did not drain after timeout").strip()
        raise subprocess.TimeoutExpired(process.args, timeout_seconds or 0, output=stdout, stderr=stderr) from error
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _timeout_seconds(mode: str) -> int | None:
    raw = os.environ.get("KAGGLE_AGENT_PROVIDER_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_AGENTIC_TIMEOUT_SECONDS if mode == "agentic" else None
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_AGENTIC_TIMEOUT_SECONDS if mode == "agentic" else None
    return value if value > 0 else None


def _first_string(obj: Any, key: str) -> str:
    if isinstance(obj, dict):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
        for item in obj.values():
            found = _first_string(item, key)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _first_string(item, key)
            if found:
                return found
    return ""


def run_codex_exec(
    *,
    prompt: str,
    schema_path: Path | None,
    workspace_root: Path,
    output_dir: Path,
    mode: str = "structured",
    extra_env: dict[str, str] | None = None,
) -> ProviderResponse:
    binary = shutil.which(CODEX_BINARY)
    if not binary:
        raise ProviderUnavailable("codex binary is not available on PATH")

    codex_api_key = os.environ.get("CODEX_API_KEY", "").strip()
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    profile = os.environ.get(CODEX_PROFILE_ENV, "").strip()
    response_path = output_dir / "provider_response.json"
    env = os.environ.copy()
    if not codex_api_key and openai_api_key:
        env["CODEX_API_KEY"] = openai_api_key
    if extra_env:
        env.update(extra_env)

    if mode == "structured":
        if schema_path is None:
            raise RuntimeError("structured codex mode requires schema_path")
        args = [
            binary,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(response_path),
            "--json",
            "--ephemeral",
            "-C",
            str(workspace_root),
            "-",
        ]
    elif mode == "agentic":
        args = [
            binary,
            "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--json",
            "--ephemeral",
            "-C",
            str(workspace_root),
            "-",
        ]
    else:
        raise RuntimeError(f"Unsupported codex mode: {mode}")
    model = os.environ.get("KAGGLE_AGENT_CODEX_MODEL", "").strip()
    if model:
        args[2:2] = ["--model", model]
    if profile:
        args[2:2] = ["--profile", profile]

    timeout_seconds = _timeout_seconds(mode)
    try:
        completed = _run_process(
            args,
            prompt=prompt,
            workspace_root=workspace_root,
            env=env,
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        timeout_label = f"{timeout_seconds}s" if timeout_seconds is not None else "the configured timeout"
        stdout = (error.stdout or error.output or "") if isinstance(getattr(error, "stdout", None) or error.output, str) else ""
        stderr = (error.stderr or "") if isinstance(error.stderr, str) else ""
        if mode == "agentic":
            runtime_bits = ["codex", f"mode:{mode}", "env:inherit"]
            if profile:
                runtime_bits.append(f"profile:{profile}")
            return ProviderResponse(
                provider="codex",
                model=model,
                payload={},
                raw_stdout=stdout,
                raw_stderr=(stderr + f"\ncodex {mode} timed out after {timeout_label}").strip(),
                event_log_text=stdout,
                exit_code=124,
                extra_meta={
                    "materialization_mode": mode,
                    "provider_runtime": " ".join(runtime_bits),
                    "codex_profile": profile,
                    "timed_out": "1",
                    "timeout_seconds": timeout_seconds or DEFAULT_AGENTIC_TIMEOUT_SECONDS,
                },
            )
        raise RuntimeError(f"codex {mode} timed out after {timeout_label}") from error
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "codex invocation failed").strip()
        raise RuntimeError(stderr)
    if mode == "structured":
        if not response_path.exists():
            raise RuntimeError("codex did not write provider_response.json")
        payload = parse_json_payload(response_path)
        if not isinstance(payload, dict):
            raise RuntimeError("codex did not return a structured object")
    else:
        payload = {}

    session_id = ""
    thread_id = ""
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not session_id:
            session_id = _first_string(event, "session_id")
        if not thread_id:
            thread_id = _first_string(event, "thread_id")
    if not session_id and thread_id:
        session_id = thread_id

    runtime_bits = ["codex", f"mode:{mode}", "env:inherit"]
    if profile:
        runtime_bits.append(f"profile:{profile}")

    return ProviderResponse(
        provider="codex",
        model=model,
        session_id=session_id,
        thread_id=thread_id,
        payload=payload,
        raw_stdout=completed.stdout,
        raw_stderr=completed.stderr,
        event_log_text=completed.stdout,
        exit_code=completed.returncode,
        extra_meta={
            "materialization_mode": mode,
            "provider_runtime": " ".join(runtime_bits),
            "codex_profile": profile,
        },
    )
