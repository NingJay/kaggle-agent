from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from kaggle_agent.adapters.command import parse_json_payload
from kaggle_agent.adapters.providers import ProviderResponse, ProviderUnavailable
from kaggle_agent.adapters.providers.claude_runtime import claude_subprocess_env


CLAUDE_BINARY = "claude"
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


@lru_cache(maxsize=1)
def _help_text() -> str:
    binary = shutil.which(CLAUDE_BINARY)
    if not binary:
        return ""
    completed = subprocess.run(
        [binary, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (completed.stdout or "") + (completed.stderr or "")


def _supports_flag(flag: str) -> bool:
    return flag in _help_text()


def _schema_json(schema_path: Path) -> str:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _model_name() -> str:
    return (
        os.environ.get("KAGGLE_AGENT_CLAUDE_CODE_MODEL", "").strip()
        or os.environ.get("KAGGLE_AGENT_CLAUDE_MODEL", "").strip()
    )


def _effort_name() -> str:
    return (
        os.environ.get("KAGGLE_AGENT_CLAUDE_CODE_EFFORT", "").strip()
        or os.environ.get("KAGGLE_AGENT_CLAUDE_EFFORT", "").strip()
    )


def _structured_payload(envelope: dict[str, Any]) -> dict[str, Any]:
    payload = envelope.get("structured_output")
    if isinstance(payload, dict):
        return payload
    result = envelope.get("result")
    if isinstance(result, str) and result.strip():
        try:
            parsed = parse_json_payload(result)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return envelope


def run_claude_code_exec(
    *,
    prompt: str,
    schema_path: Path | None,
    workspace_root: Path,
    mode: str = "structured",
    extra_env: dict[str, str] | None = None,
) -> ProviderResponse:
    binary = shutil.which(CLAUDE_BINARY)
    if not binary:
        raise ProviderUnavailable("claude binary is not available on PATH")

    args = [
        binary,
        "-p",
        "--output-format",
        "json",
    ]
    if _supports_flag("--no-session-persistence"):
        args.append("--no-session-persistence")
    if _supports_flag("--disable-slash-commands"):
        args.append("--disable-slash-commands")
    if _supports_flag("--no-chrome"):
        args.append("--no-chrome")

    if mode == "structured":
        if schema_path is None:
            raise RuntimeError("structured claude_code mode requires schema_path")
        args.extend(
            [
                "--json-schema",
                _schema_json(schema_path),
                "--tools",
                "",
                "--append-system-prompt",
                "Return only structured JSON matching the schema. Use only the supplied context. Do not call tools. If the CLI wraps the answer in a result envelope, make the result body a single JSON object.",
            ]
        )
        if _supports_flag("--bare"):
            args.append("--bare")
    elif mode == "agentic":
        args.extend(
            [
                "--dangerously-skip-permissions",
                "--permission-mode",
                "bypassPermissions",
                "--append-system-prompt",
                "Work only inside the current isolated stage workspace. Do not rely on files outside the workspace or on user-level MCP servers, plugins, or skills.",
            ]
        )
        if _supports_flag("--add-dir"):
            args.extend(["--add-dir", str(workspace_root)])
    else:
        raise RuntimeError(f"Unsupported claude_code mode: {mode}")

    model = _model_name()
    if model:
        args.extend(["--model", model])
    effort = _effort_name()
    if effort:
        args.extend(["--effort", effort])

    timeout_seconds = _timeout_seconds(mode)
    with claude_subprocess_env(isolate_home_env_var="KAGGLE_AGENT_CLAUDE_CODE_ISOLATE_HOME") as env:
        if extra_env:
            env.update(extra_env)
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
                return ProviderResponse(
                    provider="claude_code",
                    model=model,
                    payload={},
                    raw_stdout=stdout,
                    raw_stderr=(stderr + f"\nclaude code {mode} timed out after {timeout_label}").strip(),
                    event_log_text=stdout,
                    exit_code=124,
                    extra_meta={
                        "materialization_mode": mode,
                        "provider_runtime": f"claude_code mode:{mode}",
                        "isolated_home": os.environ.get("KAGGLE_AGENT_CLAUDE_CODE_ISOLATE_HOME", "1"),
                        "timed_out": "1",
                        "timeout_seconds": timeout_seconds or DEFAULT_AGENTIC_TIMEOUT_SECONDS,
                    },
                )
            raise RuntimeError(f"claude code {mode} timed out after {timeout_label}") from error
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "claude invocation failed").strip()
        raise RuntimeError(stderr)

    envelope: dict[str, Any] = {}
    if completed.stdout.strip():
        try:
            parsed = parse_json_payload(completed.stdout)
        except json.JSONDecodeError as error:
            if mode == "structured":
                snippet = (completed.stdout or completed.stderr or "").strip()[:400] or "<empty>"
                raise RuntimeError(f"claude code returned non-JSON output: {snippet}") from error
            parsed = {}
        if isinstance(parsed, dict):
            envelope = parsed

    if mode == "structured":
        payload = _structured_payload(envelope)
        if not isinstance(payload, dict):
            raise RuntimeError("claude code did not return a structured object")
    else:
        payload = {}

    session_id = _first_string(envelope, "session_id")

    return ProviderResponse(
        provider="claude_code",
        model=str(envelope.get("model", model)),
        session_id=session_id,
        thread_id=session_id,
        payload=payload,
        raw_stdout=completed.stdout,
        raw_stderr=completed.stderr,
        event_log_text=completed.stdout,
        exit_code=completed.returncode,
        extra_meta={
            "materialization_mode": mode,
            "provider_runtime": f"claude_code mode:{mode}",
            "isolated_home": os.environ.get("KAGGLE_AGENT_CLAUDE_CODE_ISOLATE_HOME", "1"),
            "permission_mode": "bypassPermissions" if mode == "agentic" else "",
        },
    )
