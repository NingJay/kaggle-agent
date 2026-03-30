from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from kaggle_agent.adapters.command import parse_json_payload
from kaggle_agent.adapters.providers import ProviderResponse, ProviderUnavailable


CODEX_BINARY = "codex"
CODEX_PROFILE_ENV = "KAGGLE_AGENT_CODEX_PROFILE"


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

    completed = subprocess.run(
        args,
        input=prompt,
        capture_output=True,
        text=True,
        cwd=workspace_root,
        env=env,
        check=False,
    )
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
