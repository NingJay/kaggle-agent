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
CODEX_PROFILE = "kaggle-agent"


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
) -> ProviderResponse:
    binary = shutil.which(CODEX_BINARY)
    if not binary:
        raise ProviderUnavailable("codex binary is not available on PATH")

    codex_api_key = os.environ.get("CODEX_API_KEY", "").strip()
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    response_path = output_dir / "provider_response.json"
    isolated_home = output_dir / "codex-home"
    codex_home = isolated_home / ".codex"
    codex_home.mkdir(parents=True, exist_ok=True)
    config_path = codex_home / "config.toml"
    if not config_path.exists():
        config_path.write_text(f"[profiles.{CODEX_PROFILE}]\n", encoding="utf-8")
    env = os.environ.copy()
    if not codex_api_key and openai_api_key:
        env["CODEX_API_KEY"] = openai_api_key
    env["HOME"] = str(isolated_home)
    env["CODEX_HOME"] = str(codex_home)
    env["XDG_CONFIG_HOME"] = str(isolated_home / ".config")

    if mode == "structured":
        if schema_path is None:
            raise RuntimeError("structured codex mode requires schema_path")
        args = [
            binary,
            "exec",
            "--profile",
            CODEX_PROFILE,
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
            "--profile",
            CODEX_PROFILE,
            "--skip-git-repo-check",
            "--full-auto",
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
            "provider_runtime": f"codex/profile:{CODEX_PROFILE} mode:{mode}",
            "codex_profile": CODEX_PROFILE,
            "isolated_home": str(isolated_home),
            "codex_home": str(codex_home),
        },
    )
