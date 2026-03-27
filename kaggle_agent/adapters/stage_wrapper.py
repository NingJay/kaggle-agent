from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kaggle_agent.adapters.command import ADAPTER_UNAVAILABLE_EXIT_CODE
from kaggle_agent.adapters.providers import ProviderResponse, ProviderUnavailable
from kaggle_agent.adapters.providers.amp_probe import AmpProbeResult, run_amp_probe
from kaggle_agent.adapters.providers.claude_headless import run_claude_headless
from kaggle_agent.adapters.providers.codex_exec import run_codex_exec
from kaggle_agent.adapters.schema_validation import SchemaValidationError, validate_payload
from kaggle_agent.utils import atomic_write_json, atomic_write_text, ensure_directory, now_utc_iso


ROOT_DOCS = ["AGENTS.md", "COMPETITION.md", "PLAYBOOK.md"]


@dataclass
class StageContext:
    stage: str
    workspace_root: Path
    input_manifest_path: Path
    output_dir: Path
    prompt_path: Path | None
    schema_path: Path
    input_manifest: dict[str, Any]

    @classmethod
    def from_env(cls) -> "StageContext":
        stage = _require_env("KAGGLE_AGENT_STAGE")
        workspace_root = Path(_require_env("KAGGLE_AGENT_WORKSPACE_ROOT")).resolve()
        input_manifest_path = Path(_require_env("KAGGLE_AGENT_INPUT_MANIFEST")).resolve()
        output_dir = Path(_require_env("KAGGLE_AGENT_OUTPUT_DIR")).resolve()
        prompt_file = _optional_env("KAGGLE_AGENT_PROMPT_FILE")
        prompt_path = Path(prompt_file).resolve() if prompt_file else None
        schema_path = (workspace_root / "schemas" / f"{stage}.schema.json").resolve()
        if not schema_path.exists():
            raise RuntimeError(f"Missing schema for stage {stage}: {schema_path}")
        if not input_manifest_path.exists():
            raise RuntimeError(f"Missing input manifest: {input_manifest_path}")
        payload = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Input manifest must be a JSON object")
        return cls(
            stage=stage,
            workspace_root=workspace_root,
            input_manifest_path=input_manifest_path,
            output_dir=ensure_directory(output_dir),
            prompt_path=prompt_path,
            schema_path=schema_path,
            input_manifest=payload,
        )


@dataclass
class CodegenWorkspace:
    snapshot_root: Path
    workspace_root: Path
    base_commit: str
    expected_config_relpath: str


def _require_env(name: str) -> str:
    value = _optional_env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_env(name: str) -> str:
    return os.environ.get(name, "")


def _read_optional(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _schema(ctx: StageContext) -> dict[str, Any]:
    payload = json.loads(ctx.schema_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Schema must be a JSON object: {ctx.schema_path}")
    return payload


def _doc_block(path: Path) -> str:
    text = _read_optional(path)
    if not text:
        return ""
    return f"## {path.name}\n\n{text}"


def _build_prompt(ctx: StageContext, codegen_workspace: CodegenWorkspace | None = None) -> str:
    prompt_sections = [f"# Stage\n\nYou are the `{ctx.stage}` stage adapter for Kaggle Agent."]
    if codegen_workspace is None:
        prompt_sections.append(
            "Return a single JSON object that matches the supplied schema exactly. Put the human-readable stage narrative in the `markdown` field and do not wrap anything in code fences."
        )
    else:
        prompt_sections.append(
            "Edit files directly inside the isolated workspace and finish with a short plain-text summary. Do not return patch text, YAML blobs, or JSON artifacts in the final message."
        )
    program = _read_optional(ctx.prompt_path)
    if program:
        prompt_sections.append(f"# Stage Program\n\n{program}")
    doc_sections = [_doc_block(ctx.workspace_root / name) for name in ROOT_DOCS]
    docs = "\n\n".join(section for section in doc_sections if section)
    if docs:
        prompt_sections.append(f"# Operating Contract\n\n{docs}")
    prompt_sections.append(
        "# Input Manifest\n\n```json\n"
        + json.dumps(ctx.input_manifest, indent=2, ensure_ascii=False)
        + "\n```"
    )
    if codegen_workspace is not None:
        prompt_sections.append(
            "# Editable Workspace\n\n"
            f"- Workspace root: `{codegen_workspace.workspace_root}`\n"
            "- Allowed edits: `train_sed.py`, `BirdCLEF-2026-Codebase/**`\n"
            "- Do not modify `state/`, `artifacts/`, or the original workspace.\n"
        )
        if codegen_workspace.expected_config_relpath:
            prompt_sections.append(
                "# Codegen Rules\n\n"
                f"Make sure the runnable config exists at `{codegen_workspace.expected_config_relpath}` inside the isolated workspace. Update that file in place or replace it with an improved generated config."
            )
        else:
            prompt_sections.append(
                "# Codegen Rules\n\nCreate or update a runnable YAML config under `BirdCLEF-2026-Codebase/configs/generated/` inside the isolated workspace."
            )
    if ctx.stage == "plan":
        prompt_sections.append(
            "# Plan Rules\n\nKeep execution config-path-oriented for now. Use empty strings or empty arrays for fields that do not apply."
        )
    return "\n\n".join(section.strip() for section in prompt_sections if section.strip()) + "\n"


def _write_raw_capture(path: Path, text: str) -> str:
    atomic_write_text(path, text)
    return str(path)


def _run_git(args: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise ProviderUnavailable("git is required to prepare the isolated codegen workspace") from error
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "git command failed").strip())
    return completed.stdout.strip()


def _prepare_codegen_workspace(ctx: StageContext) -> CodegenWorkspace:
    snapshot_root = ctx.workspace_root / "state" / "snapshots" / "codegen" / ctx.output_dir.name
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    workspace_root = ensure_directory(snapshot_root / "workspace")

    for name in [*ROOT_DOCS, "train_sed.py", "workspace.toml"]:
        source = ctx.workspace_root / name
        if source.exists():
            shutil.copy2(source, workspace_root / name)
    runtime_source = ctx.workspace_root / "BirdCLEF-2026-Codebase"
    if runtime_source.exists():
        shutil.copytree(runtime_source, workspace_root / runtime_source.name)

    _run_git(["init", "-q"], workspace_root)
    _run_git(["config", "user.email", "kaggle-agent@local"], workspace_root)
    _run_git(["config", "user.name", "kaggle-agent"], workspace_root)
    _run_git(["add", "."], workspace_root)
    _run_git(["commit", "-q", "-m", "Baseline codegen snapshot"], workspace_root)
    expected_config_relpath = str(ctx.input_manifest.get("plan", {}).get("config_path", "") or "")
    return CodegenWorkspace(
        snapshot_root=snapshot_root,
        workspace_root=workspace_root,
        base_commit=_run_git(["rev-parse", "HEAD"], workspace_root),
        expected_config_relpath=expected_config_relpath,
    )


def _resolve_generated_config_source(codegen_workspace: CodegenWorkspace) -> Path | None:
    if codegen_workspace.expected_config_relpath:
        expected = codegen_workspace.workspace_root / codegen_workspace.expected_config_relpath
        if expected.exists():
            return expected
    generated_root = codegen_workspace.workspace_root / "BirdCLEF-2026-Codebase" / "configs" / "generated"
    candidates = sorted(
        generated_root.glob("*.yaml"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ) if generated_root.exists() else []
    return candidates[0] if candidates else None


def _allow_codegen_paths(changed_files: list[str]) -> None:
    for relpath in changed_files:
        if relpath == "train_sed.py":
            continue
        if relpath.startswith("BirdCLEF-2026-Codebase/"):
            continue
        raise RuntimeError(f"codegen modified a disallowed path: {relpath}")


def _smoke_check(codegen_workspace: CodegenWorkspace) -> tuple[str, str]:
    train_entrypoint = codegen_workspace.workspace_root / "train_sed.py"
    if not train_entrypoint.exists():
        return "skipped", "train_sed.py not present in snapshot workspace."
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(train_entrypoint)],
        cwd=codegen_workspace.workspace_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return "passed", "py_compile succeeded for train_sed.py."
    return "failed", (completed.stderr or completed.stdout or "py_compile failed").strip()


def _codegen_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Codegen",
        "",
        f"- Status: `{payload['status']}`",
        f"- Reason: {payload['reason']}",
        f"- Generated config: `{payload['generated_config_path'] or 'n/a'}`",
        f"- Patch: `{payload['patch_path'] or 'n/a'}`",
        f"- Code state: `{payload['code_state_ref'] or 'n/a'}`",
        f"- Smoke: `{payload['smoke_status']}` {payload['smoke_summary']}",
    ]
    if payload["changed_files"]:
        lines.append(f"- Changed files: {', '.join(payload['changed_files'])}")
    return "\n".join(lines) + "\n"


def _materialize_codegen(
    ctx: StageContext,
    codegen_workspace: CodegenWorkspace,
) -> dict[str, Any]:
    status_lines = [line for line in _run_git(["status", "--porcelain"], codegen_workspace.workspace_root).splitlines() if line.strip()]
    changed_files: list[str] = []
    for line in status_lines:
        relpath = line[3:] if len(line) > 3 and line[2] == " " else line[2:]
        relpath = relpath.strip()
        if " -> " in relpath:
            relpath = relpath.split(" -> ", maxsplit=1)[1].strip()
        changed_files.append(relpath)
    canonical = {
        "stage": "codegen",
        "status": "noop",
        "reason": "Codex completed without modifying the isolated workspace.",
        "generated_config_path": "",
        "run_bundle_path": "",
        "patch_path": "",
        "code_state_ref": "",
        "worktree_path": "",
        "base_commit": codegen_workspace.base_commit,
        "head_commit": "",
        "changed_files": [],
        "smoke_status": "skipped",
        "smoke_summary": "No code changes were materialized.",
    }
    if not changed_files:
        return canonical

    _allow_codegen_paths(changed_files)
    config_source = _resolve_generated_config_source(codegen_workspace)
    if config_source is None:
        raise RuntimeError("codegen did not leave a runnable config in the isolated workspace")

    smoke_status, smoke_summary = _smoke_check(codegen_workspace)
    _run_git(["add", "."], codegen_workspace.workspace_root)
    _run_git(["commit", "-q", "-m", "Materialized codegen edits"], codegen_workspace.workspace_root)
    head_commit = _run_git(["rev-parse", "HEAD"], codegen_workspace.workspace_root)
    patch_text = _run_git(["diff", "--binary", f"{codegen_workspace.base_commit}..{head_commit}"], codegen_workspace.workspace_root)

    generated_config_path = ctx.output_dir / "generated_config.yaml"
    patch_path = ctx.output_dir / "patch.diff"
    run_bundle_path = ctx.output_dir / "run_bundle.json"
    atomic_write_text(generated_config_path, config_source.read_text(encoding="utf-8").rstrip() + "\n")
    atomic_write_text(patch_path, patch_text.rstrip() + ("\n" if patch_text else ""))
    run_bundle = {
        "spec_type": "experiment",
        "title": str(ctx.input_manifest.get("plan", {}).get("title", "")),
        "family": str(ctx.input_manifest.get("plan", {}).get("family", "")),
        "config_path": str(config_source.relative_to(codegen_workspace.workspace_root)),
        "launch_mode": str(ctx.input_manifest.get("plan", {}).get("launch_mode", "background")),
        "dedupe_key": str(ctx.input_manifest.get("plan", {}).get("dedupe_key", "")),
        "code_state_ref": str(codegen_workspace.workspace_root),
        "changed_files": changed_files,
        "smoke_status": smoke_status,
    }
    atomic_write_json(run_bundle_path, run_bundle)
    canonical.update(
        {
            "status": "generated",
            "reason": "Materialized Codex edits from the isolated snapshot workspace.",
            "generated_config_path": str(generated_config_path),
            "run_bundle_path": str(run_bundle_path),
            "patch_path": str(patch_path),
            "code_state_ref": str(codegen_workspace.workspace_root),
            "worktree_path": str(codegen_workspace.workspace_root),
            "head_commit": head_commit,
            "changed_files": changed_files,
            "smoke_status": smoke_status,
            "smoke_summary": smoke_summary,
        }
    )
    return canonical


def _materialize_stage_payload(
    ctx: StageContext,
    payload: dict[str, Any],
    codegen_workspace: CodegenWorkspace | None = None,
) -> tuple[dict[str, Any], str, str]:
    markdown = str(payload.get("markdown", "")).strip()
    if codegen_workspace is not None:
        canonical = _materialize_codegen(ctx, codegen_workspace)
        return canonical, _codegen_markdown(canonical), ""
    if not markdown:
        raise RuntimeError("provider output did not include markdown")
    payload = dict(payload)
    payload.pop("markdown", None)
    spec_path = ""
    if ctx.stage == "plan" and str(payload.get("plan_status", "")) == "planned":
        spec_file = ctx.output_dir / "spec.yaml"
        atomic_write_text(spec_file, yaml.safe_dump(payload, sort_keys=False))
        spec_path = str(spec_file)
    return payload, markdown + "\n", spec_path


def _provider_meta(
    *,
    ctx: StageContext,
    response: ProviderResponse,
    stdout_path: str,
    stderr_path: str,
    event_log_path: str,
    amp_probe_path: str = "",
    amp_probe_summary: str = "",
    amp_thread_id: str = "",
    codegen_workspace: CodegenWorkspace | None = None,
) -> dict[str, Any]:
    meta = {
        "provider": response.provider,
        "model": response.model,
        "schema_path": str(ctx.schema_path),
        "raw_stdout_path": stdout_path,
        "raw_stderr_path": stderr_path,
        "raw_event_log_path": event_log_path,
        "session_id": response.session_id,
        "thread_id": response.thread_id,
        "exit_code": response.exit_code,
        "started_at": response.extra_meta.get("started_at", ""),
        "completed_at": response.extra_meta.get("completed_at", ""),
        "materialization_mode": response.extra_meta.get("materialization_mode", "structured"),
    }
    if codegen_workspace is not None:
        meta["workspace_root_used"] = str(codegen_workspace.workspace_root)
        meta["snapshot_root"] = str(codegen_workspace.snapshot_root)
    if amp_probe_path:
        meta["amp_probe_path"] = amp_probe_path
        meta["amp_probe_summary"] = amp_probe_summary
        meta["amp_thread_id"] = amp_thread_id
    return meta


def _run_provider(
    provider: str,
    ctx: StageContext,
    prompt: str,
    codegen_workspace: CodegenWorkspace | None = None,
) -> tuple[ProviderResponse, AmpProbeResult | None]:
    if provider == "claude":
        return run_claude_headless(prompt=prompt, schema=_schema(ctx), workspace_root=ctx.workspace_root), None
    if provider == "codex":
        return run_codex_exec(
            prompt=prompt,
            schema_path=ctx.schema_path if codegen_workspace is None else None,
            workspace_root=ctx.workspace_root if codegen_workspace is None else codegen_workspace.workspace_root,
            output_dir=ctx.output_dir,
            mode="structured" if codegen_workspace is None else "agentic",
        ), None
    if provider == "critic":
        response = run_claude_headless(prompt=prompt, schema=_schema(ctx), workspace_root=ctx.workspace_root)
        amp_prompt = (
            "Review the following critic context and return a concise diagnostic summary. "
            "Do not modify files.\n\n"
            f"```json\n{json.dumps(ctx.input_manifest, indent=2, ensure_ascii=False)}\n```"
        )
        return response, run_amp_probe(prompt=amp_prompt, workspace_root=ctx.workspace_root)
    raise RuntimeError(f"Unknown provider: {provider}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless stage wrapper for Claude/Codex/Amp adapters.")
    parser.add_argument("--provider", required=True, choices=["claude", "codex", "critic"])
    args = parser.parse_args(argv)

    try:
        ctx = StageContext.from_env()
        codegen_workspace = _prepare_codegen_workspace(ctx) if ctx.stage == "codegen" else None
        prompt = _build_prompt(ctx, codegen_workspace)
        started_at = now_utc_iso()
        response, amp_probe = _run_provider(args.provider, ctx, prompt, codegen_workspace)
        response.extra_meta.setdefault("started_at", started_at)
        response.extra_meta.setdefault("completed_at", now_utc_iso())
        if ctx.stage != "codegen":
            validate_payload(_schema(ctx), response.payload)
        payload, markdown, spec_path = _materialize_stage_payload(ctx, response.payload, codegen_workspace)
        if ctx.stage == "codegen":
            validate_payload(_schema(ctx), payload)

        json_path = ctx.output_dir / f"{ctx.stage}.json"
        md_path = ctx.output_dir / f"{ctx.stage}.md"
        stdout_path = _write_raw_capture(ctx.output_dir / "raw_stdout.txt", response.raw_stdout)
        stderr_path = _write_raw_capture(ctx.output_dir / "raw_stderr.txt", response.raw_stderr)
        event_log_path = ""
        if response.event_log_text:
            event_log_path = _write_raw_capture(ctx.output_dir / "events.jsonl", response.event_log_text)
        if amp_probe is not None:
            amp_probe_path = _write_raw_capture(ctx.output_dir / "amp_probe.jsonl", amp_probe.event_log_text)
            payload["amp_probe_summary"] = amp_probe.summary
        else:
            amp_probe_path = ""

        atomic_write_json(json_path, payload)
        atomic_write_text(md_path, markdown)
        meta = _provider_meta(
            ctx=ctx,
            response=response,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            event_log_path=event_log_path,
            amp_probe_path=amp_probe_path,
            amp_probe_summary=amp_probe.summary if amp_probe is not None else "",
            amp_thread_id=amp_probe.thread_id if amp_probe is not None else "",
            codegen_workspace=codegen_workspace,
        )
        if spec_path:
            meta["spec_path"] = spec_path
        atomic_write_json(ctx.output_dir / "provider_meta.json", meta)
        return 0
    except ProviderUnavailable as error:
        print(str(error), file=sys.stderr)
        return ADAPTER_UNAVAILABLE_EXIT_CODE
    except SchemaValidationError as error:
        print(f"schema validation failed: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
