from __future__ import annotations

import argparse
import json
import os
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


def _build_prompt(ctx: StageContext) -> str:
    prompt_sections = [
        f"# Stage\n\nYou are the `{ctx.stage}` stage adapter for Kaggle Agent.",
        "Return a single JSON object that matches the supplied schema exactly. Put the human-readable stage narrative in the `markdown` field and do not wrap anything in code fences.",
    ]
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
    if ctx.stage == "codegen":
        prompt_sections.append(
            "# Codegen Rules\n\n"
            "Do not assume write access. Emit `generated_config_yaml`, `patch_diff`, and `run_bundle` content inside JSON. The wrapper will materialize files."
        )
    if ctx.stage == "plan":
        prompt_sections.append(
            "# Plan Rules\n\nKeep execution config-path-oriented for now. Use empty strings or empty arrays for fields that do not apply."
        )
    return "\n\n".join(section.strip() for section in prompt_sections if section.strip()) + "\n"


def _write_raw_capture(path: Path, text: str) -> str:
    atomic_write_text(path, text)
    return str(path)


def _materialize_codegen(ctx: StageContext, payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", "noop"))
    reason = str(payload.get("reason", ""))
    canonical = {
        "stage": "codegen",
        "status": status,
        "reason": reason,
        "generated_config_path": "",
        "run_bundle_path": "",
        "patch_path": "",
    }
    if status != "generated":
        return canonical

    generated_config = str(payload.get("generated_config_yaml", ""))
    patch_diff = str(payload.get("patch_diff", ""))
    run_bundle = payload.get("run_bundle", {})
    if not generated_config.strip():
        raise RuntimeError("codegen returned status=generated without generated_config_yaml")
    if not isinstance(run_bundle, dict):
        raise RuntimeError("codegen run_bundle must be a JSON object")

    generated_config_path = ctx.output_dir / "generated_config.yaml"
    patch_path = ctx.output_dir / "patch.diff"
    run_bundle_path = ctx.output_dir / "run_bundle.json"
    atomic_write_text(generated_config_path, generated_config.rstrip() + "\n")
    atomic_write_text(patch_path, patch_diff)
    atomic_write_json(run_bundle_path, run_bundle)
    canonical["generated_config_path"] = str(generated_config_path)
    canonical["run_bundle_path"] = str(run_bundle_path)
    canonical["patch_path"] = str(patch_path)
    return canonical


def _materialize_stage_payload(ctx: StageContext, payload: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    markdown = str(payload.get("markdown", "")).strip()
    if not markdown:
        raise RuntimeError("provider output did not include markdown")
    payload = dict(payload)
    payload.pop("markdown", None)
    spec_path = ""
    if ctx.stage == "codegen":
        return _materialize_codegen(ctx, payload), markdown + "\n", spec_path
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
    }
    if amp_probe_path:
        meta["amp_probe_path"] = amp_probe_path
        meta["amp_probe_summary"] = amp_probe_summary
        meta["amp_thread_id"] = amp_thread_id
    return meta


def _run_provider(provider: str, ctx: StageContext, prompt: str) -> tuple[ProviderResponse, AmpProbeResult | None]:
    if provider == "claude":
        return run_claude_headless(prompt=prompt, schema=_schema(ctx), workspace_root=ctx.workspace_root), None
    if provider == "codex":
        return run_codex_exec(
            prompt=prompt,
            schema_path=ctx.schema_path,
            workspace_root=ctx.workspace_root,
            output_dir=ctx.output_dir,
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
        prompt = _build_prompt(ctx)
        started_at = now_utc_iso()
        response, amp_probe = _run_provider(args.provider, ctx, prompt)
        response.extra_meta.setdefault("started_at", started_at)
        response.extra_meta.setdefault("completed_at", now_utc_iso())
        validate_payload(_schema(ctx), response.payload)
        payload, markdown, spec_path = _materialize_stage_payload(ctx, response.payload)

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
