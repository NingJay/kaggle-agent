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
from kaggle_agent.adapters.providers.claude_code_exec import run_claude_code_exec
from kaggle_agent.adapters.providers.claude_headless import run_claude_headless
from kaggle_agent.adapters.providers.codex_exec import run_codex_exec
from kaggle_agent.adapters.schema_validation import SchemaValidationError, validate_payload
from kaggle_agent.utils import atomic_write_json, atomic_write_text, ensure_directory, now_utc_iso, truncate


ROOT_DOCS = ["AGENTS.md", "COMPETITION.md", "PLAYBOOK.md"]
KNOWLEDGE_FILES_IN_PROMPT_ORDER = [
    "00_experiment_rules.md",
    "01_validated_findings.md",
    "03_next_experiment_priors.md",
    "04_submission_bar.md",
    "experiment_conclusions.md",
]
KNOWLEDGE_PROMPT_STAGES = {"research", "decision", "plan", "codegen", "critic"}
KNOWLEDGE_PROMPT_CHAR_BUDGET = 16000
STAGE_WORKSPACE_TOP_LEVEL_EXCLUDES = {
    ".git",
    ".venv",
    "artifacts",
    "state",
    "__pycache__",
}
CODEGEN_ALLOWED_EDIT_ROOTS = [
    "train_sed.py",
    "BirdCLEF-2026-Codebase/configs",
    "BirdCLEF-2026-Codebase/src",
    "BirdCLEF-2026-Codebase/train.py",
    "BirdCLEF-2026-Codebase/inference.py",
    "BirdCLEF-2026-Codebase/scripts",
]
CODEGEN_DENIED_PREFIX_REASONS = {
    "BirdCLEF-2026-Codebase/outputs/": "outputs",
    "BirdCLEF-2026-Codebase/models/": "model",
    "BirdCLEF-2026-Codebase/birdclef-2026/": "data",
    "artifacts/": "artifact",
    "state/": "state",
}
CODEGEN_DENIED_SUFFIX_REASONS = {
    ".npz": "artifact",
    ".pkl": "artifact",
    ".pt": "artifact",
    ".ckpt": "artifact",
    ".ipynb": "notebook",
}
CODEGEN_NOISE_SUFFIXES = {".pyc", ".pyo"}


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
    verify_root: Path
    base_commit: str
    expected_config_relpath: str
    workspace_mode: str = "snapshot-repo"


@dataclass
class StageWorkspace:
    stage_root: Path
    workspace_root: Path
    workspace_mode: str = "snapshot-repo"


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


def _knowledge_blocks(root: Path, manifest: dict[str, Any], *, limit: int = KNOWLEDGE_PROMPT_CHAR_BUDGET) -> str:
    knowledge_root = root / "knowledge"
    if not knowledge_root.exists():
        return ""

    candidates: list[Path] = []
    seen: set[Path] = set()
    run_id = str(manifest.get("run", {}).get("run_id", "") or "")
    ordered_relpaths = list(KNOWLEDGE_FILES_IN_PROMPT_ORDER)
    if run_id:
        ordered_relpaths.insert(4, f"research/{run_id}.md")
    for relative in ordered_relpaths:
        path = knowledge_root / relative
        if path.exists() and path.is_file():
            candidates.append(path)
            seen.add(path.resolve())

    for path in sorted(knowledge_root.glob("research/*.md"), reverse=True):
        resolved = path.resolve()
        if resolved not in seen:
            candidates.append(path)
            seen.add(resolved)
        if len(candidates) >= 8:
            break

    sections: list[str] = []
    total = 0
    for path in candidates:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        section = f"## knowledge/{path.relative_to(knowledge_root).as_posix()}\n\n{text}"
        remaining = limit - total
        if remaining <= 0:
            break
        if len(section) > remaining:
            section = truncate(section, limit=remaining)
        sections.append(section)
        total += len(section) + 2
    return "\n\n".join(section for section in sections if section)


def _build_prompt(
    ctx: StageContext,
    codegen_workspace: CodegenWorkspace | None = None,
    *,
    provider_workspace_root: Path | None = None,
) -> str:
    prompt_sections = [f"# Stage\n\nYou are the `{ctx.stage}` stage adapter for Kaggle Agent."]
    if codegen_workspace is None:
        prompt_sections.append(
            "Return a single JSON object that matches the supplied schema exactly. Put the human-readable stage narrative in the `markdown` field and do not wrap anything in code fences."
        )
        schema = _schema(ctx)
        required = [str(item) for item in schema.get("required", [])]
        properties = schema.get("properties", {})
        if required:
            lines = ["# Output Contract", "", "Required fields:"]
            for key in required:
                prop = properties.get(key, {})
                field_type = prop.get("type")
                if isinstance(field_type, list):
                    rendered_type = " | ".join(str(item) for item in field_type)
                elif field_type is None:
                    rendered_type = "any"
                else:
                    rendered_type = str(field_type)
                lines.append(f"- `{key}`: {rendered_type}")
            prompt_sections.append("\n".join(lines))
    else:
        prompt_sections.append(
            "Edit files directly inside the isolated workspace and finish with a short plain-text summary. Do not return patch text, YAML blobs, or JSON artifacts in the final message."
        )
    program = _read_optional(ctx.prompt_path)
    if program:
        prompt_sections.append(f"# Stage Program\n\n{program}")
    docs_root = provider_workspace_root or ctx.workspace_root
    doc_sections = [_doc_block(docs_root / name) for name in ROOT_DOCS]
    docs = "\n\n".join(section for section in doc_sections if section)
    if docs:
        prompt_sections.append(f"# Operating Contract\n\n{docs}")
    if ctx.stage in KNOWLEDGE_PROMPT_STAGES:
        knowledge = _knowledge_blocks(docs_root, ctx.input_manifest)
        if knowledge:
            prompt_sections.append(f"# Knowledge Context\n\n{knowledge}")
    prompt_sections.append(
        "# Input Manifest\n\n```json\n"
        + json.dumps(ctx.input_manifest, indent=2, ensure_ascii=False)
        + "\n```"
    )
    if codegen_workspace is not None:
        previous_attempt = ctx.input_manifest.get("previous_codegen_attempt")
        if isinstance(previous_attempt, dict):
            changed_files = previous_attempt.get("changed_files", [])
            changed_files_lines = ""
            if isinstance(changed_files, list) and changed_files:
                changed_files_lines = "\n".join(f"- `{str(item)}`" for item in changed_files)
            prompt_sections.append(
                "# Retry Context\n\n"
                f"This is codegen repair attempt `{ctx.input_manifest.get('codegen_attempt_number', '')}`.\n\n"
                f"- Previous attempt: `{previous_attempt.get('attempt_number', '')}`\n"
                f"- Previous status: `{previous_attempt.get('status', '')}`\n"
                f"- Previous verify status: `{previous_attempt.get('verify_status', '')}`\n"
                f"- Previous verify summary: {previous_attempt.get('verify_summary', '')}\n"
                + (f"- Previous changed files:\n{changed_files_lines}\n" if changed_files_lines else "")
                + "\n"
                "First repair the previous verify failure before attempting any new optimization.\n"
                "Prefer the smallest change that restores a passing verify run.\n"
                "If the previous attempt edited runtime source files and verify failed, revert or narrow those source edits unless they are strictly required."
            )
        prompt_sections.append(
            "# Editable Workspace\n\n"
            f"- Workspace root: `{codegen_workspace.workspace_root}`\n"
            f"- Verify artifacts root: `{codegen_workspace.verify_root}`\n"
            "- Allowed edits:\n"
            + "\n".join(f"  - `{item}`" for item in CODEGEN_ALLOWED_EDIT_ROOTS)
            + "\n"
            "- Never modify `BirdCLEF-2026-Codebase/outputs/`, `BirdCLEF-2026-Codebase/models/`, `BirdCLEF-2026-Codebase/birdclef-2026/`, `state/`, or `artifacts/`.\n"
            "- Never create notebooks or binary artifacts such as `.ipynb`, `.npz`, `.pkl`, `.pt`, or `.ckpt`.\n"
            "- Keep the source tree clean. The harness owns the final verify run and artifact export.\n"
        )
        if codegen_workspace.expected_config_relpath:
            prompt_sections.append(
                "# Codegen Rules\n\n"
                f"Make sure the runnable config exists at `{codegen_workspace.expected_config_relpath}` inside the isolated workspace. Update that file in place or replace it with an improved generated config.\n"
                "Do not return patch text, YAML blobs, or JSON manifests in the final message. Finish with a short plain-text summary of source edits only."
            )
        else:
            prompt_sections.append(
                "# Codegen Rules\n\n"
                "Create or update a runnable YAML config under `BirdCLEF-2026-Codebase/configs/generated/` inside the isolated workspace.\n"
                "Do not return patch text, YAML blobs, or JSON manifests in the final message. Finish with a short plain-text summary of source edits only."
            )
    if ctx.stage == "plan":
        prompt_sections.append(
            "# Plan Rules\n\n"
            "Keep execution config-path-oriented for now. Use empty strings or empty arrays for fields that do not apply.\n"
            "For ordinary experiment iteration, debug reruns, and code-fix follow-ups, return `plan_status` as `planned`.\n"
            "Use `submission_candidate` only for explicit submission-packaging or leaderboard-promotion plans.\n"
            "Use the knowledge context explicitly. When validation metrics are available, treat `val_soundscape_macro_roc_auc` as the keep/discard metric and treat resubstitution `soundscape_macro_roc_auc` as diagnostic only.\n"
            "If the blocking issue is class imbalance or class coverage, propose a plan that addresses coverage before calibration-only tuning."
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


def _copy_stage_source_tree(source_root: Path, destination_root: Path) -> None:
    for item in source_root.iterdir():
        if item.name in STAGE_WORKSPACE_TOP_LEVEL_EXCLUDES:
            continue
        if item.name.startswith(".git"):
            continue
        destination = destination_root / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache"),
            )
        else:
            shutil.copy2(item, destination)


def _sanitize_stage_workspace(workspace_root: Path) -> None:
    runtime_outputs = workspace_root / "BirdCLEF-2026-Codebase" / "outputs"
    if runtime_outputs.exists():
        shutil.rmtree(runtime_outputs, ignore_errors=True)


def _prepare_stage_workspace(ctx: StageContext) -> StageWorkspace:
    stage_root = ctx.workspace_root / "state" / "worktrees" / ctx.stage / ctx.output_dir.name
    if stage_root.exists():
        shutil.rmtree(stage_root)
    workspace_root = ensure_directory(stage_root / "workspace")
    _copy_stage_source_tree(ctx.workspace_root, workspace_root)
    _sanitize_stage_workspace(workspace_root)
    return StageWorkspace(stage_root=stage_root, workspace_root=workspace_root)


def _prepare_codegen_workspace(ctx: StageContext, stage_workspace: StageWorkspace | None = None) -> CodegenWorkspace:
    base_workspace = stage_workspace or _prepare_stage_workspace(ctx)
    snapshot_root = base_workspace.stage_root
    if snapshot_root.exists():
        ensure_directory(snapshot_root)
    workspace_root = ensure_directory(snapshot_root / "workspace")
    verify_root = ensure_directory(snapshot_root / "verify_runtime")

    _run_git(["init", "-q"], workspace_root)
    _run_git(["config", "user.email", "kaggle-agent@local"], workspace_root)
    _run_git(["config", "user.name", "kaggle-agent"], workspace_root)
    _run_git(["add", "."], workspace_root)
    _run_git(["commit", "-q", "-m", "Baseline codegen snapshot"], workspace_root)
    expected_config_relpath = str(ctx.input_manifest.get("plan", {}).get("config_path", "") or "")
    return CodegenWorkspace(
        snapshot_root=snapshot_root,
        workspace_root=workspace_root,
        verify_root=verify_root,
        base_commit=_run_git(["rev-parse", "HEAD"], workspace_root),
        expected_config_relpath=expected_config_relpath,
        workspace_mode=base_workspace.workspace_mode,
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
        normalized = relpath.replace("\\", "/").strip()
        for denied_prefix, reason in CODEGEN_DENIED_PREFIX_REASONS.items():
            if normalized.startswith(denied_prefix):
                raise RuntimeError(f"codegen modified a disallowed {reason} path: {normalized}")
        if "__pycache__" in Path(normalized).parts:
            raise RuntimeError(f"codegen modified a disallowed cache path: {normalized}")
        for denied_suffix, reason in CODEGEN_DENIED_SUFFIX_REASONS.items():
            if normalized.endswith(denied_suffix):
                raise RuntimeError(f"codegen modified a disallowed {reason} path: {normalized}")
        if normalized == "train_sed.py":
            continue
        if normalized in {
            "BirdCLEF-2026-Codebase/train.py",
            "BirdCLEF-2026-Codebase/inference.py",
        }:
            continue
        if normalized.startswith("BirdCLEF-2026-Codebase/configs/"):
            continue
        if normalized.startswith("BirdCLEF-2026-Codebase/src/"):
            continue
        if normalized.startswith("BirdCLEF-2026-Codebase/scripts/"):
            continue
        raise RuntimeError(f"codegen modified a disallowed path: {normalized}")

def _collect_worktree_changed_files(codegen_workspace: CodegenWorkspace) -> list[str]:
    status_lines = [line for line in _run_git(["status", "--porcelain"], codegen_workspace.workspace_root).splitlines() if line.strip()]
    changed_files: list[str] = []
    for line in status_lines:
        relpath = line[3:] if len(line) > 3 and line[2] == " " else line[2:]
        relpath = relpath.strip()
        if " -> " in relpath:
            relpath = relpath.split(" -> ", maxsplit=1)[1].strip()
        changed_files.append(relpath)
    return sorted(dict.fromkeys(changed_files))


def _head_commit(codegen_workspace: CodegenWorkspace) -> str:
    return _run_git(["rev-parse", "HEAD"], codegen_workspace.workspace_root)


def _collect_committed_changed_files(codegen_workspace: CodegenWorkspace, *, head_commit: str) -> list[str]:
    if head_commit == codegen_workspace.base_commit:
        return []
    output = _run_git(
        ["diff", "--name-only", f"{codegen_workspace.base_commit}..{head_commit}"],
        codegen_workspace.workspace_root,
    )
    changed_files = [line.strip() for line in output.splitlines() if line.strip()]
    return sorted(dict.fromkeys(changed_files))


def _collect_materialized_files(codegen_workspace: CodegenWorkspace) -> tuple[str, list[str], list[str]]:
    head_commit = _head_commit(codegen_workspace)
    committed_files = _collect_committed_changed_files(codegen_workspace, head_commit=head_commit)
    worktree_files = _collect_worktree_changed_files(codegen_workspace)
    changed_files = sorted(dict.fromkeys([*committed_files, *worktree_files]))
    return head_commit, changed_files, worktree_files


def _prune_codegen_noise(codegen_workspace: CodegenWorkspace) -> None:
    outputs_root = _runtime_root(codegen_workspace) / "outputs"
    if outputs_root.exists():
        shutil.rmtree(outputs_root, ignore_errors=True)
    for path in codegen_workspace.workspace_root.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    for suffix in CODEGEN_NOISE_SUFFIXES:
        for path in codegen_workspace.workspace_root.rglob(f"*{suffix}"):
            if path.is_file():
                path.unlink(missing_ok=True)


def _runtime_root(codegen_workspace: CodegenWorkspace) -> Path:
    return codegen_workspace.workspace_root / "BirdCLEF-2026-Codebase"


def _verify_config_argument(codegen_workspace: CodegenWorkspace, config_source: Path) -> str:
    runtime_root = _runtime_root(codegen_workspace)
    if config_source.is_relative_to(runtime_root):
        return str(config_source.relative_to(runtime_root))
    return str(config_source)


def _verify_summary(codegen_workspace: CodegenWorkspace, completed: subprocess.CompletedProcess[str]) -> str:
    result_path = codegen_workspace.verify_root / "result.json"
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            primary_name = str(payload.get("primary_metric_name", "") or "")
            primary_value = payload.get("primary_metric_value")
            verdict = str(payload.get("verdict", "") or "")
            if primary_name and primary_value is not None:
                return f"Verify run completed with {primary_name}={primary_value} and verdict={verdict or 'n/a'}."
            if verdict:
                return f"Verify run completed with verdict={verdict}."
    output = completed.stderr or completed.stdout or "deterministic verify command failed"
    return truncate(output, limit=240)


def _verify_codegen_workspace(codegen_workspace: CodegenWorkspace, config_source: Path) -> tuple[str, str, str]:
    train_entrypoint = codegen_workspace.workspace_root / "train_sed.py"
    if not train_entrypoint.exists():
        return "skipped", "", "train_sed.py not present in snapshot workspace."
    verify_command = f"{sys.executable} ./train_sed.py --config {_verify_config_argument(codegen_workspace, config_source)}"
    env = os.environ.copy()
    env.update(
        {
            "KAGGLE_AGENT_RUN_DIR": str(codegen_workspace.verify_root),
            "KAGGLE_AGENT_DISABLE_OUTPUT_MIRROR": "1",
            "KAGGLE_AGENT_VERIFY_MODE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(train_entrypoint), "--config", _verify_config_argument(codegen_workspace, config_source)],
        cwd=codegen_workspace.workspace_root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    status = "passed" if completed.returncode == 0 else "failed"
    return status, verify_command, _verify_summary(codegen_workspace, completed)


def _codegen_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Codegen",
        "",
        f"- Status: `{payload['status']}`",
        f"- Reason: {payload['reason']}",
        f"- Generated config: `{payload['generated_config_path'] or 'n/a'}`",
        f"- Patch: `{payload['patch_path'] or 'n/a'}`",
        f"- Code state: `{payload['code_state_ref'] or 'n/a'}`",
        f"- Verify: `{payload['verify_status']}` {payload['verify_summary']}",
        f"- Verify artifacts: `{payload['verify_artifacts_ref'] or 'n/a'}`",
    ]
    if payload["changed_files"]:
        lines.append(f"- Changed files: {', '.join(payload['changed_files'])}")
    return "\n".join(lines) + "\n"


def _materialize_codegen(
    ctx: StageContext,
    codegen_workspace: CodegenWorkspace,
    *,
    provider_runtime: str,
) -> dict[str, Any]:
    _prune_codegen_noise(codegen_workspace)
    head_commit_before_verify, changed_files, worktree_files = _collect_materialized_files(codegen_workspace)
    canonical = {
        "stage": "codegen",
        "status": "noop",
        "reason": "The agentic codegen provider completed without modifying the isolated stage workspace.",
        "generated_config_path": "",
        "run_bundle_path": "",
        "patch_path": "",
        "code_state_ref": "",
        "verify_artifacts_ref": "",
        "verify_command": "",
        "verify_status": "skipped",
        "verify_summary": "No code changes were materialized.",
        "worktree_path": "",
        "base_commit": codegen_workspace.base_commit,
        "head_commit": head_commit_before_verify if head_commit_before_verify != codegen_workspace.base_commit else "",
        "changed_files": [],
        "provider_runtime": provider_runtime,
        "allowed_edit_roots": list(CODEGEN_ALLOWED_EDIT_ROOTS),
        "smoke_status": "skipped",
        "smoke_summary": "No code changes were materialized.",
    }
    if not changed_files:
        return canonical

    _allow_codegen_paths(changed_files)
    config_source = _resolve_generated_config_source(codegen_workspace)
    if config_source is None:
        raise RuntimeError("codegen did not leave a runnable config in the isolated workspace")

    verify_status, verify_command, verify_summary = _verify_codegen_workspace(codegen_workspace, config_source)
    _prune_codegen_noise(codegen_workspace)
    head_commit_after_verify, _post_verify_changes, worktree_files = _collect_materialized_files(codegen_workspace)
    head_commit = head_commit_after_verify
    if worktree_files:
        _run_git(["add", "-A", "--", "."], codegen_workspace.workspace_root)
        _run_git(["commit", "-q", "-m", "Materialized codegen edits"], codegen_workspace.workspace_root)
        head_commit = _head_commit(codegen_workspace)
    final_changed_files = _collect_committed_changed_files(codegen_workspace, head_commit=head_commit)
    if not final_changed_files:
        canonical.update(
            {
                "reason": "The agentic codegen provider completed without leaving net source changes after transient codegen noise was pruned.",
                "head_commit": head_commit if head_commit != codegen_workspace.base_commit else "",
            }
        )
        return canonical

    _allow_codegen_paths(final_changed_files)
    staging_roots = [
        item
        for item in CODEGEN_ALLOWED_EDIT_ROOTS
        if (codegen_workspace.workspace_root / item).exists()
        or any(path == item or path.startswith(f"{item}/") for path in final_changed_files)
    ]
    patch_text = _run_git(
        ["diff", "--binary", f"{codegen_workspace.base_commit}..{head_commit}", "--", *staging_roots],
        codegen_workspace.workspace_root,
    )

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
        "verify_artifacts_ref": str(codegen_workspace.verify_root),
        "verify_command": verify_command,
        "verify_status": verify_status,
        "verify_summary": verify_summary,
        "changed_files": final_changed_files,
        "provider_runtime": provider_runtime,
    }
    atomic_write_json(run_bundle_path, run_bundle)
    canonical.update(
        {
            "status": "generated",
            "reason": "Materialized agentic codegen edits from the isolated stage workspace and recorded the deterministic verify result.",
            "generated_config_path": str(generated_config_path),
            "run_bundle_path": str(run_bundle_path),
            "patch_path": str(patch_path),
            "code_state_ref": str(codegen_workspace.workspace_root),
            "verify_artifacts_ref": str(codegen_workspace.verify_root),
            "verify_command": verify_command,
            "verify_status": verify_status,
            "verify_summary": verify_summary,
            "worktree_path": str(codegen_workspace.workspace_root),
            "head_commit": head_commit,
            "changed_files": final_changed_files,
            "smoke_status": verify_status,
            "smoke_summary": verify_summary,
        }
    )
    return canonical


def _materialize_stage_payload(
    ctx: StageContext,
    payload: dict[str, Any],
    codegen_workspace: CodegenWorkspace | None = None,
    *,
    provider_runtime: str = "",
) -> tuple[dict[str, Any], str, str]:
    markdown = str(payload.get("markdown", "")).strip()
    if codegen_workspace is not None:
        canonical = _materialize_codegen(ctx, codegen_workspace, provider_runtime=provider_runtime)
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
    stage_workspace: StageWorkspace | None = None,
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
        "provider_runtime": response.extra_meta.get("provider_runtime", ""),
    }
    if stage_workspace is not None:
        meta["stage_workspace_root"] = str(stage_workspace.workspace_root)
        meta["stage_workspace_storage_root"] = str(stage_workspace.stage_root)
        meta["stage_workspace_mode"] = stage_workspace.workspace_mode
    if codegen_workspace is not None:
        meta["workspace_root_used"] = str(codegen_workspace.workspace_root)
        meta["stage_workspace_storage_root"] = str(codegen_workspace.snapshot_root)
        meta["stage_workspace_mode"] = codegen_workspace.workspace_mode
        meta["verify_root"] = str(codegen_workspace.verify_root)
    for key in ["codex_profile", "adapter_alias"]:
        value = response.extra_meta.get(key, "")
        if value:
            meta[key] = value
    if amp_probe_path:
        meta["amp_probe_path"] = amp_probe_path
        meta["amp_probe_summary"] = amp_probe_summary
        meta["amp_thread_id"] = amp_thread_id
    return meta


def _build_repair_prompt(ctx: StageContext, validation_error: SchemaValidationError, payload: dict[str, Any]) -> str:
    schema = _schema(ctx)
    return (
        f"# Structured Output Repair\n\n"
        f"Your previous `{ctx.stage}` JSON did not match the schema.\n"
        f"Validation error: `{validation_error}`\n\n"
        "Return a single corrected JSON object that matches the schema exactly. "
        "Do not omit required fields, do not add extra fields, and keep the human-readable narrative in `markdown`.\n\n"
        "## Required Fields\n\n"
        + "\n".join(f"- `{item}`" for item in schema.get("required", []))
        + "\n\n"
        "## Previous Payload\n\n```json\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
        "```\n"
    )


def _merge_provider_attempts(
    original: ProviderResponse,
    repaired: ProviderResponse,
    *,
    validation_error: SchemaValidationError,
) -> ProviderResponse:
    merged = repaired
    stdout_parts = [part.strip() for part in [original.raw_stdout, repaired.raw_stdout] if part.strip()]
    stderr_parts = [part.strip() for part in [original.raw_stderr, repaired.raw_stderr] if part.strip()]
    event_parts = [part.strip() for part in [original.event_log_text, repaired.event_log_text] if part.strip()]
    merged.raw_stdout = "\n\n".join(stdout_parts)
    merged.raw_stderr = "\n\n".join(stderr_parts)
    merged.event_log_text = "\n\n".join(event_parts)
    merged.extra_meta.setdefault("validation_repair", str(validation_error))
    merged.extra_meta.setdefault("repair_attempted", "1")
    return merged


def _validate_or_repair_response(
    provider: str,
    ctx: StageContext,
    prompt: str,
    response: ProviderResponse,
    amp_probe: AmpProbeResult | None,
    *,
    provider_workspace_root: Path,
    codegen_workspace: CodegenWorkspace | None = None,
) -> tuple[ProviderResponse, AmpProbeResult | None]:
    if ctx.stage == "codegen":
        return response, amp_probe

    schema = _schema(ctx)
    try:
        validate_payload(schema, response.payload)
        return response, amp_probe
    except SchemaValidationError as error:
        repair_prompt = _build_repair_prompt(ctx, error, response.payload)
        repaired_response, repaired_amp_probe = _run_provider(
            provider,
            ctx,
            repair_prompt,
            provider_workspace_root=provider_workspace_root,
            codegen_workspace=codegen_workspace,
        )
        repaired_response = _merge_provider_attempts(response, repaired_response, validation_error=error)
        validate_payload(schema, repaired_response.payload)
        return repaired_response, repaired_amp_probe if repaired_amp_probe is not None else amp_probe


def _run_provider(
    provider: str,
    ctx: StageContext,
    prompt: str,
    *,
    provider_workspace_root: Path,
    codegen_workspace: CodegenWorkspace | None = None,
) -> tuple[ProviderResponse, AmpProbeResult | None]:
    codegen_env = (
        {
            "KAGGLE_AGENT_RUN_DIR": str(codegen_workspace.verify_root),
            "KAGGLE_AGENT_DISABLE_OUTPUT_MIRROR": "1",
            "KAGGLE_AGENT_VERIFY_MODE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if codegen_workspace is not None
        else None
    )
    if provider == "claude":
        return run_claude_headless(prompt=prompt, schema=_schema(ctx), workspace_root=provider_workspace_root), None
    if provider == "codex":
        return run_codex_exec(
            prompt=prompt,
            schema_path=ctx.schema_path if codegen_workspace is None else None,
            workspace_root=provider_workspace_root,
            output_dir=ctx.output_dir,
            mode="structured" if codegen_workspace is None else "agentic",
            extra_env=codegen_env,
        ), None
    if provider == "claude_code":
        return run_claude_code_exec(
            prompt=prompt,
            schema_path=ctx.schema_path if codegen_workspace is None else None,
            workspace_root=provider_workspace_root,
            mode="structured" if codegen_workspace is None else "agentic",
            extra_env=codegen_env,
        ), None
    if provider == "critic":
        response = run_claude_headless(prompt=prompt, schema=_schema(ctx), workspace_root=provider_workspace_root)
        amp_prompt = (
            "Review the following critic context and return a concise diagnostic summary. "
            "Do not modify files.\n\n"
            f"```json\n{json.dumps(ctx.input_manifest, indent=2, ensure_ascii=False)}\n```"
        )
        return response, run_amp_probe(prompt=amp_prompt, workspace_root=ctx.workspace_root)
    raise RuntimeError(f"Unknown provider: {provider}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless stage wrapper for Claude/Codex/Amp adapters.")
    parser.add_argument("--provider", required=True, choices=["claude", "claude_code", "codex", "critic"])
    parser.add_argument(
        "--dangerously-bypass-approvals-and-sandbox",
        action="store_true",
        help="Accepted for backward compatibility with older workspace command templates.",
    )
    args = parser.parse_args(argv)

    ctx: StageContext | None = None
    response: ProviderResponse | None = None
    amp_probe: AmpProbeResult | None = None
    try:
        ctx = StageContext.from_env()
        stage_workspace = _prepare_stage_workspace(ctx) if ctx.stage in {"plan", "codegen"} and args.provider in {"claude_code", "codex"} else None
        codegen_workspace = _prepare_codegen_workspace(ctx, stage_workspace) if ctx.stage == "codegen" else None
        provider_workspace_root = (
            codegen_workspace.workspace_root
            if codegen_workspace is not None
            else stage_workspace.workspace_root
            if stage_workspace is not None
            else ctx.workspace_root
        )
        prompt = _build_prompt(ctx, codegen_workspace, provider_workspace_root=provider_workspace_root)
        started_at = now_utc_iso()
        response, amp_probe = _run_provider(
            args.provider,
            ctx,
            prompt,
            provider_workspace_root=provider_workspace_root,
            codegen_workspace=codegen_workspace,
        )
        response.extra_meta.setdefault("started_at", started_at)
        response.extra_meta.setdefault("completed_at", now_utc_iso())
        response, amp_probe = _validate_or_repair_response(
            args.provider,
            ctx,
            prompt,
            response,
            amp_probe,
            provider_workspace_root=provider_workspace_root,
            codegen_workspace=codegen_workspace,
        )
        payload, markdown, spec_path = _materialize_stage_payload(
            ctx,
            response.payload,
            codegen_workspace,
            provider_runtime=str(response.extra_meta.get("provider_runtime", "")),
        )
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
            stage_workspace=stage_workspace,
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
        if ctx is not None and response is not None:
            _write_raw_capture(ctx.output_dir / "raw_stdout.txt", response.raw_stdout)
            _write_raw_capture(ctx.output_dir / "raw_stderr.txt", response.raw_stderr)
            if response.event_log_text:
                _write_raw_capture(ctx.output_dir / "events.jsonl", response.event_log_text)
        print(f"schema validation failed: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001
        if ctx is not None and response is not None:
            _write_raw_capture(ctx.output_dir / "raw_stdout.txt", response.raw_stdout)
            _write_raw_capture(ctx.output_dir / "raw_stderr.txt", response.raw_stderr)
            if response.event_log_text:
                _write_raw_capture(ctx.output_dir / "events.jsonl", response.event_log_text)
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
