# Kaggle Research OS Contract

Operate this repository as an agent-primary, artifact-driven, spec-enforced Kaggle research system for BirdCLEF 2026.

## Hard Rules

- Keep `state/ledger.db` as machine truth.
- Keep `artifacts/` as the provenance layer for runs, stage outputs, validations, and submission bundles.
- Every autonomous stage must emit both `*.json` and `*.md`.
- Keep `train_sed.py` as the stable runtime bridge.
- Preserve the strict stage graph: `execute -> evidence -> report -> research -> decision -> plan -> codegen -> critic -> validate -> submission`.
- Keep scored submission defaults CPU-only and internet-off.
- For this worktree, run experiment orchestration, monitoring, status checks, and training from `ssh hpcgpu13` unless the human explicitly overrides it. Do not treat the local non-HPC shell as the authoritative runtime environment.
- When starting any experiment, stage adapter, monitor loop, or long-running CLI command, stay attached and keep monitoring until it reaches a terminal state or a deliberate human handoff is recorded.
- Do not fire-and-forget background work. While waiting, keep checking process status, stage directories, and ledger-visible progress so stalls or provider hangs are caught and handled immediately.
- If a turn is interrupted while work is still running, re-establish monitoring first on the next turn before doing anything else.
- Never downgrade, weaken, reinterpret away, or silently narrow explicit human instructions or quality requirements. When in doubt, preserve the stricter requirement and carry it forward into implementation, monitoring, and reporting.
- Never download model weights or runtime model assets from the internet for SED or related experiment backbones. SED paths must use local model files only, and any missing local-model requirement must be treated as a blocker to fix rather than a reason to fall back to online fetches.

## Environment Bootstrap

- Before running CLI commands or repo-local tests, execute `source /home/staff/jiayining/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-agent`.
- Runtime launch scripts should preserve the same `kaggle-agent` conda activation contract through `workspace.toml`.

## Ask-First Boundaries

- Real Kaggle submissions
- Publishing Kaggle datasets or kernels
- Killing non-agent processes
- Deleting archived legacy state
- Installing heavyweight dependencies globally

## Repo Surface

The human working surface is:

- `CHECKLIST.md`
- `JOURNAL.md`
- `FINDINGS.md`
- `ISSUES.md`
- `SUBMISSIONS.md`
- `reports/*.html`

The machine interfaces are:

- `state/ledger.db`
- `artifacts/*/*.json`
- `artifacts/*/*.md`

Long prose is not the API. HTML is not the source of truth.
