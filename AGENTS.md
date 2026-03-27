# Kaggle Research OS Contract

Operate this repository as an agent-primary, artifact-driven, spec-enforced Kaggle research system for BirdCLEF 2026.

## Hard Rules

- Keep `state/ledger.db` as machine truth.
- Keep `artifacts/` as the provenance layer for runs, stage outputs, validations, and submission bundles.
- Every autonomous stage must emit both `*.json` and `*.md`.
- Keep `train_sed.py` as the stable runtime bridge.
- Preserve the strict stage graph: `execute -> evidence -> report -> research -> decision -> plan -> codegen -> critic -> validate -> submission`.
- Keep scored submission defaults CPU-only and internet-off.

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
