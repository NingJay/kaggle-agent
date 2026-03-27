# Codegen Program

Operate only inside the isolated codegen workspace.

Goals:
- Leave a runnable config inside the isolated workspace.
- Keep source edits inside the explicit allowlist.
- Keep verify artifacts out of the source tree.
- Let the harness own the final deterministic verify run and manifest export.

Rules:
- Edit only `train_sed.py`, `BirdCLEF-2026-Codebase/configs/**`, `BirdCLEF-2026-Codebase/src/**`, `BirdCLEF-2026-Codebase/train.py`, `BirdCLEF-2026-Codebase/inference.py`, and `BirdCLEF-2026-Codebase/scripts/**`.
- Never edit `BirdCLEF-2026-Codebase/outputs/**`, `BirdCLEF-2026-Codebase/models/**`, `BirdCLEF-2026-Codebase/birdclef-2026/**`, `state/**`, or `artifacts/**`.
- Never create notebooks or binary artifact files such as `.ipynb`, `.npz`, `.pkl`, `.pt`, or `.ckpt`.
- Do not return patch text, YAML blobs, or JSON manifests in the final message.
- Finish with a short plain-text summary of source edits only.
