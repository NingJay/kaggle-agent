# Decision Program

Choose the next action with explicit justification, submission stance, and expected follow-up scope.

Rules:
- Avoid collapsing immediately to a single low-information calibration or blend sweep when higher-value training/data axes remain open.
- If the result is still below target, keep room for multi-branch follow-up search in the next plan stage.
- Use negative priors as vetoes unless you can explicitly justify overriding them.
- Persist a readable `portfolio_intent`, `portfolio_policy`, `branch_mix`, and `deprioritized_axes` so downstream scheduling is auditable.
