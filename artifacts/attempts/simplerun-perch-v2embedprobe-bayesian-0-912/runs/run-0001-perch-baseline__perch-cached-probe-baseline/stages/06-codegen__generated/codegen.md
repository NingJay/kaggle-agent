# Codegen

- Status: `generated`
- Reason: Materialized agentic codegen edits from the isolated stage workspace and recorded the deterministic verify result.
- Generated config: `/home/staff/jiayining/kaggle/worktrees/kaggle-agent-branch-search/artifacts/attempts/simplerun-perch-v2embedprobe-bayesian-0-912/runs/run-0001-perch-baseline__perch-cached-probe-baseline/stages/06-codegen__running/generated_config.yaml`
- Patch: `/home/staff/jiayining/kaggle/worktrees/kaggle-agent-branch-search/artifacts/attempts/simplerun-perch-v2embedprobe-bayesian-0-912/runs/run-0001-perch-baseline__perch-cached-probe-baseline/stages/06-codegen__running/patch.diff`
- Code state: `/home/staff/jiayining/kaggle/worktrees/kaggle-agent-branch-search/state/worktrees/codegen/06-codegen__running/workspace`
- Verify: `passed` Verify run completed with val_soundscape_macro_roc_auc=0.6718809613270702 and verdict=submission-required.
- Verify artifacts: `/home/staff/jiayining/kaggle/worktrees/kaggle-agent-branch-search/state/worktrees/codegen/06-codegen__running/verify_runtime`
- Changed files: BirdCLEF-2026-Codebase/configs/generated/class-coverage-fix-resolve-23-unfitted-classes.yaml

## Branch Context
- branch_role: `primary`
- idea_class: `grounded`
- portfolio_id: `portfolio-run-0001-perch-baseline-perch-cached-probe`
- motivation: Primary branch. Required pattern class_coverage. Strongest policy support (conditional with positive empirical evidence: 52→67 gave +0.013). 31% coverage deficit is the structural blocker blocking all downstream improvements.

## Policy Trace
- `role:primary:8.0`

## Scheduler Hints
- portfolio_cap: `1`
- idea_class_cap: `1`
- dispatch_priority: `18.04`
- expected_information_gain: `0.84`
- novelty_score: `1.0`
- low_information_flag: `False`
- grounding_mode: `grounded`
- cost_tier: `low`
- cost_units: `1.0`
- cost_budget: `0.0`
- max_budget_share: `0.0`
- cost_caps: `{}`
- smoke_only_first: `False`
- canary_eval_required: `False`
- auto_kill_threshold: `0.0`

## Typing
- proposal_typing_id: `proposal-run-0001-perch-baseline-class-coverage-fix-resolve-23-unfitted-classes-probe-head`
- realized_typing_id: `realized-run-0001-perch-baseline-class-coverage-fix-resolve-23-unfitted-classes-class-cove`
