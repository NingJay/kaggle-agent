# Plan run-0001-perch-baseline

- Status: `planned`
- Title: Perch cached-probe baseline follow-up
- Config: `BirdCLEF-2026-Codebase/configs/generated/perch-cached-probe-baseline-follow-up.yaml`
- Launch mode: `background`
- Lifecycle: `branch_experiment`
- Stage plan: `codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission`
- Target run: `n/a`
- Hypothesis: Continue frontier search, but enforce grounded-vs-novel slot allocation and minimum information gain.

## Branch Portfolio
- `primary` | Perch cached-probe baseline follow-up | idea=class_coverage | grounding=`grounded` | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=21.34 | info_gain=0.89 | novelty=1.0 | config=`BirdCLEF-2026-Codebase/configs/generated/perch-cached-probe-baseline-follow-up.yaml`
  - falsify: Reject if this branch does not improve holdout validation.
  - kill: Kill the branch if it only changes low-information post-processing.
- `hedge` | Perch cached-probe baseline follow-up branch 2 | idea=probe_head | grounding=`grounded` | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=18.34 | info_gain=0.89 | novelty=1.0 | config=`BirdCLEF-2026-Codebase/configs/generated/perch-cached-probe-baseline-follow-up-branch-2.yaml`
  - falsify: Reject if this branch does not improve holdout validation.
  - kill: Kill the branch if it only changes low-information post-processing.
- `explore` | Perch cached-probe baseline follow-up branch 3 | idea=optimization | grounding=`grounded` | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=15.38 | info_gain=0.73 | novelty=1.0 | config=`BirdCLEF-2026-Codebase/configs/generated/perch-cached-probe-baseline-follow-up-branch-3.yaml`
  - falsify: Reject if this branch does not improve holdout validation.
  - kill: Kill the branch if it only changes low-information post-processing.

## Pruned Branches
-  | reason=grounded_budget | vetoes=n/a

## Policy Trace
- `rule:policy-prior-calibration:conditional`
- `role:primary:8.0`
- `required-pattern:coverage_first`
- `role:hedge:5.0`
- `required-pattern:probe_training_change`
- `role:explore:3.0`
- `required-pattern:schedule_recovery`

## Capability Packs
- `branch_typing_compiler`
- `veto_checker`
- `ledger_miner`

## Open Questions
- What removes the current bottleneck without repeating low-information sweeps: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.?
