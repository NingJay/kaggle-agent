# Plan run-0001-perch-baseline

- Status: `submission_candidate`
- Title: Raw probe baseline submission bundle
- Config: `BirdCLEF-2026-Codebase/configs/default.yaml`
- Launch mode: `foreground`
- Lifecycle: `submission_from_target_run`
- Stage plan: `submission`
- Target run: `run-0001-perch-baseline`
- Hypothesis: Packaging run-0001-perch-baseline as a raw-probe submission bundle with no prior fusion and no calibration will establish the first leaderboard anchor at approximately val ROC-AUC 0.665, grounding all future calibration and coverage-expansion decisions.

## Branch Portfolio
- `primary` | Class coverage expansion probe | idea=data_coverage | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=8.0 | config=`BirdCLEF-2026-Codebase/configs/generated/class-coverage-expansion-probe.yaml`
- `tertiary` | Calibrated prior fusion v2 | idea=post_processing | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=0.0 | config=`BirdCLEF-2026-Codebase/configs/generated/calibrated-prior-fusion-v2.yaml`

## Pruned Branches
- Raw probe baseline submission bundle | reason=portfolio_budget | vetoes=n/a
- Domain shift mitigation soundscape augmentation | reason=portfolio_budget | vetoes=n/a

## Policy Trace
- `card:01-validated-findings-md-perch-and-embedding-findings:require`
- `card:01-validated-findings-md-validation-and-metric-semantics:require`
- `card:research-run-0001-perch-baseline-md-root-cause:prefer`
- `card:research-run-0001-perch-baseline-md-adopt-now:prefer`
- `card:research-run-0001-perch-baseline-md-reject:avoid`
- `card:research-run-0002-0001-perch-probe-iteration-class-coverage-verification-and-calibration-md-run-0002-0001-perch-probe-iteration-class-coverage-verification-and-calibration:conditional`
- `role:primary:8.0`
- `role:tertiary:0.0`
