# Plan run-0002-0002-calibrated-prior-fusion-v2

- Status: `planned`
- Title: Ship leader bundle and open probe-head architecture axis
- Config: `BirdCLEF-2026-Codebase/configs/generated/ship-leader-bundle-and-open-probe-head-architecture-axis.yaml`
- Launch mode: `background`
- Lifecycle: `submission_from_target_run`
- Stage plan: `submission`
- Target run: `run-0002-0002-calibrated-prior-fusion-v2`
- Hypothesis: The prior fusion axis is exhausted. The immediate priority is packaging the critic-approved leader (run-0003-0001, val ROC-AUC 0.6652) for submission. After securing a submission slot, the next gains are likely from probe-head architecture improvements (MLP regularization without prior fusion, based on the coverage-MLP context card showing 0.672 on a different split) and fixing the abnormally high training temperature (7.39) identified in the conditional lead.

## Branch Portfolio
- `primary` | Ship leader bundle and open probe-head architecture axis | idea=submission_packaging | lifecycle=`submission_from_target_run` | target_run=`run-0002-0002-calibrated-prior-fusion-v2` | stages=`submission` | score=8.0 | config=`BirdCLEF-2026-Codebase/configs/generated/ship-leader-bundle-and-open-probe-head-architecture-axis.yaml`
- `secondary` | Ship leader bundle and open probe-head architecture axis branch 2 | idea=probe_head_architecture | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=0.0 | config=`BirdCLEF-2026-Codebase/configs/generated/ship-leader-bundle-and-open-probe-head-architecture-axis-branch-2.yaml`
- `tertiary` | Ship leader bundle and open probe-head architecture axis branch 3 | idea=temperature_fix | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=0.0 | config=`BirdCLEF-2026-Codebase/configs/generated/ship-leader-bundle-and-open-probe-head-architecture-axis-branch-3.yaml`

## Pruned Branches
- Submission bundle: run-0003-0001 leader | reason=portfolio_budget | vetoes=n/a
- MLP probe-head regularization without prior fusion | reason=portfolio_budget | vetoes=n/a
- Temperature audit: reduce from 7.39 to 1.5 | reason=portfolio_budget | vetoes=n/a

## Policy Trace
- `card:01-validated-findings-md-perch-and-embedding-findings:require`
- `card:research-run-0003-0002-prior-fusion-weight-calibration-md-adopt-now:prefer`
- `card:research-run-0003-0002-prior-fusion-weight-calibration-md-root-cause:avoid`
- `card:03-next-experiment-priors-md-perch-probe-lane:conditional`
- `card:research-run-0002-0002-calibrated-prior-fusion-v2-md-adopt-now:prefer`
- `card:research-run-0002-0002-calibrated-prior-fusion-v2-md-result-summary:avoid`
- `role:primary:8.0`
- `role:secondary:0.0`
- `role:tertiary:0.0`
