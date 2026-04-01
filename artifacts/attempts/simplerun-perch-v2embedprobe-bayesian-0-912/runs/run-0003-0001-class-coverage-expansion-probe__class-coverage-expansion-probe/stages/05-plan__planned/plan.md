# Plan run-0003-0001-class-coverage-expansion-probe

- Status: `planned`
- Title: MLP probe head adoption from run-0002-0003
- Config: `BirdCLEF-2026-Codebase/configs/generated/mlp-probe-head-adoption-from-run-0002-0003.yaml`
- Launch mode: `background`
- Lifecycle: `branch_experiment`
- Stage plan: `codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission`
- Target run: `n/a`
- Hypothesis: Adopting the lane-leader MLP probe head (0.6801) as the primary configuration, then applying temperature scaling calibration, will close the 0.015 AUC gap before submission packaging. A speculative domain-shift augmentation branch may further narrow the 0.327 train-val gap.

## Branch Portfolio
- `primary` | MLP probe head adoption from run-0002-0003 | idea=data_coverage | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=8.0 | config=`BirdCLEF-2026-Codebase/configs/generated/mlp-probe-head-adoption-from-run-0002-0003.yaml`
- `speculative` | Domain-shift augmentation on training soundscapes | idea=augmentation | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=0.0 | config=`BirdCLEF-2026-Codebase/configs/generated/domain-shift-augmentation-on-training-soundscapes.yaml`
- `secondary` | Temperature scaling calibration on MLP logits | idea=calibration | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=0.0 | config=`BirdCLEF-2026-Codebase/configs/generated/temperature-scaling-calibration-on-mlp-logits.yaml`

## Pruned Branches
- MLP probe head adoption from run-0002-0003 | reason=portfolio_budget | vetoes=n/a
- Domain-shift augmentation on training soundscapes | reason=portfolio_budget | vetoes=n/a
- Temperature scaling calibration on MLP logits | reason=portfolio_budget | vetoes=n/a

## Policy Trace
- `card:01-validated-findings-md-perch-and-embedding-findings:require`
- `card:research-run-0002-0003-coverage-mlp-combined-full-75-class-mlp-probe-md-adopt-now:prefer`
- `card:experiment-conclusions-md-run-0001-perch-baseline:prefer`
- `card:research-run-0009-0008-perch-probe-round-9-embedding-level-mixup-augmentation-on-run-0008-mlp-head-code-state-md-axis-exhaustion-summary:avoid`
- `card:research-run-0011-0009-perch-probe-round-11-mlp-dropout-regularization-on-run-0008-leader-md-full-trajectory:avoid`
- `card:research-run-0001-perch-baseline-md-root-cause:prefer`
- `role:primary:8.0`
- `role:speculative:0.0`
- `role:secondary:0.0`
