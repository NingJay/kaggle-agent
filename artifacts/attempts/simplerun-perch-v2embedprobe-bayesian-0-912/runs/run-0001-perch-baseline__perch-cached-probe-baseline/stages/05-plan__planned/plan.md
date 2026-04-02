# Plan run-0001-perch-baseline

- Status: `planned`
- Title: Bundle parity: CPU-only submission dry-run
- Config: `BirdCLEF-2026-Codebase/configs/generated/bundle-parity-cpu-only-submission-dry-run.yaml`
- Launch mode: `background`
- Lifecycle: `branch_experiment`
- Stage plan: `codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission`
- Target run: `n/a`
- Hypothesis: Packaging the Perch embedding cache + linear probe + Bayesian prior into a CPU-only, internet-off notebook will reproduce val_soundscape_macro_roc_auc=0.665 within tolerance, unblocking the submission pipeline.

## Branch Portfolio
- `primary` | Bundle parity: CPU-only submission dry-run | idea=backbone | grounding=`grounded` | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=19.48 | info_gain=0.84 | novelty=1.0 | config=`BirdCLEF-2026-Codebase/configs/generated/bundle-parity-cpu-only-submission-dry-run.yaml`
  - falsify: Reject if Bundle parity: CPU-only submission dry-run does not improve holdout validation.
  - kill: Kill the branch if it only changes low-information post-processing.
- `primary` | Bundle parity: CPU-only submission dry-run | idea=grounded | grounding=`grounded` | lifecycle=`branch_terminal_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report` | score=17.08 | info_gain=0.68 | novelty=1.0 | config=`BirdCLEF-2026-Codebase/configs/default.yaml`
  - falsify: Dry-run prediction checksum does not match local run-0001 predictions within 1e-6 tolerance.
  - kill: Bundle fails to run within 90-minute CPU limit or produces empty submission CSV.
- `hedge` | Prior calibration: anchor-based refinement | idea=grounded | grounding=`grounded` | lifecycle=`branch_terminal_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report` | score=14.08 | info_gain=0.68 | novelty=1.0 | config=`BirdCLEF-2026-Codebase/configs/default.yaml`
  - falsify: Calibrated prior fusion score does not improve over 0.662 on the same holdout split.
  - kill: Calibration introduces regression below 0.65 on any holdout metric.

## Pruned Branches
- Prior calibration: anchor-based refinement | reason=grounded_budget | vetoes=n/a
- Class coverage expansion | reason=grounded_budget | vetoes=n/a
- Probe diagnostic: fitted vs active class count | reason=grounded_budget | vetoes=n/a
- Preprocessing augmentation sweep | reason=grounded_budget | vetoes=n/a
- Pseudo-label self-training | reason=grounded_budget | vetoes=n/a
- Probe diagnostic: fitted vs active class count | reason=grounded_budget | vetoes=n/a
- Prior calibration: anchor-based refinement | reason=policy_veto | vetoes=low-information-typing

## Policy Trace
- `rule:policy-backbone:prefer`
- `rule:policy-class-coverage:conditional`
- `rule:policy-prior-calibration:conditional`
- `rule:policy-probe-head:conditional`
- `rule:policy-preprocessing-aug:prefer`
- `role:primary:8.0`
- `policy:prefer:backbone:policy-backbone`
- `role:hedge:5.0`

## Capability Packs
- `veto_checker`
- `branch_diversifier`
- `branch_typing_compiler`
- `novel_hypothesis_generator`

## Open Questions
- What is the fitted_class_count vs active_class_count for the run-0001 probe head?
- Does the current code produce a submission-ready CSV, or does the bundle need new inference glue code?
- What is the expected LB score range given 0.665 local validation?
