# Plan run-0001-perch-baseline

- Status: `planned`
- Title: Class coverage fix — resolve 23 unfitted classes
- Config: `BirdCLEF-2026-Codebase/configs/generated/class-coverage-fix-resolve-23-unfitted-classes.yaml`
- Launch mode: `background`
- Lifecycle: `branch_terminal_experiment`
- Stage plan: `codegen -> critic -> validate -> execute -> evidence -> report`
- Target run: `n/a`
- Hypothesis: Fixing the 23 unfitted classes will raise val AUC above 0.68 by closing the class coverage gap from 52/75 to 75/75 fitted classes.

## Branch Portfolio
- `primary` | Class coverage fix — resolve 23 unfitted classes | idea=grounded | grounding=`grounded` | lifecycle=`branch_terminal_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report` | score=18.04 | info_gain=0.84 | novelty=1.0 | config=`BirdCLEF-2026-Codebase/configs/generated/class-coverage-fix-resolve-23-unfitted-classes.yaml`
  - falsify: Val AUC does not improve when all 75 classes are fitted
  - kill: After fixing coverage, val AUC remains below 0.66
  - requires evidence: fitted_class_count reaches 75/75, val_soundscape_macro_roc_auc improves above 0.665 baseline
- `novel` | Pseudo-label unfitted classes from Bayesian prior | idea=novel | grounding=`novel` | lifecycle=`branch_terminal_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report` | score=9.28 | info_gain=0.88 | novelty=1.0 | config=`BirdCLEF-2026-Codebase/configs/generated/pseudo-label-unfitted-classes-from-bayesian-prior.yaml`
  - falsify: Pseudo-labeled classes do not achieve above-random predictions (AUC > 0.5)
  - kill: Pseudo-labeling produces no class coverage improvement or degrades existing class performance
  - unsupported: pseudo_label may unlock holdout improvements despite limited direct support
  - requires evidence: Pseudo-labeled classes achieve non-random predictions (AUC > 0.5), Overall val AUC improves vs baseline 0.665
- `primary` | Class coverage fix — resolve 23 unfitted classes | idea=grounded | grounding=`grounded` | lifecycle=`branch_experiment` | target_run=`n/a` | stages=`codegen -> critic -> validate -> execute -> evidence -> report -> research -> decision -> plan -> submission` | score=17.08 | info_gain=0.68 | novelty=1.0 | config=`BirdCLEF-2026-Codebase/configs/generated/class-coverage-fix-resolve-23-unfitted-classes.yaml`
  - falsify: Reject if Class coverage fix — resolve 23 unfitted classes does not improve holdout validation.
  - kill: Kill the branch if it only changes low-information post-processing.

## Pruned Branches
- Class coverage fix — resolve 23 unfitted classes | reason=grounded_budget | vetoes=n/a
- Label smoothing regularization on probe logits | reason=grounded_budget | vetoes=n/a
- Label smoothing regularization on probe logits | reason=grounded_budget | vetoes=n/a
- Pseudo-label unfitted classes from Bayesian prior | reason=grounded_budget | vetoes=n/a
-  | reason=policy_veto | vetoes=low-information-typing

## Policy Trace
- `rule:policy-class-coverage:conditional`
- `rule:policy-prior-calibration:conditional`
- `rule:policy-probe-head:conditional`
- `role:primary:8.0`
- `role:novel:0.0`

## Capability Packs
- `veto_checker`
- `branch_diversifier`
- `branch_typing_compiler`
- `novel_hypothesis_generator`

## Open Questions
- Will fixing the 23 unfitted classes alone close the train/val AUC gap, or does the probe architecture itself need fundamental change?
- Can pseudo-labeling from the Bayesian prior produce reliable labels for the 23 unfitted classes?
- Is label smoothing sufficient to reduce overfitting given the small ~700-window training set?
