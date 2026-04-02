# Lifecycle run-0003-0002-pseudo-label-unfitted-classes-from-bayesian-prior__pseudo-label-unfitted-classes-from-bayesian-prior

- Run id: `run-0003-0002-pseudo-label-unfitted-classes-from-bayesian-prior`
- Lifecycle template: `branch_terminal_experiment`
- Run status: `succeeded`
- Current cursor: `complete`
- Latest realized stage: `02-report__completed`
- Note: Stage directory prefixes keep canonical stage ids. Actual execution order follows `stage_plan` below.

## Planned Execution Order
1. `codegen` | status=`noop` | canonical_dir=`06-codegen__*` | artifact=`06-codegen__noop`
2. `critic` | status=`approved` | canonical_dir=`07-critic__*` | artifact=`07-critic__approved`
3. `validate` | status=`not-required` | canonical_dir=`08-validate__*` | artifact=`08-validate__not-required`
4. `execute` | status=`completed` | canonical_dir=`execute`
5. `evidence` | status=`succeeded` | canonical_dir=`01-evidence__*` | artifact=`01-evidence__succeeded`
6. `report` | status=`completed` | canonical_dir=`02-report__*` | artifact=`02-report__completed`
