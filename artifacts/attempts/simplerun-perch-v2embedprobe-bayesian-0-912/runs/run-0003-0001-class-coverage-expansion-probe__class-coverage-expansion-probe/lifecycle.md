# Lifecycle run-0003-0001-class-coverage-expansion-probe__class-coverage-expansion-probe

- Run id: `run-0003-0001-class-coverage-expansion-probe`
- Lifecycle template: `branch_experiment`
- Run status: `succeeded`
- Current cursor: `complete`
- Latest realized stage: `09-submission__skipped`
- Note: Stage directory prefixes keep canonical stage ids. Actual execution order follows `stage_plan` below.

## Planned Execution Order
1. `codegen` | status=`noop` | canonical_dir=`06-codegen__*` | artifact=`06-codegen__noop`
2. `critic` | status=`approved` | canonical_dir=`07-critic__*` | artifact=`07-critic__approved`
3. `validate` | status=`not-required` | canonical_dir=`08-validate__*` | artifact=`08-validate__not-required`
4. `execute` | status=`completed` | canonical_dir=`execute`
5. `evidence` | status=`succeeded` | canonical_dir=`01-evidence__*` | artifact=`01-evidence__succeeded`
6. `report` | status=`completed` | canonical_dir=`02-report__*` | artifact=`02-report__completed`
7. `research` | status=`completed` | canonical_dir=`03-research__*` | artifact=`03-research__completed`
8. `decision` | status=`completed` | canonical_dir=`04-decision__*` | artifact=`04-decision__completed`
9. `plan` | status=`planned` | canonical_dir=`05-plan__*` | artifact=`05-plan__planned`
10. `submission` | status=`skipped` | canonical_dir=`09-submission__*` | artifact=`09-submission__skipped`
