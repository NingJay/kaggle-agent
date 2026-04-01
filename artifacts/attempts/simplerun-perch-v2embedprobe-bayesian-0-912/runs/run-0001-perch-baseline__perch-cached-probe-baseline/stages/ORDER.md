# Lifecycle run-0001-perch-baseline__perch-cached-probe-baseline

- Run id: `run-0001-perch-baseline`
- Lifecycle template: `recursive_experiment`
- Run status: `succeeded`
- Current cursor: `complete`
- Latest realized stage: `09-submission__candidate-created`
- Note: Stage directory prefixes keep canonical stage ids. Actual execution order follows `stage_plan` below.

## Planned Execution Order
1. `execute` | status=`completed` | canonical_dir=`execute`
2. `evidence` | status=`succeeded` | canonical_dir=`01-evidence__*` | artifact=`01-evidence__succeeded`
3. `report` | status=`completed` | canonical_dir=`02-report__*` | artifact=`02-report__completed`
4. `research` | status=`completed` | canonical_dir=`03-research__*` | artifact=`03-research__completed`
5. `decision` | status=`completed` | canonical_dir=`04-decision__*` | artifact=`04-decision__completed`
6. `plan` | status=`submission-candidate` | canonical_dir=`05-plan__*` | artifact=`05-plan__submission-candidate`
7. `codegen` | status=`noop` | canonical_dir=`06-codegen__*` | artifact=`06-codegen__noop`
8. `critic` | status=`approved` | canonical_dir=`07-critic__*` | artifact=`07-critic__approved`
9. `validate` | status=`validated` | canonical_dir=`08-validate__*` | artifact=`08-validate__validated`
10. `submission` | status=`candidate-created` | canonical_dir=`09-submission__*` | artifact=`09-submission__candidate-created`
