# Lifecycle run-0002-0001-bundle-parity-cpu-only-submission-dry-run__bundle-parity-cpu-only-submission-dry-run

- Run id: `run-0002-0001-bundle-parity-cpu-only-submission-dry-run`
- Lifecycle template: `branch_experiment`
- Run status: `running`
- Current cursor: `complete`
- Latest realized stage: `08-validate__not-required`
- Note: Stage directory prefixes keep canonical stage ids. Actual execution order follows `stage_plan` below.

## Planned Execution Order
1. `codegen` | status=`noop` | canonical_dir=`06-codegen__*` | artifact=`06-codegen__noop`
2. `critic` | status=`approved` | canonical_dir=`07-critic__*` | artifact=`07-critic__approved`
3. `validate` | status=`not-required` | canonical_dir=`08-validate__*` | artifact=`08-validate__not-required`
4. `execute` | status=`running` | canonical_dir=`execute`
5. `evidence` | status=`pending` | canonical_dir=`01-evidence__*`
6. `report` | status=`pending` | canonical_dir=`02-report__*`
7. `research` | status=`pending` | canonical_dir=`03-research__*`
8. `decision` | status=`pending` | canonical_dir=`04-decision__*`
9. `plan` | status=`pending` | canonical_dir=`05-plan__*`
10. `submission` | status=`pending` | canonical_dir=`09-submission__*`
