# Lifecycle run-0001-perch-baseline__perch-cached-probe-baseline

- Run id: `run-0001-perch-baseline`
- Lifecycle template: `recursive_experiment`
- Run status: `running`
- Current cursor: `complete`
- Latest realized stage: `n/a`
- Note: Stage directory prefixes keep canonical stage ids. Actual execution order follows `stage_plan` below.

## Planned Execution Order
1. `execute` | status=`running` | canonical_dir=`execute`
2. `evidence` | status=`pending` | canonical_dir=`01-evidence__*`
3. `report` | status=`pending` | canonical_dir=`02-report__*`
4. `research` | status=`pending` | canonical_dir=`03-research__*`
5. `decision` | status=`pending` | canonical_dir=`04-decision__*`
6. `plan` | status=`pending` | canonical_dir=`05-plan__*`
7. `codegen` | status=`pending` | canonical_dir=`06-codegen__*`
8. `critic` | status=`pending` | canonical_dir=`07-critic__*`
9. `validate` | status=`pending` | canonical_dir=`08-validate__*`
10. `submission` | status=`pending` | canonical_dir=`09-submission__*`
