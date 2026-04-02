# Decision run-0001-perch-baseline

- Decision type: `tune`
- Next action: `run_new_experiment`
- Submission recommendation: `no`
- Root cause: The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration.
- Why: Continue frontier search, but enforce grounded-vs-novel slot allocation and minimum information gain.
- Portfolio mode: `frontier_search`
- Grounded slots: `2`
- Novel slots: `1`

## Branch Mix
- `primary` -> `class_coverage`
- `hedge` -> `probe_head`
- `explore` -> `optimization`

## Forbidden Patterns

## Required Patterns
- `coverage_first`
- `probe_training_change`
- `schedule_recovery`

## Capability Packs
- `branch_typing_compiler`
- `veto_checker`
- `ledger_miner`
