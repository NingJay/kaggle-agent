# Research Axes From Harness References

This is the compact working summary of the local harness reference set.

## Axis 1: Report-Driven Operating Loop

- The strongest pattern across `master_report`, `experiment_report`, `discovery_report`, and `progress_report` is that the project behaves like a long-running research cockpit.
- Reports are being used to decide what is trustworthy, what is stale, which probes are worth promoting, and which issues need another wave of experiments.
- This validates keeping reports, findings, issues, and submission notes as persistent knowledge objects.

## Axis 2: Validation Discipline

- `experiment_report_20260316.html` and `progress_report.html` emphasize holdout construction and explicit no-leak language.
- `nohuman_pp_eval_20260315_2158.html` shows that local raw scores, post-process gains, predicted LB, and actual LB must be separated.
- The durable lesson is that metric names alone are insufficient; split and trust semantics must travel with the number.

## Axis 3: Submission Intelligence

- `automatic_submission_workflow.txt` captures the code-competition mechanics: local artifacts, Kaggle dataset packaging, notebook push, rerun on Kaggle.
- `nohuman_pp_eval_20260315_2158.html` adds the missing competitive layer: LB anchors, calibration formulas, and post-process probe strategy.
- Together they imply that submission work is not just bundle generation; it is candidate selection plus calibration plus slot management.

## Axis 4: Public SED And Internal SED Roadmap

- `public_sed_comparison_report.html` is a structured adoption document, not just a benchmark page.
- `sed_improvement_plan.html` and `sed_model_comparison_20260320.html` show a separate long-running SED workstream with backbone, loss, pseudo-label, and fold strategy comparisons.
- This material belongs in knowledge because it is repeat-use design memory for future planners.

## Axis 5: Domain Generalization With Perch

- `domain_generalization_report_20260320.html` reframes the competition as domain generalization, not source-free domain adaptation.
- The main takeaway is not any single method name; it is the framing:
  source focal clips and Pantanal soundscapes are different domains, and the labeled soundscape subset is a supervised bridge rather than a substitute for the leaderboard.

## Practical Use For Future Sessions

- Use `index.md` first to locate the right reference family.
- Use `evaluation_contract.md` before comparing any local metric to another local metric or to LB.
- Use `raw_inventory.md` and `raw/*.txt` when a later report, planner, or human needs the full historical wording instead of just the distilled theme.
