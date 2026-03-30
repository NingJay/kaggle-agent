# Report Program

Write a report that helps the next decision. Favor root cause, evidence strength, and queue implications over style.

Required output behavior:
- Fill every required schema field. Do not omit `stage`, `headline`, `focus`, `best_run_id`, `best_run_metric`, `primary_metric_value`, `root_cause`, `verdict`, `finding_titles`, `issue_titles`, or `markdown`.
- `headline` should be one short sentence that tells the operator what changed.
- `focus` should name the immediate next focus area, not a generic description.
- `best_run_id` and `best_run_metric` should come from the current leader in the input manifest, or empty string / null if absent.
- `primary_metric_value` must be this run's primary metric from the manifest.
- `finding_titles` and `issue_titles` should be arrays of titles copied from the manifest, not prose paragraphs.
- `markdown` should summarize the situation in operator-facing language and stay consistent with the structured fields.
