# Stage 6 — Realized-Outcome Feedback

## When to run

Run Stage 6 only after a forecast target month has closed and canonical realized
sales for that month are available. It matches each forecast by `series_key` and
`target_month`; unmatched forecasts remain visible and reduce the match ratio.

## Metrics

Metrics are grouped by horizon and selected model:

- WAPE;
- MAE and RMSE;
- signed bias;
- empirical 80% interval coverage;
- empirical 95% interval coverage;
- matched-cell count.

Duplicate actual series/month keys are a hard error because they make realized
truth ambiguous.

## Champion action

The combined outcome gate maps to one operational action:

- `retain_champion`: match completeness, error, and coverage pass;
- `investigate_and_shadow_challengers`: one or more warning thresholds breach;
- `retrain_revalidate_or_rollback`: blocking error, coverage, or completeness.

The outcome report does not automatically retrain or overwrite a production
forecast. It creates an auditable decision input for the next governed run.

## Evidence boundary

The Stage 6 validation suite generates simulated healthy and stress actuals only
to test the mechanism. Those outcomes are not real automotive accuracy results.
