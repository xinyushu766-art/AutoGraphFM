# Stage 4 — Monitoring, Release Gate, and Audit

## Monitoring dimensions

- global population stability index for train versus recent log sales;
- recent global missingness;
- per-series mean-level shift in training standard deviations;
- per-series recent missingness;
- point forecast validity and forecast-to-last-sales ratio;
- 80% and 95% interval ordering;
- rolling and neural graph falsification status.

Thresholds are explicit in `MonitoringSpec`. The operational result is one of:

- `pass`: no threshold breach;
- `warn`: deploy only with documented review;
- `block`: do not release the forecast without resolving or accepting the cause.

## Separate decisions

Stage 4 deliberately separates:

1. **engineering status** — whether the complete workflow executed and validated;
2. **deployment status** — whether current data and forecasts pass operational gates;
3. **graph research claim** — whether graph controls support the claimed effect;
4. **evidence scope** — demo-only or user data;
5. **release readiness** — requires non-demo evidence and a non-blocking gate.

A green engineering result cannot override a red deployment result. A green
deployment result cannot authorize a graph-effect research statement.

## Audit output

`08_report/AUDIT_REPORT.html` is standalone and includes lineage, quality, held-out
metrics, graph controls, monitoring, and forecast samples. It uses no external
assets and can be opened directly in a browser.
