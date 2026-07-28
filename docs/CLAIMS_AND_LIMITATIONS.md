# Claims and limitations

## Supported claims

- The engineering pipeline is reproducible and covered by 108 automated tests.
- The public aggregate benchmark is a real time-series evaluation.
- Literature-inspired challengers improve four of six aggregate target/horizon cells.
- Stability selection improves known-graph edge recovery and cross-seed consistency.

## Unsupported claims

- The project does not claim a globally optimal forecasting model.
- It does not claim causal effects between vehicle series.
- It does not claim that graph learning beats all temporal baselines on real vehicle-level data.
- Demo data is not presented as business performance.
- Production release is not approved without user data, monitoring gates, and realized-outcome validation.

## Main remaining limitation

The complete 38,806-row vehicle-month panel is not bundled. A final vehicle-level graph claim requires running the locked protocol on that panel and beating temporal, static, identity, and permuted controls across multiple time folds.
