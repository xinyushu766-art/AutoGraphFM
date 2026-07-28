# Stage 3 — Production Forecast and Calibration

## Selection and refit

Models are selected per horizon by validation WAPE. The test period remains a
one-time held-out measurement and does not participate in selection. After the
workflow is frozen, the selected model is refitted through the latest panel month
and used to forecast future months.

## Intervals

For each selected horizon/model pair, absolute validation residual quantiles are
used to construct empirical symmetric 80% and 95% intervals. Intervals are clipped
at zero and checked for monotonicity:

`0 <= lower_95 <= lower_80 <= prediction <= upper_80 <= upper_95`.

These are empirical error bands, not distributional guarantees. Calibration rows,
sample count, source backtest ID, forecast ID, and hashes are persisted.

## Deployment artifact

`production_forecast.csv` contains series identity, origin, horizon, forecast
month, selected model, point prediction, and both interval levels. `MODEL_CARD.md`
records evidence boundaries and known limitations.
