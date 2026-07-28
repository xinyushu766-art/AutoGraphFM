# Stage 2 — Rolling Backtest and Graph Controls

## Evaluation design

AutoGraphFM uses expanding rolling origins. At every origin, trainable baselines
are refitted using data available at that origin. Horizons are evaluated
separately, and model selection is performed using validation predictions only.
The chosen model is then measured on held-out test predictions.

## Baselines

- Last Value;
- Seasonal Naive 12;
- moving averages over 3 and 6 months;
- linear drift;
- pooled residual Ridge;
- true-graph residual Ridge;
- identity-graph Ridge;
- seeded-permuted-graph Ridge.

## Graph-claim gate

For each horizon the true graph must beat pooled, identity, and permuted controls.
Failure of any comparison yields `NO-CLAIM`; it does not make the pipeline fail.
This prevents model complexity or common temporal signal from being mislabeled as
a graph effect.

Reported outputs include prediction-level rows, aggregate leaderboard, per-series
metrics, validation-selected held-out metrics, and paired bootstrap confidence
intervals against Last Value.
