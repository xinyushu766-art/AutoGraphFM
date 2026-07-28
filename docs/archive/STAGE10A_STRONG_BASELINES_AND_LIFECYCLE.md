# Stage 10A: Strong Baselines, Lifecycle, and Multi-window Evaluation

## Why this stage exists

Increasing neural-network depth before establishing stable non-graph baselines
can hide data leakage and unstable model selection. Stage 10A therefore changes
the evaluation foundation before architectural work continues.

## Leakage-safe eligibility

Panel inclusion now defaults to `eligibility_basis="train"`. History length,
internal missingness, and staleness are evaluated only through `train_end`.
Future validation/test survival can no longer determine which nodes enter the
model.

`lifecycle_at_train_end` is safe for model features:

- `established_at_train_end`;
- `new_launch_at_train_end`;
- `intermittent_at_train_end`;
- `inactive_at_train_end`;
- `not_observed_by_train`.

`outcome_status_at_test_end` is audit-only and must not be used as a feature.

## Strong non-graph baselines

- Last Value and Seasonal Naive 12;
- Moving Average 3/6 and endpoint Drift;
- robust log-scale Damped Trend;
- Seasonal Growth with damped annual change;
- direct autoregressive Ridge;
- Seasonal Residual Ridge;
- Regime-Guarded blend;
- Robust Ensemble for the aggregate benchmark.

## Multi-window protocol

For the 2018–2024.4 aggregate series, Stage 10A creates three expanding annual
folds:

1. train through 2020, validate 2021, test 2022;
2. train through 2021, validate 2022, test 2023;
3. train through 2022, validate 2023, test 2024.1–2024.4.

Each fold selects a model using validation WAPE and measures it on the held-out
rolling test period. The report exposes winner changes across folds.

## Run

```cmd
python scripts\run_stage10a.py ^
  --input data\china_auto_market\ev_sales_monthly.csv ^
  --output-dir outputs\stage10a_real ^
  --overwrite
```

Acceptance:

```cmd
python scripts\validate_stage10a.py --overwrite
```

## Research boundary

Aggregate-market results can justify regime-aware modeling and stronger
exogenous features. They cannot establish a vehicle-relation graph effect.
