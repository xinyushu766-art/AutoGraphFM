# Machine-learning methodology

## Forecasting families

- **Statistical baselines:** last value, seasonal naive, moving average, drift, damped trend, seasonal growth.
- **Supervised residual learning:** autoregressive Ridge and seasonal-residual Ridge.
- **Literature-inspired short-series models:** NLinear-style last-value normalization and DLinear-style trend/remainder decomposition.
- **Deep temporal models:** GRU encoders trained on log-residual targets with early stopping, gradient clipping, and multiple random seeds.
- **Regime-aware ensemble:** an error-predicting mixture of experts uses level, slope, volatility, calendar, and change-point features to weight candidate forecasters.

## Graph-learning families

- **Static automotive graph:** brand, energy type, body type, price proximity, and train-period sales correlation.
- **AGCRN-style graph:** node embeddings generate a soft row-normalized adjacency.
- **MTGNN-style graph:** asymmetric node scores produce a directed Top-K adjacency.
- **Sparse residual graph:** learns a row-sparse residual adjacency around a static prior.
- **Stability-selected graph:** repeats train-only graph discovery across seeds, combines discovery frequency with lagged Ridge screening, freezes a candidate support, and retrains on the fixed graph.

## Evaluation controls

Every graph challenger is compared against:

1. temporal-only model;
2. fixed static graph;
3. identity graph;
4. permuted graph;
5. low-rank adaptive graph;
6. simple last-value baseline.

Model promotion uses held-out temporal splits, multiple random seeds, and paired bootstrap confidence intervals. A lower mean error without a stable confidence interval is not treated as a research contribution.
