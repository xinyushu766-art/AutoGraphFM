# Stage 11 — Machine Learning Coverage and Graph Falsification

## Why this stage exists

AutoGraphFM already uses machine learning across the forecasting lifecycle, but
"uses a graph neural network" is not itself evidence that vehicle relations are
useful. Stage 11 separates three questions:

1. Can the implementation learn a known dependency graph at all?
2. Does an adaptive graph improve vehicle-level forecasts beyond non-graph and
   falsification controls?
3. Is the evidence scope strong enough to support a research claim?

The graph claim remains blocked unless all three questions are answered
positively.

## Machine-learning layers in AutoGraphFM

### Feature learning and supervised regression

- lagged log sales, rolling means, momentum and seasonal encodings;
- price, brand, segment, body type, energy type and lifecycle features;
- pooled Ridge, autoregressive Ridge and seasonal-residual Ridge;
- NLinear- and DLinear-inspired short-series regressors.

### Deep temporal learning

- shared GRU encoders for all vehicle histories;
- residual forecasts anchored to Last Value;
- early stopping, gradient clipping and multi-seed evaluation;
- validation-only model selection and held-out test measurement.

### Ensemble and regime learning

- robust change-point features;
- growth, stable, decline, transition and shock regimes;
- error-predicting mixture-of-experts weights;
- shadow promotion instead of automatic champion replacement.

### Graph machine learning

- static attribute and train-only demand-correlation graph;
- identity and seeded-permutation falsification graphs;
- AGCRN-style embedding-similarity adjacency;
- MTGNN-style directed Top-K adjacency;
- prior/adaptive graph blending;
- conservative zero-initialized graph residual adapter.

### Probabilistic ML and MLOps

- empirical 80% and 95% intervals;
- drift and forecast-sanity monitoring;
- evidence-scope, deployment and graph-claim gates;
- immutable release packets and realized-outcome reconciliation.

## Stage 11 models

The vehicle-panel experiment executes the following independent controls:

- Last Value;
- residual temporal GRU;
- fixed static graph GRU;
- identity graph GRU;
- permuted graph GRU;
- adaptive AGCRN-style graph GRU;
- adaptive MTGNN-style graph GRU;
- hybrid static plus AGCRN graph GRU;
- hybrid static plus MTGNN graph GRU.

All trainable models use the same history, horizon, hidden dimension, optimizer,
early-stopping protocol and seed roles. This reduces the chance that a graph
model wins only because it receives more training or a different split.

## Architecture identifiability benchmark

A synthetic directed graph generates a multivariate process with known
spillovers. Five controls are compared:

- identity graph;
- permuted graph;
- oracle graph;
- AGCRN embedding graph;
- MTGNN directed embedding graph;
- unconstrained direct sparse graph as a positive identifiability control.

The direct sparse control answers whether the simulated data contain enough
information to recover the edges. If it succeeds while a low-rank embedding
learner fails, the problem is the graph parameterization or optimization—not
an absence of graph signal.

## Claim gate

The best adaptive vehicle model must beat all of the following:

1. residual temporal GRU;
2. fixed static graph GRU;
3. identity graph GRU;
4. permuted graph GRU.

For each comparison, prediction cells are aligned by seed, origin, horizon and
vehicle. A paired bootstrap estimates the WAPE difference. The graph claim is
authorized only when:

- mean WAPE is lower against every control;
- the 95% bootstrap upper bound is below zero against every control;
- the panel evidence scope is explicitly real vehicle-level data.

Demo or synthetic evidence can validate engineering but can never authorize a
vehicle-sales graph claim.

## Current Stage 11 finding

The direct sparse graph recovers 56.9% of the configured Top-K edges, showing
that the benchmark is identifiable. The AGCRN and MTGNN low-rank embedding learners recover 26.4% and 22.2% of the edges respectively. They improve predictive error over some controls but do not recover the configured edge threshold consistently. Therefore:

- engineering status: `GO`;
- low-rank adaptive graph promotion: `NO-GO`;
- real vehicle-panel evaluation: pending the complete 38,806-row local file;
- graph research claim: `NO-CLAIM`.

The next architecture should test a regularized direct sparse graph or a hybrid
where the static automotive prior constrains the learned residual adjacency.
