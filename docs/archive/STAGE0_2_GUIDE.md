# Stage 0.2 Guide: Controlled Graph Ablation

## 1. Research question

Does a meaningful vehicle relation graph improve demand forecasting beyond a
temporal-only model?

This question cannot be answered by training only one graph model. We need
controlled baselines.

## 2. Models

### Last Value

\[
\hat{y}_{t+h}=y_t
\]

It has no trainable parameters. A complex neural model should normally beat this
baseline before we consider it useful.

### Temporal-only GRU

Each vehicle is processed independently, but all vehicles share the same GRU
parameters:

\[
h_i = \operatorname{GRU}(X_i)
\]

\[
\hat{Y}_i = W_o h_i
\]

### Static Graph GRU

The temporal state is combined with graph context:

\[
H_g=\hat{A}H_t
\]

\[
g=\sigma(W_g[H_t;H_g])
\]

\[
H_f=g\odot H_t+(1-g)\odot H_g
\]

The symbol \(\odot\) means element-wise multiplication.

## 3. Graph controls

### Correct relation graph

The Stage 0 synthetic graph uses segment and price similarity.

### Identity graph

Only self-relations remain. No vehicle receives information from another
vehicle.

### Permuted graph

\[
A_{\mathrm{perm}} = P A P^T
\]

This keeps the same weights, sparsity, symmetry, and broad degree statistics, but
assigns the relation pattern to the wrong vehicle identities.

## 4. Metrics

Training uses MSE in log-sales space. Evaluation first restores sales with
`expm1`, then reports MAE, RMSE, WAPE, and sMAPE.

## 5. Interpretation

A useful Stage 0 graph result should satisfy most of the following:

1. Static Graph GRU beats Last Value.
2. Static Graph GRU beats Temporal-only GRU.
3. Static Graph GRU beats Permuted Graph GRU.
4. Static Graph GRU differs meaningfully from Identity Graph GRU.

One seed is only a smoke experiment. Later stages require multiple seeds and
uncertainty reporting.

## 6. Code-reading questions

1. Why does Last Value have zero parameters?
2. Why do all trainable models reset the same random seed before construction?
3. Why is the graph permutation applied to both rows and columns?
4. Why does the runner use the same train and validation loaders for all models?
5. Why are business metrics computed after `expm1`?
6. Why can Identity Graph GRU differ from Temporal-only GRU even though neither
   receives information from other vehicles?
