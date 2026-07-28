# Stage 0.2D: Residual Forecasting

## 1. Why absolute prediction was difficult

The previous neural models directly predicted future log-sales:

\[
\hat{y}_{t+h}^{\log}=f_\theta(X_{1:t})
\]

At initialization, a neural output is often near zero. After inversion:

\[
\exp(0)-1=0
\]

The network therefore started from an unrealistic near-zero sales forecast,
while Last Value already represented the strong persistence in the synthetic
market.

## 2. Residual formulation

The new model predicts a correction:

\[
\hat{y}_{t+h}^{\log}
=
y_t^{\log}
+
\Delta_{t+h}^{\log}
\]

The delta head is initialized to zero:

\[
W_\Delta=0,\qquad b_\Delta=0
\]

Therefore, before training:

\[
\Delta=0
\]

and:

\[
\hat{y}_{t+h}^{\log}=y_t^{\log}
\]

Every trainable model starts exactly from Last Value.

## 3. Temporal residual model

\[
H_t=\operatorname{GRU}(X)
\]

\[
\Delta=W_\Delta H_t
\]

\[
\hat{Y}=Y_{\text{last}}+\Delta
\]

## 4. Graph residual model

\[
H_g=\hat{A}H_t
\]

\[
g=\sigma(W_g[H_t;H_g])
\]

\[
H_f=g\odot H_t+(1-g)\odot H_g
\]

\[
\Delta=W_\Delta H_f
\]

\[
\hat{Y}=Y_{\text{last}}+\Delta
\]

## 5. Why the zero head still trains

At the first optimization step, the zero head receives a gradient and updates.
The upstream encoder initially receives little or no gradient through the zero
head. Once the head is non-zero, gradients propagate to the GRU and graph
layers. This makes the first updates conservative and keeps the starting
forecast stable.

## 6. Diagnostics

The experiment reports:

- WAPE, MAE, and RMSE;
- mean predicted sales;
- mean target sales;
- mean prediction bias;
- prediction and target standard deviations;
- relative WAPE improvement over Last Value.

The relative improvement is:

\[
100
\times
\frac{
\operatorname{WAPE}_{LV}
-
\operatorname{WAPE}_{model}
}{
\operatorname{WAPE}_{LV}
}
\]

Positive values mean improvement. Negative values mean deterioration.

## 7. Interpretation rules

A convincing graph result should satisfy:

1. Residual Static Graph GRU beats Last Value.
2. Residual Static Graph GRU beats Residual Temporal GRU.
3. Residual Static Graph GRU beats Residual Identity Graph GRU.
4. Residual Static Graph GRU beats Residual Permuted Graph GRU.
5. The result remains stable across multiple seeds.

Stage 0.2D initially checks one seed. Multi-seed evaluation comes next.
