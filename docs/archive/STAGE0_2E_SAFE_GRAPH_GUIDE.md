# Stage 0.2E: Safe Graph Adapter and Held-Out Testing

## 1. What the previous result showed

At 30 epochs:

- residual temporal GRU beat Last Value;
- all graph models became more negatively biased;
- correct graph beat the permuted graph;
- graph models still lost to the temporal model.

This suggests that relation structure may contain information, but the old
fusion architecture used it too aggressively.

## 2. Conservative graph adapter

The temporal representation is:

\[
H_t=\operatorname{GRU}(X)
\]

The neighbor representation is:

\[
H_n=\hat{A}H_t
\]

The graph correction is:

\[
C_g=\phi(H_n-H_t)
\]

The final hidden state is:

\[
H_f=H_t+\tanh(\alpha)\odot C_g
\]

where:

\[
\alpha=0
\]

at initialization. Therefore:

\[
H_f=H_t
\]

before learning. The graph branch must earn influence through training.

## 3. Residual forecast

\[
\hat{Y}
=
Y_{\mathrm{last}}
+
W_\Delta H_f
\]

The delta head is also initialized to zero, so the complete untrained model
equals Last Value.

## 4. Why train, validation, and test are separate

- training data updates parameters;
- validation data selects the best epoch;
- test data is used once for final reporting.

Using the same validation set for both early stopping and final claims would
produce optimistic results.

## 5. Purged boundaries

Sliding windows overlap. A gap of:

\[
H-1
\]

windows is inserted at each split boundary, where \(H\) is the forecast horizon.
This prevents the same target month from appearing in two subsets.

## 6. Early stopping

After every epoch, validation log-MSE is measured. The best model state is
saved. Training stops after `patience` consecutive epochs without sufficient
improvement.

This is especially important because the old graph models deteriorated between
epochs 5 and 30.

## 7. Graph scale

The reported graph scale is:

\[
\operatorname{mean}(|\tanh(\alpha)|)
\]

Interpretation:

- near zero: graph context was mostly ignored;
- moderate value: graph correction contributed;
- large value: graph branch strongly changed the temporal state.

A large graph scale is not automatically good. It must coincide with lower
held-out test error.
