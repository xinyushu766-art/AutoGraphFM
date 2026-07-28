# Stage 0.1 Architecture

## 1. Problem formulation

Let:

- \(B\): batch size
- \(L\): historical window length
- \(N\): number of vehicle models
- \(F\): number of features
- \(D\): hidden dimension
- \(H\): forecast horizon

The input tensor is:

\[
X \in \mathbb{R}^{B \times L \times N \times F}
\]

The competition graph is:

\[
A \in \mathbb{R}^{N \times N}
\]

The model predicts future log-sales:

\[
\hat{Y} \in \mathbb{R}^{B \times H \times N}
\]

## 2. Data flow

```text
X [B,L,N,F]
    │
    ├─ rearrange nodes into the batch dimension
    ▼
[B·N,L,F]
    │
    ├─ shared GRU
    ▼
Temporal state [B,N,D]
    │
    ├─ normalized graph aggregation
    ▼
Graph state [B,N,D]
    │
    ├─ gated residual fusion
    ▼
Fused state [B,N,D]
    │
    ├─ linear forecast head
    ▼
Prediction [B,H,N]
```

## 3. Why one GRU is shared across vehicles

A separate GRU for every vehicle cannot naturally generalize to a new model.
Sharing the temporal encoder makes the architecture inductive: every node is
processed by the same function, while its own history determines its state.

## 4. Graph normalization

The implementation uses:

\[
\tilde{A} = A + I
\]

\[
\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}
\]

Adding \(I\) preserves each vehicle's own state. Symmetric degree
normalization prevents high-degree nodes from producing unbounded aggregated
representations.

## 5. Message passing

For hidden node states \(H_t \in \mathbb{R}^{B\times N\times D}\):

\[
H_g = \hat{A}H_t
\]

The code uses:

```python
torch.einsum("ij,bjd->bid", normalized_adjacency, temporal_state)
```

For vehicle \(i\), this computes a weighted combination of the hidden states of
all connected vehicles \(j\).

## 6. Gated fusion

The model does not assume graph information is always useful.

\[
g = \sigma(W_g[H_t;H_g])
\]

\[
H_f = g \odot H_t + (1-g)\odot H_g
\]

A gate value close to one favors the vehicle's own temporal state. A value close
to zero favors information aggregated from competitors.

## 7. Current limitations

- The graph is static.
- There is only one node type.
- The graph is manually constructed rather than learned.
- Text is represented by a scalar sentiment feature.
- The output is deterministic rather than probabilistic.
- The model does not handle unseen node attributes explicitly.
- There is no causal identification.

These limitations are deliberate. Each one becomes a later research stage.
