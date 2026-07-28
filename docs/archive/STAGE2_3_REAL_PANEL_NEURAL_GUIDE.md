# Stage 2.3 — Real-Panel Neural Controls

## Models

The neural suite evaluates Last Value, residual temporal GRU, safe static-graph
GRU, identity-graph GRU, and permuted-graph GRU under matched data, split, loss,
and seed conditions.

The residual models predict corrections to a strong temporal reference rather
than forecasting raw sales without an anchor. Loss and evaluation are masked to
original observed target cells. Training uses early stopping, gradient clipping,
weight decay, and persisted per-seed predictions.

## Claim policy

The static graph must beat Last Value, temporal, identity, and permuted controls.
The winner column alone is insufficient: if a permuted graph wins, or if the true
graph does not beat the temporal model, the neural graph claim remains false.

Multi-seed full mode is required for research evidence. Single-seed quick mode is
only an engineering acceptance path.
