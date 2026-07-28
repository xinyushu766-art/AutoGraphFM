# Stage 11B — Regularized Sparse Residual Graph

## Motivation

Stage 11 showed that direct sparse adjacency could recover a known graph, while
AGCRN/MTGNN low-rank embedding parameterizations could not meet the configured
edge-recovery threshold. Stage 11B therefore changes the graph parameterization
instead of increasing network depth.

## Model

The final adjacency is a conservative blend of a static automotive prior and a
learned direct Top-K residual graph:

\[
A = (1-g)A_{\text{static}} + gA_{\text{residual}},
\qquad g=\sigma(\eta).
\]

The residual graph is parameterized directly by an \(N\times N\) score matrix,
with self-edges masked and only the row-wise Top-K edges retained. Training adds:

- an L1 penalty on residual scores;
- a deviation penalty from the static prior;
- a gate penalty discouraging unnecessary residual influence.

The forecasting path remains residual and safe: both the graph correction and
forecast head start at zero, so initialization equals Last Value.

## Stability selection

Each seed exports its learned adjacency. Stage 11B computes:

- Top-K edge sets per seed;
- pairwise edge Jaccard;
- edge selection frequency;
- consensus adjacency at a configured frequency threshold;
- stable-edge ratio over the union of candidate edges.

## Claim gate

The sparse challenger must beat:

1. temporal GRU;
2. fixed static graph GRU;
3. identity graph GRU;
4. permuted graph GRU;
5. MTGNN-style low-rank graph GRU.

Each paired-bootstrap 95% upper bound must be below zero, and the selected graph
must pass cross-seed stability thresholds. Demo evidence can never authorize a
real vehicle graph claim.

## Current result

- sparse residual known-graph recovery: GO;
- low-rank known-graph recovery: NO-GO;
- regularized residual edge precision@K: 0.500;
- architecture eligible for real vehicle challenge: YES;
- demo vehicle graph claim: NO-CLAIM;
- full real vehicle graph experiment: NOT_EVALUATED.
