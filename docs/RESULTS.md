# Results and evidence scopes

## 1. Public real aggregate automotive data

The bundled aggregate benchmark covers monthly China automotive market data from January 2018 through April 2024. It supports real time-series claims, but not vehicle-level graph claims.

| Target | Horizon | Incumbent WAPE | Literature-informed WAPE | Relative change |
|---|---:|---:|---:|---:|
| EV sales | 1 | 19.55% | **18.82%** | **+3.72%** |
| EV sales | 3 | 27.49% | **24.80%** | **+9.81%** |
| EV sales | 6 | 29.76% | 29.76% | 0.00% |
| Total market | 1 | **14.26%** | 14.90% | -4.47% |
| Total market | 3 | 16.11% | **15.63%** | **+2.96%** |
| Total market | 6 | 16.68% | **16.43%** | **+1.46%** |

Interpretation: the new challenger set improved four of six cells, remained unchanged in one, and regressed in one. It was promoted to a shadow challenger rather than a universal champion.

## 2. Controlled graph identifiability

The known-graph benchmark verifies whether the learning procedure can recover a planted directed dependency graph.

| Metric | Stability-selected result |
|---|---:|
| Edge Precision@K | **83.3%** |
| Edge Recall@K | **83.3%** |
| Final edge Jaccard across seeds | **1.00** |

## 3. Demo vehicle panel

The stability-selected static-residual graph significantly beat the temporal, identity, and permuted controls, but did not beat Last Value or the fixed static graph. The correct research conclusion is therefore `NO-CLAIM`.

## Evidence hierarchy

| Scope | Valid use |
|---|---|
| Controlled synthetic graph | Architecture identifiability and edge recovery |
| High-fidelity demo vehicle panel | Engineering and falsification-path validation |
| Public real aggregate data | Real market-level time-series evaluation |
| Complete real vehicle panel | Required for vehicle-level graph contribution |
