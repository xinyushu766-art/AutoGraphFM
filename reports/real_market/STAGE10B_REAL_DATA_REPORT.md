# AutoGraphFM v0.15.0 Stage 10B Report

- Run ID: `b725281e17a3fbf6`
- Engineering: **PASS**
- Promotion: **promote_to_shadow_champion**
- New models selected in at least one fold: `['nlinear_ridge', 'regime_error_moe']`
- Vehicle-level graph claim: **NO-CLAIM**

## Real aggregate comparison

| Target | H | Stage 10A WAPE | Stage 10B WAPE | Relative change |
|---|---:|---:|---:|---:|
| EV_Sales | 1 | 19.55% | 18.82% | +3.72% |
| EV_Sales | 3 | 27.49% | 24.80% | +9.81% |
| EV_Sales | 6 | 29.76% | 29.76% | +0.00% |
| Total_Sales | 1 | 14.26% | 14.90% | -4.47% |
| Total_Sales | 3 | 16.11% | 15.63% | +2.96% |
| Total_Sales | 6 | 16.68% | 16.43% | +1.46% |

## Adopted literature mechanisms

- NLinear-inspired last-value normalization.
- DLinear-inspired moving-average trend/remainder decomposition.
- Window change-point features for level, slope, and volatility shifts.
- TFT/TiDE-inspired known-calendar features and lightweight expert gate.
- AGCRN-style soft adjacency and MTGNN-style directed Top-K adjacency.

## Boundary

Literature mechanisms are implemented and aggregate-tested. Adaptive graph code is structural only until the vehicle panel runs.
