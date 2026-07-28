# Interview guide

## 60-second explanation

> AutoGraphFM is an end-to-end automotive demand forecasting project. I built the pipeline from messy bilingual CSV ingestion and data-quality auditing through leakage-safe monthly panels, rolling backtests, model selection, calibrated forecasts, drift monitoring, and release governance. On the modeling side, I compared statistical baselines, Ridge models, GRUs, regime-aware ensembles, and several graph-learning strategies. I also built identity and permuted graph controls plus paired bootstrap tests, so a graph model is not credited merely because it has more parameters. The project has 108 automated tests. On public China market data, the literature-informed challenger improved four of six forecasting tasks, with a 9.81% relative WAPE improvement on three-month EV forecasting. On a controlled graph-recovery benchmark, stability selection reached 83.3% Precision@K and perfect cross-seed edge consistency, while I still kept the real vehicle graph conclusion as NO-CLAIM because the full vehicle panel has not yet passed the final evaluation.

## Likely questions

### Why use a graph?
Vehicle series can share brand, price band, body type, energy technology, and market substitution patterns. A graph allows message passing across related series, but the project requires graph controls to verify that any gain is relation-specific.

### How did you prevent leakage?
All features, lifecycle eligibility, graph candidate screening, and model selection are computed at or before each forecast origin. Test-period survival and future launches are audit fields only.

### Why did simple baselines remain important?
Short monthly series often make deep models unstable. Last Value and Seasonal Naive reveal whether architectural complexity creates genuine value.

### What was the hardest technical issue?
Adaptive low-rank graphs produced weak edge recovery and unstable relations. I replaced them with direct sparse residual graphs and stability selection, improving Precision@K to 83.3% and cross-seed Jaccard to 1.0.

### What would you do next?
Run the locked multi-fold protocol on the complete vehicle-month panel, add lifecycle-specific cold-start evaluation, and promote a graph model only if it beats all temporal and graph falsification controls.
