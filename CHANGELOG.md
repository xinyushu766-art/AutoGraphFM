# Changelog

## 0.18.0 — Stage 12A stability-selected consensus graph

- added repeated train-only graph discovery and cross-seed edge-frequency scoring;
- added lagged-ridge and static-prior candidate screening with fixed Top-K support;
- added consensus adjacency retraining and stability-selected sparse residual GRUs;
- raised controlled graph recovery precision@K to 83.3% with final edge Jaccard 1.0;
- added seven-control paired Bootstrap and evidence-scope-aware graph authorization;
- retained demo vehicle forecasting as `NO-CLAIM` because Last Value remains stronger;
- documented rejected common-factor and single-validation selection experiments;
- validated Stage 12A without using test data for graph discovery or candidate selection.

## 0.17.0 — Stage 11B sparse residual graph

- added a direct Top-K residual graph around the static automotive prior;
- added L1, prior-deviation and residual-gate regularization;
- added seed-wise edge stability, consensus-frequency and consensus-graph artifacts;
- added five-control paired Bootstrap against temporal, static, identity, permuted and MTGNN models;
- separated architecture eligibility from real vehicle graph-claim authorization;
- revalidated v0.16 independently with 98/98 tests and a fresh Stage 11 run;
- validated 104 tests and the Stage 11B 10/10 acceptance gate.

## 0.16.0 — Stage 11 adaptive graph falsification

- added adaptive AGCRN/MTGNN residual GRUs and static-prior hybrids;
- added learned-adjacency diagnostics and prediction-cell paired bootstrap;
- added synthetic known-graph recovery with a direct sparse positive control;
- blocked low-rank graph promotion when edge recovery fails despite identifiable data;
- added evidence-scope-aware vehicle graph claim authorization;
- validated 98 tests and the Stage 11 10/10 acceptance gate.

## 0.15.0 — Stage 10B

- added NLinear- and DLinear-inspired short-series Ridge challengers;
- added leakage-safe change-point regime features and error-predicting MoE gate;
- added AGCRN soft adjacency and MTGNN directed Top-K adjacency scaffolds;
- added real Stage 10A versus Stage 10B comparison and promotion gate;
- promoted literature models to shadow challengers after improving 4/6 cells;
- retained graph `NO-CLAIM` until the full vehicle panel is executed;
- validated 93 tests plus Stage 10B, Stage 10A, Stage 4, and Stage 6 gates.

## 0.14.0 — Stage 10A leakage-safe lifecycle and strong baselines

### Added

- train-cutoff-only panel eligibility and lifecycle cohort audit;
- damped trend, seasonal growth, regime-guarded, and seasonal-residual Ridge baselines;
- annual multi-window real-market evaluation with selection-stability reporting;
- combined EV and total-market Stage 10A reports and acceptance workflow.

### Fixed

- series eligibility no longer uses validation/test observations by default;
- production forecasting supports all newly selectable Stage 10A panel baselines.

### Validation

- 86 automated tests pass;
- real 2018-01 through 2024-04 aggregate benchmark completes across three annual folds;
- graph evidence remains `NO-CLAIM` until the vehicle-level panel is executed.

## 0.12.0 — Stage 8.0 SRNI authenticity and evidence governance

### Added

- pre-training SRNI source inspection with local SHA-256 and published-profile checks;
- explicit `official_real_data`, `real_unverified`, and `synthetic_only` scopes;
- strict official runner and engineering-only nonofficial override;
- synthetic fixture detection and release blocking;
- two-endpoint downloader attempt logs, standalone file validator, and PowerShell helper;
- Stage 8 acceptance script and provenance documentation.

### Fixed

- official-schema synthetic acceptance data can no longer be labeled `user_data`;
- generic SRNI source inference defaults to `real_unverified` unless inspection is passed explicitly.

### Validation

- 79 automated tests pass;
- strict mode rejects the synthetic Stage 7 fixture before training;
- engineering mode completes with `synthetic_only` and is not release-ready;
- real SRNI accuracy remains not evaluated until the official Sales CSV is available.

## 0.11.0 — Stage 7.0 SRNI-CAR real-data onboarding

### Added

- direct conversion of the official SRNI-CAR single Sales CSV into the existing
  AutoGraphFM three-file contract;
- support for separate `Year`/`Month` fields and numeric, English, abbreviated,
  or Chinese month labels;
- deterministic vehicle catalog and identity mapping generation;
- row-level rejection, duplicate-key counts, static-attribute conflict audits,
  SHA-256 outputs, and a content-derived SRNI dataset ID;
- guarded Figshare downloader with explicit manual-download fallback;
- one-command SRNI preparation plus Stage 4 real-data evaluation entry point;
- Stage 7 guide and two integration tests.

### Validation

- 76 automated tests pass;
- Python compilation passes;
- SRNI fixture conversion enters the unchanged three-file canonicalizer with
  zero unmatched rows;
- real SRNI accuracy remains not evaluated until the official Sales CSV is
  present locally.

## 0.10.0 — Stage 1.0C through Stage 6.0

### Added — real-data contract

- adaptive Chinese/English three-file column resolution;
- canonical series mapping with conflict and unmatched-row reporting;
- bad-row quarantine and explicit duplicate-month policies;
- source and output SHA-256 manifests with content-derived dataset IDs.

### Added — leakage-safe panel and graph

- fixed monthly node universe with history, missingness, and staleness gates;
- structural-zero and internal-gap handling with original-observation masks;
- train-only feature scaling and lagged/rolling feature construction;
- static and train-only correlation relation graph;
- identity and seeded-permutation falsification graphs;
- graph connectivity, density, degree, and edge diagnostics.

### Added — Stage 2 evaluation

- expanding rolling-origin evaluation for 1/3/6-month horizons;
- last-value, seasonal-naive, moving-average, drift, pooled Ridge, and graph Ridge;
- observed-target-only WAPE/MAE/RMSE/bias metrics;
- validation-only model selection and held-out test measurement;
- paired bootstrap confidence intervals against Last Value;
- per-series metrics and prediction-level audit tables.

### Added — Stage 2.3 neural controls

- real-panel residual temporal GRU;
- safe static, identity, and permuted graph residual GRUs;
- multi-seed aggregation, early stopping, clipping, predictions, and claim gates.

### Added — Stage 3 production forecast

- refit through the latest available month after model selection;
- future 1/3/6-month forecasts;
- validation-residual empirical 80% and 95% prediction intervals;
- calibration manifest and deployable model card.

### Added — Stage 4 release governance

- global PSI and recent missingness monitoring;
- per-series level-shift and missingness severity;
- forecast ratio and interval sanity checks;
- explicit pass/warn/block deployment decision;
- standalone HTML audit and Markdown summary;
- one-command Stage 4 runner and conservative release acceptance suite.

### Added — Stage 5 governed release

- production and shadow release modes;
- explicit evidence-scope, monitoring, and optional graph-claim policy;
- no deployment bundle when production requirements fail;
- immutable bundle checksums and release-readiness decision.

### Added — Stage 6 outcome feedback

- realized-sales matching by canonical series and target month;
- WAPE, MAE, RMSE, bias, and 80%/95% interval coverage;
- match-completeness and performance gates;
- champion actions for retain, investigation/shadow challengers, or retrain/rollback.

### Validation

- 74 automated tests pass;
- Stage 4 acceptance checks pass 14/14;
- Stage 6 release and feedback checks pass 7/7;
- demo run is correctly marked `deployment=block`, `graph=no-claim`, and
  `evidence_scope=demo_only` rather than being promoted as real-data evidence.

## 0.4.0 — Stage 1.0B

- audited bilingual monthly-sales CSV ingestion;
- row-level rejection with persistent ingestion and quality audit records;
- idempotent immutable business-key handling;
- deterministic chronological snapshot builder;
- CSV and explicit optional Parquet output;
- content-derived snapshot IDs and SHA-256 manifests.


## Portfolio-ready GitHub edition

- Reorganized public documentation and evidence.
- Added CI, licensing, citation, contribution, and security files.
- Corrected project authorship to Xinyu Shu.
- Removed environment-specific audit paths and redundant release files.
