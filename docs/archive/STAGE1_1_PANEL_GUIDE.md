# Stage 1.1 — Leakage-Safe Monthly Panel

## Panel contract

The panel uses a fixed ordered vehicle-series node universe and a complete monthly
calendar. It stores sales, original-observation mask, availability mask, catalog,
feature tensor, feature names, and temporal split boundaries.

## Inclusion gates

A series must satisfy minimum history, maximum internal missingness, and maximum
trailing staleness. Exclusions are written with reasons. Values outside a series'
observed lifespan are structural zeros; eligible internal gaps are imputed by the
configured method but remain false in the original-observation mask.

## Leakage controls

Rolling means and momentum at month `t` use information available no later than
`t`. Standardization statistics are fitted only through `train_end`. Window split
membership is determined by target-end month, preventing a multi-horizon target
from crossing train, validation, or test boundaries.

Model metrics default to original observed targets only. Imputation supports
features and tensor continuity; it does not manufacture evaluation truth.
