# Stage 1.0C — Three-File Adapter Guide

## Objective

Convert monthly sales, series aliases, and vehicle attributes into two stable
artifacts: `canonical_monthly_sales.csv` and `canonical_catalog.csv`.

## Resolution rules

The adapter resolves common Chinese and English field aliases, records the exact
source column and resolution method, and rejects ambiguity rather than silently
selecting a weak match. Explicit mappings remain available for unusual files.

Sales values must be finite and non-negative. Months are normalized to the first
day of each month. Brand/series text is normalized before mapping, while original
human-readable values are retained in the catalog.

## Audit outputs

- rejected sales and vehicle rows;
- mapping conflicts;
- unmatched source series;
- duplicate month-key count and configured aggregation policy;
- source and canonical SHA-256 hashes;
- a content-derived `dataset_id`.

`strict_unmatched=True` converts unmatched aliases from an audit condition into
a hard failure. Use it for controlled production sources after mappings stabilize.
