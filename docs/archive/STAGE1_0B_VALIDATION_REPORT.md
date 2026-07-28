# Stage 1.0B Validation Report

## Baseline package reconciliation

The uploaded full package already contained every database-patch file with
identical content. The full package was therefore used as the single baseline;
no duplicate patch overlay was required.

## Automated tests

```text
python -m pytest -q
48 passed
```

The test set covers the previous Stage 0 and Stage 1.0A functionality plus:

- Chinese and English header resolution;
- month and integer parsing;
- canonical catalog and source-alias creation;
- repeat-import idempotency;
- invalid-row and duplicate-row isolation;
- persistent ingestion-run and quality-result audits;
- chronological split boundaries;
- minimum-history and continuity filters;
- deterministic CSV hashes and snapshot IDs;
- explicit failure for empty splits.

## Compile check

```text
python -m compileall -q src scripts tests
```

Result: passed.

## Static-quality limitation

`ruff check .` could not be executed in this sandbox because the internal Python
package index returned HTTP 503 while resolving `ruff`. This is an environment
limitation, not a passing result. As substitutes, all Python files passed
`compileall`, CLI help entry points loaded, and the repository-wide 88-character
line-length and trailing-whitespace scan passed. Run `ruff check .` in the local
project environment before committing.

## End-to-end SQLite acceptance

The acceptance workflow used `examples/stage1_0b/monthly_sales_sample.csv`.

First import:

```text
records_read      16
records_written   16
records_skipped    0
records_rejected   0
status             succeeded
```

Second identical import:

```text
records_read      16
records_written    0
records_skipped   16
records_rejected   0
status             succeeded
```

Database counts after both runs:

```text
brands                        1
vehicle_series                2
vehicle_aliases               2
ingestion_runs                2
monthly_sales                16
monthly_series_features       0
data_quality_results          8
```

Snapshot result:

```text
train rows        8   (2024-01 through 2024-04)
validation rows   4   (2024-05 through 2024-06)
test rows         4   (2024-07 through 2024-08)
included series   2
excluded series   0
snapshot_id       48b2f6b181abf03e
```

## Remaining external acceptance

A real PostgreSQL container was not required for the new logic because the
existing migration and PostgreSQL DDL tests remain in the full suite. Before
using production data, run the documented Docker PostgreSQL sequence locally.

The exact schemas of `sales_aligned_completed.csv`,
`series_mapping_completed.csv`, and `vehicles_completed.csv` were not present in
this delivery. Stage 1.0C integration against those files therefore remains
schema-dependent rather than guessed.
