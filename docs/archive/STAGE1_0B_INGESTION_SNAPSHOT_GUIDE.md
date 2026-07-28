# Stage 1.0B: Audited Ingestion and Model Snapshots

## 1. Objective

Stage 1.0A established the relational schema. Stage 1.0B closes the first real
machine-learning data loop:

```text
Source CSV
    │
    ├─ header resolution
    ├─ row parsing and contract validation
    ├─ canonical brand/series resolution
    ├─ row-level rejection
    └─ ingestion audit
          │
          ▼
PostgreSQL monthly_sales
          │
          ├─ explicit source
          ├─ explicit data_version
          └─ immutable business key
                │
                ▼
Chronological snapshot builder
                │
                ├─ history filters
                ├─ train/validation/test split
                ├─ SHA-256 file hashes
                └─ content-derived snapshot_id
```

## 2. CSV contract

Required canonical fields:

| Canonical field | Required | Examples recognized automatically |
|---|---:|---|
| `brand` | yes, unless `default_brand` is set | `brand`, `品牌`, `厂商` |
| `series` | yes | `series_name`, `vehicle_series`, `车系`, `车型` |
| `month` | yes | `month`, `year_month`, `年月`, `销售月份` |
| `sales` | yes | `sales`, `sales_volume`, `销量`, `月销量` |

Optional fields include `canonical_series`, `sales_rank`, `country`, `segment`,
`body_type`, `energy_type`, and `status`.

Header matching is deterministic. If two columns both match one canonical field,
the import fails at the CSV contract gate and requires `--column` overrides.
There is no arbitrary first-column selection.

## 3. Parsing rules

Months may use `YYYY-MM`, `YYYY/MM`, `YYYY-MM-DD`, `YYYY年M月`, or
`YYYY年M月D日`. They are normalized to the first day of the month.

Sales values must be non-negative integers. Thousands separators and integral
decimal strings such as `12,345` or `120.0` are accepted. Fractional sales,
negative values, empty values, and booleans are rejected.

Status values map to the four database states:

```text
active / discontinued / upcoming / unknown
```

Common Chinese values such as `在售`, `停售`, `停产`, and `预售` are mapped
explicitly.

## 4. Entity resolution

The importer first resolves a source-specific approved alias. If no alias exists,
it creates or reuses the canonical brand and series, then records an exact alias.

A source alias that already points to a series under a different brand is
rejected. Automatic fuzzy matching is deliberately excluded from this stage.
Ambiguous mappings belong in a reviewed mapping workflow rather than a silent
heuristic.

## 5. Idempotency and versioning

The monthly-sales business key is:

```text
series_id + month + source + data_version
```

When the same key already exists, the row is counted as `records_skipped`; it is
not overwritten. Corrections require a new explicit `data_version`.

This rule prevents a repeated crawler run or notebook execution from silently
changing the training source of truth.

## 6. Audit semantics

Every import creates an `ingestion_runs` row with:

- input filename and SHA-256;
- source and data version;
- resolved column mapping;
- records read, written, skipped, and rejected;
- final status: `succeeded`, `partial`, or `failed`.

Four quality results are persisted:

1. CSV-level contract;
2. row-level contracts;
3. duplicate business keys inside the input file;
4. idempotent existing-row skips.

Expected bad rows are isolated. Unexpected programming or database-operational
errors are raised rather than converted into normal rejections.

## 7. Chronological snapshot contract

For boundaries `t_train`, `t_validation`, and optional `t_test`:

```text
train:       month <= t_train
validation:  t_train < month <= t_validation
test:        t_validation < month <= t_test
```

The builder refuses empty train, validation, or test splits. Dates must use the
first day of a month.

Optional history gates are applied before splitting:

- minimum number of monthly observations;
- complete month-to-month continuity.

Excluded series and reasons are written to the manifest.

## 8. Snapshot artifact

Each snapshot contains:

```text
train.csv or train.parquet
validation.csv or validation.parquet
test.csv or test.parquet
manifest.json
```

The manifest records:

- source and data version;
- exact time boundaries;
- history-filter configuration;
- row counts and series counts per split;
- month range per split;
- SHA-256 per data file;
- excluded series and reasons;
- content-derived `snapshot_id`.

The `snapshot_id` excludes wall-clock creation time, so rebuilding identical
inputs and configuration produces the same identity.

## 9. Mapping the existing three real-data files

The intended integration path is:

1. `sales_aligned_completed.csv` → monthly sales fact import, after verifying its
   concrete brand/series/month/sales headers;
2. `series_mapping_completed.csv` → reviewed source-alias adapter;
3. `vehicles_completed.csv` → canonical catalog and static vehicle attributes.

Stage 1.0B provides the generic monthly-sales path. Stage 1.0C should add the two
explicit adapters after the exact file schemas are available; fuzzy or positional
joins must not be inserted as a shortcut.

## 10. Acceptance criteria

- bilingual sample imports without rejected rows;
- second identical import writes zero new facts;
- bad rows do not roll back valid rows;
- all imports produce persistent quality results;
- chronological split boundaries do not overlap;
- insufficient or discontinuous series are auditable exclusions;
- repeated snapshot builds preserve file hashes and snapshot identity;
- Parquet requests fail explicitly when the optional dependency is absent.
