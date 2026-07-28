# Stage 7.0 — SRNI-CAR Real-Data Onboarding


> **Stage 8 update:** formal SRNI runs now require the authenticity preflight in
> `docs/STAGE8_DATA_AUTHENTICITY_GUIDE.md`. A schema-compatible file alone is
> insufficient for a real-data claim.

## Purpose

Stage 7.0 removes the last manual transformation between the public SRNI-CAR
Sales CSV and AutoGraphFM. SRNI-CAR stores sales, vehicle attributes, year, and
month in one file. AutoGraphFM's governed pipeline consumes three logical files.
The Stage 7 adapter deterministically produces those files and retains an audit
manifest.

## Source boundary

- Project: SRNI-CAR, Chinese automotive market data, 2016–2022.
- Required file: the official Sales CSV only.
- License declared by the project: CC BY-NC 4.0.
- The repository does not redistribute the source dataset.
- Commercial use requires a separate data-rights review.

## Download

Try the guarded downloader:

```cmd
python scripts\download_srni_car_sales.py ^
  --destination data\srni_car\Sales.csv ^
  --validation-dir outputs\srni_download_check ^
  --overwrite
```

Figshare private-link downloads can be blocked by managed networks. If the
command fails, use a browser to open the official SRNI-CAR project page, select
the 3.6 MB Sales CSV, and save it as `data\srni_car\Sales.csv`.

## Convert only

```cmd
python scripts\prepare_srni_car.py ^
  --sales-csv data\srni_car\Sales.csv ^
  --output-dir data\srni_car\three_file ^
  --overwrite
```

Outputs:

```text
sales_aligned_completed.csv
series_mapping_completed.csv
vehicles_completed.csv
rejected_srni_rows.json
srni_attribute_conflicts.csv
srni_manifest.json
```

The adapter supports official fields including `Car series`, `Brand`, `Year`,
`Month`, `Sales`, `Car model type`, `Brand energy type`, `Size`, country, and
launch/brand dates. Month values may be numeric, abbreviated English, full
English, or Chinese numeric month labels.

## One-command real run

```cmd
python scripts\run_srni_car_pipeline.py ^
  --sales-csv data\srni_car\Sales.csv ^
  --output-dir outputs\srni_real ^
  --train-end 2020-12 ^
  --validation-end 2021-12 ^
  --test-end 2022-12 ^
  --min-history-months 36 ^
  --horizons 1 3 6 ^
  --seeds 42 43 44 ^
  --overwrite
```

Recommended split:

- training: through December 2020;
- validation/model selection: January–December 2021;
- final held-out test: January–December 2022.

Use `--quick --skip-neural` only for an engineering smoke run. The full run is
required before interpreting model rankings.

## Claim policy

A successful command does not imply that the graph model is effective. A graph
claim remains blocked unless the true graph beats pooled temporal, identity,
and permuted-graph controls on held-out observations. Dataset download,
conversion, and pipeline completion are separate from research effectiveness.
