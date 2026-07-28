# Stage 8.0 — SRNI-CAR Authenticity and Claim Governance

## Purpose

Stage 8 prevents a schema-compatible fixture, an incomplete export, or an
unknown mirror from being reported as the official SRNI-CAR benchmark. The
official-schema adapter remains available, but the formal runner now inspects
the source before any model training.

## Published reference profile

The project page and resource paper describe the Sales CSV as:

- Chinese automotive sales from 2016 through 2022;
- 39,496 monthly sales observations;
- 1,236 vehicle series;
- 155 brands;
- a 3.6 MB CSV with `Car series`, `Brand`, `Year`, `Month`, and `Sales`;
- CC BY-NC 4.0 licensed.

References:

- `https://srni-car.github.io/`
- `https://arxiv.org/abs/2401.05395`

No publisher checksum is exposed on the project page. AutoGraphFM therefore
retains the local SHA-256 and uses a transparent profile match rather than
claiming cryptographic identity with the publisher-hosted bytes.

## Evidence scopes

| Scope | Meaning | Formal result allowed? | Production release? |
|---|---|---:|---:|
| `official_real_data` | Schema and published profile match; local hash retained | Yes, with profile-match caveat | Eligible for other gates |
| `real_unverified` | Plausible real rows but profile does not match | No official SRNI claim | No |
| `synthetic_only` | Generated or known fixture pattern detected | Engineering only | No |
| `demo_only` | Built-in AutoGraphFM demo | Engineering only | No |

A profile match is not a substitute for source provenance. Download the file
from the official project page and retain the generated authenticity manifest.

## Download and validate

Try both official Figshare download endpoints:

```cmd
python scripts\download_srni_car_sales.py ^
  --destination data\srni_car\Sales.csv ^
  --validation-dir outputs\srni_download_check ^
  --overwrite
```

If the network blocks Figshare, download the Sales CSV manually from the
project page and run:

```cmd
python scripts\validate_srni_car_file.py ^
  --sales-csv data\srni_car\Sales.csv ^
  --output-dir outputs\srni_file_validation ^
  --overwrite
```

The command returns a non-zero exit status unless the file matches the official
profile. It writes:

```text
srni_authenticity_manifest.json
SRNI_AUTHENTICITY_REPORT.md
```

## Formal real-data run

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

The formal runner stops before model training when the evidence scope is not
`official_real_data`.

## Engineering-only run

For a small fixture or an incomplete sample, explicitly opt into a non-claiming
run:

```cmd
python scripts\run_srni_car_pipeline.py ^
  --sales-csv path\to\fixture.csv ^
  --output-dir outputs\srni_engineering_check ^
  --quick ^
  --skip-neural ^
  --allow-nonofficial ^
  --overwrite
```

The resulting HTML and Stage 4 manifest retain `synthetic_only` or
`real_unverified`, and production release remains blocked.

## Windows one-command helper

```powershell
powershell -ExecutionPolicy Bypass -File scripts/acquire_and_run_srni.ps1
```

The helper attempts both official endpoints, validates the result, and only
then starts the formal pipeline.

## Acceptance

```cmd
python scripts\validate_stage8.py --overwrite
```

This acceptance deliberately generates a synthetic official-schema fixture. It
passes only when strict mode rejects that fixture, engineering mode labels it
`synthetic_only`, the audit contains no `USER_DATA` claim, and production
release is blocked.
