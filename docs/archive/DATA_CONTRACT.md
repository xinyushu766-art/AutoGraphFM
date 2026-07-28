# AutoGraphFM Data Contract

## Canonical identifiers

All persistent entities use UUID primary keys. External website identifiers
must never replace canonical IDs.

## Brands

| Field | Required | Rule |
|---|---:|---|
| brand_name | yes | original display name |
| normalized_name | yes | deterministic normalized key |
| country | no | free-text country or region |

## Vehicle series

| Field | Required | Rule |
|---|---:|---|
| brand_id | yes | foreign key to brands |
| series_name | yes | canonical display name |
| normalized_name | yes | unique within one brand |
| energy_type | yes | defaults to unknown |
| status | yes | active/discontinued/upcoming/unknown |
| launch_date | no | must not follow discontinue_date |

## Monthly sales

| Field | Required | Rule |
|---|---:|---|
| series_id | yes | canonical forecast node |
| month | yes | first calendar day only |
| sales | yes | non-negative integer |
| source | yes | source system |
| collected_at | yes | acquisition timestamp |
| data_version | yes | source-data revision |

## Monthly dynamic features

Prices and counts are non-negative. Rating is in `[0, 5]`, sentiment in
`[-1, 1]`, and positive ratio in `[0, 1]`.

## Provenance

Every production ingestion should reference an `ingestion_run`. Manual seed data
may omit `run_id`, but real imports should not.
