# Data sources

## Bundled aggregate benchmark

`data/china_auto_market/ev_sales_monthly.csv` contains 76 monthly aggregate observations for EV and total automotive sales from January 2018 through April 2024. It was derived from the public vehicle-month dataset referenced by the project documentation.

The file is included only to reproduce the market-level benchmark. Source attribution and downstream use must follow the original dataset terms.

## External vehicle-level datasets

AutoGraphFM includes adapters for:

- SRNI-CAR Sales CSV;
- China Automobile Monthly Sales Data (2018–2024.4);
- an operational three-file contract: sales, series mapping, and vehicle attributes.

Large external raw files are not redistributed in this repository. Download them from their original publishers and retain a source URL and SHA-256 hash in the run manifest.

## Data contract

The canonical monthly sales grain is:

```text
vehicle_series_id × calendar_month
```

Required target fields:

- canonical vehicle-series identifier;
- month;
- non-negative sales count.

Optional static fields include brand, manufacturer, body type, energy type, price range, origin country, and launch date.
