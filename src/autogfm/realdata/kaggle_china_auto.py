"""Onboarding for the China automobile monthly-sales Kaggle dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autogfm.realdata.common import (
    RealDataError,
    parse_month,
    parse_nonnegative_int,
    read_csv_rows,
    sha256_file,
    stable_payload_id,
    write_csv_rows,
    write_json,
)

KAGGLE_DATASET_SLUG = "felixzhao/china-automobile-monthly-sales-data-2018-2024-4"
KAGGLE_DATASET_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET_SLUG}"
GITHUB_MIRROR_REPOSITORY = "hoangphuc-ngo/Forecasting-China-EV-Sales"
GITHUB_MIRROR_COMMIT = "69a4a61284b301b66c7a31c1de813b476220b55c"
GITHUB_RAW_PATH = "Data/Raw/China Automobile Sales Data.csv"
EXPECTED_BLOB_SHA = "07c5345af9edf3410ff9940f18db8862f6a2bf20"

_REQUIRED = {
    "model",
    "units_sold",
    "make",
    "year_month",
    "is_ev",
    "body_type",
    "brand",
    "brand_country",
}


@dataclass(frozen=True)
class ChinaAutoPreparationResult:
    output_dir: Path
    dataset_id: str
    sales_csv: Path
    mapping_csv: Path
    vehicles_csv: Path
    manifest: dict[str, Any]


def prepare_china_auto_sales(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> ChinaAutoPreparationResult:
    """Convert the public single-file panel into AutoGraphFM's three-file contract."""
    source = Path(source_csv)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    headers, rows, encoding = read_csv_rows(source)
    missing = sorted(_REQUIRED - set(headers))
    if missing:
        raise RealDataError(f"Missing required China-auto columns: {missing}")

    sales_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    attributes: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    keys: Counter[tuple[str, str]] = Counter()

    for line_number, row in enumerate(rows, start=2):
        try:
            model = str(row.get("model", "")).strip()
            make = str(row.get("make", "")).strip()
            if not model or not make:
                raise RealDataError("model and make are required")
            month = parse_month(str(row.get("year_month", "")))
            sales = parse_nonnegative_int(str(row.get("units_sold", "")))
            key = f"{make}::{model}"
            keys[(key, month.isoformat())] += 1
            sales_rows.append(
                {
                    "series_name": key,
                    "brand": str(row.get("brand", "")).strip() or make,
                    "month": month.isoformat(),
                    "sales": sales,
                }
            )
            for field in (
                "brand",
                "make",
                "is_ev",
                "body_type",
                "brand_country",
                "low_price",
                "high_price",
            ):
                value = str(row.get(field, "")).strip()
                if value:
                    attributes[key][field].append(value)
        except (RealDataError, ValueError) as exc:
            rejected_rows.append(
                {"line_number": line_number, "reason": str(exc), "raw_row": repr(row)}
            )

    if not sales_rows:
        raise RealDataError("No valid rows were found in the China-auto dataset.")

    duplicate_keys = {key: count for key, count in keys.items() if count > 1}
    mapping_rows = [
        {
            "source_series": key,
            "canonical_series": key,
            "brand": Counter(values.get("brand", []) or values.get("make", [])).most_common(1)[0][0],
        }
        for key, values in sorted(attributes.items())
    ]

    vehicle_rows: list[dict[str, Any]] = []
    for key, values in sorted(attributes.items()):
        def mode(field: str) -> str:
            candidates = values.get(field, [])
            return Counter(candidates).most_common(1)[0][0] if candidates else ""

        vehicle_rows.append(
            {
                "series_name": key,
                "brand": mode("brand") or mode("make"),
                "body_type": mode("body_type"),
                "energy_type": mode("is_ev"),
                "country": mode("brand_country"),
                "min_price": mode("low_price"),
                "max_price": mode("high_price"),
                "status": "unknown",
            }
        )

    sales_path = destination / "sales_aligned_completed.csv"
    mapping_path = destination / "series_mapping_completed.csv"
    vehicles_path = destination / "vehicles_completed.csv"
    write_csv_rows(sales_path, ["series_name", "brand", "month", "sales"], sales_rows)
    write_csv_rows(
        mapping_path,
        ["source_series", "canonical_series", "brand"],
        mapping_rows,
    )
    write_csv_rows(
        vehicles_path,
        [
            "series_name",
            "brand",
            "body_type",
            "energy_type",
            "country",
            "min_price",
            "max_price",
            "status",
        ],
        vehicle_rows,
    )
    write_csv_rows(
        destination / "rejected_rows.csv",
        ["line_number", "reason", "raw_row"],
        rejected_rows,
    )

    months = sorted({row["month"] for row in sales_rows})
    dataset_id = "china_auto_" + stable_payload_id(
        {
            "sha256": sha256_file(source),
            "valid_rows": len(sales_rows),
            "series": len(attributes),
            "month_min": months[0],
            "month_max": months[-1],
        },
    )
    manifest = {
        "dataset_id": dataset_id,
        "source": {
            "kaggle_slug": KAGGLE_DATASET_SLUG,
            "kaggle_url": KAGGLE_DATASET_URL,
            "mirror_repository": GITHUB_MIRROR_REPOSITORY,
            "mirror_commit": GITHUB_MIRROR_COMMIT,
            "mirror_path": GITHUB_RAW_PATH,
            "mirror_blob_sha": EXPECTED_BLOB_SHA,
            "local_sha256": sha256_file(source),
            "encoding": encoding,
        },
        "profile": {
            "input_rows": len(rows),
            "valid_rows": len(sales_rows),
            "rejected_rows": len(rejected_rows),
            "series_count": len(attributes),
            "brand_count": len({row["brand"] for row in sales_rows}),
            "month_min": months[0],
            "month_max": months[-1],
            "duplicate_key_count": len(duplicate_keys),
        },
        "evidence_scope": "public_real_model_panel",
        "limitations": [
            "The dataset is a public mirror of the Kaggle source, not a publisher-signed artifact.",
            "Duplicate model-month keys must be explicitly aggregated downstream.",
        ],
    }
    write_json(destination / "PREPARATION_MANIFEST.json", manifest)
    return ChinaAutoPreparationResult(
        output_dir=destination,
        dataset_id=dataset_id,
        sales_csv=sales_path,
        mapping_csv=mapping_path,
        vehicles_csv=vehicles_path,
        manifest=manifest,
    )
