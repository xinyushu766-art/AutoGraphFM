"""Pre-ingestion checks for raw monthly sales dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class QualityIssue:
    """One deterministic quality failure."""

    row_index: int
    check_name: str
    message: str


def validate_raw_monthly_sales(
    records: list[dict[str, Any]],
) -> list[QualityIssue]:
    """Check required fields, month format, non-negative sales, and duplicates."""
    issues: list[QualityIssue] = []
    seen_keys: set[tuple[Any, ...]] = set()

    for index, record in enumerate(records):
        required = ("series_id", "month", "sales", "source", "data_version")
        missing = [name for name in required if name not in record]
        if missing:
            issues.append(
                QualityIssue(
                    row_index=index,
                    check_name="required_fields",
                    message=f"Missing fields: {', '.join(missing)}",
                )
            )
            continue

        month = record["month"]
        if not isinstance(month, date) or month.day != 1:
            issues.append(
                QualityIssue(
                    row_index=index,
                    check_name="month_first_day",
                    message="month must be a date on the first calendar day.",
                )
            )

        sales = record["sales"]
        if not isinstance(sales, int) or isinstance(sales, bool) or sales < 0:
            issues.append(
                QualityIssue(
                    row_index=index,
                    check_name="sales_nonnegative_integer",
                    message="sales must be a non-negative integer.",
                )
            )

        key = (
            record["series_id"],
            record["month"],
            record["source"],
            record["data_version"],
        )
        if key in seen_keys:
            issues.append(
                QualityIssue(
                    row_index=index,
                    check_name="duplicate_business_key",
                    message=(
                        "Duplicate series/month/source/data_version record."
                    ),
                )
            )
        seen_keys.add(key)

    return issues
