from datetime import date
from uuid import uuid4

from autogfm.data_quality import validate_raw_monthly_sales


def test_raw_quality_checks_find_invalid_and_duplicate_rows() -> None:
    series_id = uuid4()
    records = [
        {
            "series_id": series_id,
            "month": date(2025, 1, 1),
            "sales": 100,
            "source": "test",
            "data_version": "v1",
        },
        {
            "series_id": series_id,
            "month": date(2025, 1, 1),
            "sales": -2,
            "source": "test",
            "data_version": "v1",
        },
    ]

    issues = validate_raw_monthly_sales(records)
    checks = {issue.check_name for issue in issues}

    assert "sales_nonnegative_integer" in checks
    assert "duplicate_business_key" in checks
