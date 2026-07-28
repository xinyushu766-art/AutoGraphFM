from datetime import UTC, date, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from autogfm.schemas import MonthlySalesCreate, VehicleSeriesCreate
from autogfm.schemas.common import normalize_entity_name


def test_entity_normalization_handles_spacing_and_width() -> None:
    assert normalize_entity_name("秦 PLUS") == normalize_entity_name(
        "秦ＰＬＵＳ"
    )


def test_monthly_sales_rejects_non_month_start() -> None:
    with pytest.raises(ValidationError):
        MonthlySalesCreate(
            series_id=uuid4(),
            month=date(2025, 1, 15),
            sales=100,
            source="test",
            collected_at=datetime.now(UTC),
        )


def test_vehicle_series_rejects_invalid_lifecycle() -> None:
    with pytest.raises(ValidationError):
        VehicleSeriesCreate(
            brand_id=uuid4(),
            series_name="Example",
            launch_date=date(2025, 5, 1),
            discontinue_date=date(2025, 4, 1),
        )
