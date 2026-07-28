from datetime import UTC, date, datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from autogfm.database.models import Brand, MonthlySales, VehicleSeries


def _create_series(session: Session) -> VehicleSeries:
    brand = Brand(
        brand_name="Test Brand",
        normalized_name="testbrand",
    )
    series = VehicleSeries(
        brand=brand,
        series_name="Test Series",
        normalized_name="testseries",
        energy_type="ev",
        status="active",
    )
    session.add(series)
    session.flush()
    return series


def test_negative_sales_are_rejected(session: Session) -> None:
    series = _create_series(session)
    session.add(
        MonthlySales(
            series_id=series.id,
            month=date(2025, 1, 1),
            sales=-1,
            source="test",
            collected_at=datetime.now(UTC),
            data_version="v1",
        )
    )

    with pytest.raises(IntegrityError):
        session.flush()


def test_duplicate_business_key_is_rejected(session: Session) -> None:
    series = _create_series(session)
    common = {
        "series_id": series.id,
        "month": date(2025, 1, 1),
        "sales": 100,
        "source": "test",
        "collected_at": datetime.now(UTC),
        "data_version": "v1",
    }
    session.add_all([MonthlySales(**common), MonthlySales(**common)])

    with pytest.raises(IntegrityError):
        session.flush()


def test_non_first_day_month_is_rejected(session: Session) -> None:
    series = _create_series(session)
    session.add(
        MonthlySales(
            series_id=series.id,
            month=date(2025, 1, 2),
            sales=100,
            source="test",
            collected_at=datetime.now(UTC),
            data_version="v1",
        )
    )

    with pytest.raises(IntegrityError):
        session.flush()
