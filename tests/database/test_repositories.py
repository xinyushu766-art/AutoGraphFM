from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from autogfm.database.repositories import CatalogRepository, MonthlyRepository
from autogfm.schemas import (
    BrandCreate,
    MonthlySalesCreate,
    VehicleAliasCreate,
    VehicleSeriesCreate,
)


def test_catalog_is_idempotent_and_alias_resolves(session: Session) -> None:
    catalog = CatalogRepository(session)

    first_brand = catalog.get_or_create_brand(
        BrandCreate(brand_name="BYD")
    )
    second_brand = catalog.get_or_create_brand(
        BrandCreate(brand_name="ＢＹＤ")
    )
    assert first_brand.id == second_brand.id

    series = catalog.get_or_create_series(
        VehicleSeriesCreate(
            brand_id=first_brand.id,
            series_name="Qin PLUS",
            energy_type="phev",
        )
    )
    catalog.add_alias(
        VehicleAliasCreate(
            source="demo",
            source_name="秦 PLUS 新能源",
            series_id=series.id,
        )
    )

    resolved = catalog.resolve_alias("demo", "秦PLUS新能源")
    assert resolved is not None
    assert resolved.id == series.id


def test_monthly_repository_returns_chronological_rows(
    session: Session,
) -> None:
    catalog = CatalogRepository(session)
    monthly = MonthlyRepository(session)

    brand = catalog.get_or_create_brand(BrandCreate(brand_name="Test"))
    series = catalog.get_or_create_series(
        VehicleSeriesCreate(
            brand_id=brand.id,
            series_name="Series",
        )
    )

    for month in [date(2025, 2, 1), date(2025, 1, 1)]:
        monthly.add_sales(
            MonthlySalesCreate(
                series_id=series.id,
                month=month,
                sales=100,
                source="test",
                collected_at=datetime.now(UTC),
            )
        )

    rows = monthly.list_sales(series.id)
    assert [row.month for row in rows] == [
        date(2025, 1, 1),
        date(2025, 2, 1),
    ]
