"""Insert a small idempotent development catalog and sales sample."""

from datetime import UTC, date, datetime

from sqlalchemy import select

from autogfm.database import (
    create_database_engine,
    create_session_factory,
    session_scope,
)
from autogfm.database.models import MonthlySales
from autogfm.database.repositories import CatalogRepository, MonthlyRepository
from autogfm.schemas import (
    BrandCreate,
    MonthlySalesCreate,
    VehicleAliasCreate,
    VehicleSeriesCreate,
)


def main() -> None:
    engine = create_database_engine()
    factory = create_session_factory(engine)

    with session_scope(factory) as session:
        catalog = CatalogRepository(session)
        monthly = MonthlyRepository(session)

        brand = catalog.get_or_create_brand(
            BrandCreate(brand_name="BYD", country="China")
        )
        series = catalog.get_or_create_series(
            VehicleSeriesCreate(
                brand_id=brand.id,
                series_name="Qin PLUS",
                segment="compact sedan",
                body_type="sedan",
                energy_type="phev",
                launch_date=date(2021, 3, 1),
            )
        )

        if catalog.resolve_alias("demo", "秦PLUS新能源") is None:
            catalog.add_alias(
                VehicleAliasCreate(
                    source="demo",
                    source_name="秦PLUS新能源",
                    series_id=series.id,
                    match_method="manual",
                    match_score=1.0,
                )
            )

        for month, sales in [
            (date(2025, 1, 1), 24000),
            (date(2025, 2, 1), 21000),
            (date(2025, 3, 1), 26500),
        ]:
            existing = session.scalar(
                select(MonthlySales).where(
                    MonthlySales.series_id == series.id,
                    MonthlySales.month == month,
                    MonthlySales.source == "demo",
                    MonthlySales.data_version == "seed-v1",
                )
            )
            if existing is None:
                monthly.add_sales(
                    MonthlySalesCreate(
                        series_id=series.id,
                        month=month,
                        sales=sales,
                        source="demo",
                        collected_at=datetime.now(UTC),
                        data_version="seed-v1",
                    )
                )

    print("Development seed data is ready.")


if __name__ == "__main__":
    main()
