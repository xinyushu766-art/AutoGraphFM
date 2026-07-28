"""Print database table names and row counts."""

from sqlalchemy import func, inspect, select

from autogfm.database import (
    create_database_engine,
    create_session_factory,
    session_scope,
)
from autogfm.database.models import (
    Brand,
    DataQualityResult,
    IngestionRun,
    MonthlySales,
    MonthlySeriesFeature,
    VehicleAlias,
    VehicleSeries,
)

MODELS = [
    Brand,
    VehicleSeries,
    VehicleAlias,
    IngestionRun,
    MonthlySales,
    MonthlySeriesFeature,
    DataQualityResult,
]


def main() -> None:
    engine = create_database_engine()
    tables = inspect(engine).get_table_names()

    print("Tables:")
    for table_name in sorted(tables):
        print(f"  - {table_name}")

    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        print("\nRow counts:")
        for model in MODELS:
            count = session.scalar(
                select(func.count()).select_from(model)
            )
            print(f"  {model.__tablename__:<28} {count}")


if __name__ == "__main__":
    main()
