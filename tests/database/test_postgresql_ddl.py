from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from autogfm.database.models import MonthlySales


def test_monthly_sales_compiles_for_postgresql() -> None:
    ddl = str(
        CreateTable(MonthlySales.__table__).compile(
            dialect=postgresql.dialect()
        )
    )

    assert "CREATE TABLE monthly_sales" in ddl
    assert "EXTRACT(day FROM month)" in ddl
