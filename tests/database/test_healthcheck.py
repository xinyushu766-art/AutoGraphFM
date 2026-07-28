from sqlalchemy import create_engine

from autogfm.database.session import database_healthcheck


def test_sqlite_healthcheck_passes() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    assert database_healthcheck(engine)
