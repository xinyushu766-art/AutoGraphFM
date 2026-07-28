from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_initial_migration_upgrades_and_downgrades(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "migration.sqlite"
    database_url = f"sqlite+pysqlite:///{database_path}"

    monkeypatch.setenv("AUTOGFM_DATABASE_URL", database_url)

    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option(
        "script_location",
        str(project_root / "migrations"),
    )

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    tables = set(inspect(engine).get_table_names())
    assert {
        "brands",
        "vehicle_series",
        "vehicle_aliases",
        "ingestion_runs",
        "monthly_sales",
        "monthly_series_features",
        "data_quality_results",
    }.issubset(tables)

    command.downgrade(config, "base")
    remaining_tables = set(inspect(engine).get_table_names())
    assert remaining_tables <= {"alembic_version"}
