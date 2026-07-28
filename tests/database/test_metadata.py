from autogfm.database.base import Base


def test_all_expected_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == {
        "brands",
        "vehicle_series",
        "vehicle_aliases",
        "ingestion_runs",
        "monthly_sales",
        "monthly_series_features",
        "data_quality_results",
    }
