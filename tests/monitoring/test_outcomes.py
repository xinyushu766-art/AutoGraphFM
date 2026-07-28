from autogfm.monitoring import OutcomeSpec, evaluate_forecast_outcomes
from autogfm.realdata.common import write_csv_rows


FORECAST_FIELDS = [
    "series_index",
    "series_key",
    "brand_name",
    "series_name",
    "origin_month",
    "target_month",
    "horizon",
    "selected_model",
    "prediction",
    "calibration_cells",
    "lower_80",
    "upper_80",
    "lower_95",
    "upper_95",
]
ACTUAL_FIELDS = [
    "series_key",
    "brand_name",
    "series_name",
    "month",
    "sales",
    "sales_rank",
    "source",
    "data_version",
]


def _write_inputs(tmp_path, actual_values):
    forecasts = []
    actuals = []
    for index, actual in enumerate(actual_values):
        prediction = 100.0
        forecasts.append(
            {
                "series_index": index,
                "series_key": f"b::s{index}",
                "brand_name": "b",
                "series_name": f"s{index}",
                "origin_month": "2025-12-01",
                "target_month": "2026-01-01",
                "horizon": 1,
                "selected_model": "last_value",
                "prediction": prediction,
                "calibration_cells": 20,
                "lower_80": 80,
                "upper_80": 120,
                "lower_95": 60,
                "upper_95": 140,
            }
        )
        actuals.append(
            {
                "series_key": f"b::s{index}",
                "brand_name": "b",
                "series_name": f"s{index}",
                "month": "2026-01-01",
                "sales": actual,
                "sales_rank": index + 1,
                "source": "test",
                "data_version": "v1",
            }
        )
    forecast_path = tmp_path / "forecast.csv"
    actual_path = tmp_path / "actual.csv"
    write_csv_rows(forecast_path, FORECAST_FIELDS, forecasts)
    write_csv_rows(actual_path, ACTUAL_FIELDS, actuals)
    return forecast_path, actual_path


def test_outcome_monitor_retains_healthy_champion(tmp_path) -> None:
    forecast_path, actual_path = _write_inputs(tmp_path, [95, 100, 105, 110])
    result = evaluate_forecast_outcomes(
        forecast_path,
        actual_path,
        tmp_path / "healthy",
        overwrite=True,
    )
    assert result.status == "pass"
    assert result.action == "retain_champion"
    assert (result.output_dir / "OUTCOME_REPORT.md").is_file()


def test_outcome_monitor_blocks_large_realized_error(tmp_path) -> None:
    forecast_path, actual_path = _write_inputs(tmp_path, [20, 25, 30, 35])
    result = evaluate_forecast_outcomes(
        forecast_path,
        actual_path,
        tmp_path / "bad",
        OutcomeSpec(wape_warn_percent=15, wape_block_percent=30),
        overwrite=True,
    )
    assert result.status == "block"
    assert result.action == "retrain_revalidate_or_rollback"
