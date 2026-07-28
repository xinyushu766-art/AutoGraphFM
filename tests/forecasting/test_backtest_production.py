from autogfm.realdata.common import read_csv_rows


def test_backtest_selects_models_only_from_validation(stage3_artifacts: dict[str, object]) -> None:
    backtest = stage3_artifacts["backtest"]
    _, rows, _ = read_csv_rows(backtest.output_dir / "leaderboard.csv")
    for horizon, selected in backtest.manifest["selected_models"].items():
        candidates = [
            row for row in rows
            if row["split"] == "validation" and row["horizon"] == horizon
        ]
        winner = min(candidates, key=lambda row: float(row["wape_percent"]))
        assert selected == winner["model_name"]


def test_selected_model_metrics_are_held_out(stage3_artifacts: dict[str, object]) -> None:
    backtest = stage3_artifacts["backtest"]
    _, rows, _ = read_csv_rows(backtest.output_dir / "selected_model_test_metrics.csv")
    assert len(rows) == 2
    assert {int(row["horizon"]) for row in rows} == {1, 3}
    assert all(float(row["test_wape_percent"]) >= 0 for row in rows)


def test_production_forecast_has_calibrated_intervals(stage3_artifacts: dict[str, object]) -> None:
    production = stage3_artifacts["production"]
    _, rows, _ = read_csv_rows(production.output_dir / "production_forecast.csv")
    assert len(rows) == 24
    for row in rows:
        prediction = float(row["prediction"])
        assert float(row["lower_80"]) <= prediction <= float(row["upper_80"])
        assert float(row["lower_95"]) <= prediction <= float(row["upper_95"])
        assert float(row["lower_95"]) <= float(row["lower_80"])
        assert float(row["upper_95"]) >= float(row["upper_80"])
