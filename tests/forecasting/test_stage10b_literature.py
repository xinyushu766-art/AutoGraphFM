from __future__ import annotations

from pathlib import Path

import numpy as np

from autogfm.forecasting.literature_models import (
    dlinear_ridge_forecast,
    nlinear_ridge_forecast,
)
from autogfm.forecasting.market_benchmark import (
    STAGE10B_MODEL_NAMES,
    run_multiwindow_market_benchmark,
)
from autogfm.forecasting.regime import regime_snapshot


def _months(count: int) -> list[str]:
    rows = []
    year, month = 2017, 1
    for _ in range(count):
        rows.append(f"{year:04d}-{month:02d}-01")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return rows


def test_literature_forecasts_are_nonnegative_and_future_safe() -> None:
    months = _months(76)
    values = np.asarray(
        [1000.0 + 20.0 * index + 100.0 * np.sin(index / 3.0) for index in range(76)]
    )
    origin = 60
    baseline = (
        nlinear_ridge_forecast(values, months, origin, 3),
        dlinear_ridge_forecast(values, months, origin, 3),
    )
    mutated = values.copy()
    mutated[origin + 1 :] *= 100.0
    repeated = (
        nlinear_ridge_forecast(mutated, months, origin, 3),
        dlinear_ridge_forecast(mutated, months, origin, 3),
    )
    assert np.allclose(baseline, repeated)
    assert all(np.isfinite(value) and value >= 0.0 for value in baseline)


def test_regime_snapshot_detects_shock() -> None:
    stable = np.full(36, 1000.0)
    stable[-4:] = np.asarray([500.0, 1800.0, 300.0, 2200.0])
    snapshot = regime_snapshot(stable, len(stable) - 1)
    assert snapshot.label == "shock"
    assert snapshot.change_score >= 3.0 or snapshot.volatility_ratio >= 2.2


def test_expanded_multiwindow_benchmark_registers_literature_models(
    tmp_path: Path,
) -> None:
    source = tmp_path / "market.csv"
    rows = ["year_month,EV_Sales,Total_Sales"]
    for index, month in enumerate(_months(76)):
        seasonal = 250 if month[5:7] in {"01", "02"} else 800
        rows.append(
            f"{month},{1000 + index * 30 + seasonal},"
            f"{10000 + index * 50 + seasonal}"
        )
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = run_multiwindow_market_benchmark(
        source,
        tmp_path / "out",
        models=STAGE10B_MODEL_NAMES,
    )
    assert {
        "nlinear_ridge",
        "dlinear_ridge",
        "regime_error_moe",
    }.issubset(set(result.manifest["models"]))
    prediction_text = (result.output_dir / "predictions.csv").read_text(
        encoding="utf-8"
    )
    assert "expert_weights" in prediction_text
    assert "regime_error_moe" in prediction_text
