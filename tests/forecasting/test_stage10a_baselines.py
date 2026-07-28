from __future__ import annotations

from pathlib import Path

import numpy as np

from autogfm.forecasting.baselines import (
    damped_trend_forecast,
    regime_guarded_forecast,
    seasonal_growth_forecast,
)
from autogfm.forecasting.market_benchmark import (
    run_multiwindow_market_benchmark,
)


def test_strong_baselines_are_nonnegative_and_shape_safe() -> None:
    sales = np.arange(1, 73, dtype=np.float32)[:, None] * np.array(
        [[10.0, 20.0]], dtype=np.float32
    )
    for forecast in (
        damped_trend_forecast(sales, 60, (1, 3, 6)),
        seasonal_growth_forecast(sales, 60, (1, 3, 6)),
        regime_guarded_forecast(sales, 60, (1, 3, 6)),
    ):
        assert forecast.shape == (3, 2)
        assert np.isfinite(forecast).all()
        assert (forecast >= 0).all()


def test_multiwindow_market_benchmark(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    rows = ["year_month,EV_Sales,Total_Sales"]
    year, month = 2018, 1
    for index in range(76):
        seasonal = 30 if month in {1, 2} else 100
        rows.append(
            f"{year:04d}-{month:02d}-01,"
            f"{1000 + index * 20 + seasonal},"
            f"{10000 + index * 30 + seasonal}"
        )
        month += 1
        if month == 13:
            year += 1
            month = 1
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = run_multiwindow_market_benchmark(
        source,
        tmp_path / "out",
        horizons=(1, 3, 6),
    )
    assert result.manifest["source_rows"] == 76
    assert len(result.manifest["folds"]) == 3
    observed_horizons = {
        row["horizon"]
        for row in result.manifest["aggregate_selected_metrics"]
    }
    assert observed_horizons == {
        1,
        3,
        6,
    }
    assert (result.output_dir / "MULTIWINDOW_MARKET_REPORT.html").is_file()
