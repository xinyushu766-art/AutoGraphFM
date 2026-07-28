import numpy as np

from autogfm.forecasting.baselines import (
    PooledResidualRidge,
    last_value_forecast,
    seasonal_naive_forecast,
)
from autogfm.realdata import load_panel


def test_last_value_repeats_origin() -> None:
    sales = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    prediction = last_value_forecast(sales, 2, (1, 3))
    np.testing.assert_allclose(prediction, [[5, 6], [5, 6]])


def test_seasonal_naive_uses_twelve_month_reference() -> None:
    sales = np.arange(30, dtype=np.float32)[:, None]
    prediction = seasonal_naive_forecast(sales, 20, (1, 3))
    assert prediction[:, 0].tolist() == [9.0, 11.0]


def test_pooled_ridge_predicts_nonnegative_shape(stage3_artifacts: dict[str, object]) -> None:
    panel = load_panel(stage3_artifacts["panel"].output_dir)
    month_numbers = np.array([int(value[5:7]) for value in panel.months])
    model = PooledResidualRidge(alpha=5.0, month_numbers=month_numbers)
    model.fit(panel.sales, panel.static_features, 45, history=12, horizons=(1, 3))
    prediction = model.predict(panel.sales, panel.static_features, 45, (1, 3))
    assert prediction.shape == (2, panel.sales.shape[1])
    assert np.all(prediction >= 0)
