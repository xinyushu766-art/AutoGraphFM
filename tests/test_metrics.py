import torch

from autogfm.metrics import compute_forecast_metrics


def test_perfect_prediction_has_zero_error() -> None:
    target = torch.log1p(
        torch.tensor([[[100.0, 200.0], [120.0, 210.0]]])
    )

    metrics = compute_forecast_metrics(target, target)

    assert metrics.log_mse == 0.0
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.wape_percent == 0.0
    assert metrics.smape_percent == 0.0


def test_metrics_are_computed_on_original_scale() -> None:
    target = torch.log1p(torch.tensor([[[100.0]]]))
    prediction = torch.log1p(torch.tensor([[[110.0]]]))

    metrics = compute_forecast_metrics(prediction, target)

    assert abs(metrics.mae - 10.0) < 1e-4
    assert abs(metrics.wape_percent - 10.0) < 1e-4
