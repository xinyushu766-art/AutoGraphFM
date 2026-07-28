"""Forecasting metrics evaluated on the original sales scale."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ForecastMetrics:
    """Unified Stage 0.2 evaluation result."""

    log_mse: float
    mae: float
    rmse: float
    wape_percent: float
    smape_percent: float

    def to_dict(self) -> dict[str, float]:
        """Return a CSV-friendly dictionary."""
        return asdict(self)


def inverse_log_sales(log_sales: Tensor) -> Tensor:
    """Invert `log1p` and enforce non-negative sales predictions."""
    return torch.expm1(log_sales.clamp(max=20.0)).clamp_min(0.0)


def compute_forecast_metrics(
    prediction_log_sales: Tensor,
    target_log_sales: Tensor,
    epsilon: float = 1e-8,
) -> ForecastMetrics:
    """Compute log-space loss and original-scale business metrics."""
    if prediction_log_sales.shape != target_log_sales.shape:
        raise ValueError("prediction and target must have identical shapes.")

    prediction_sales = inverse_log_sales(prediction_log_sales)
    target_sales = inverse_log_sales(target_log_sales)

    error = prediction_sales - target_sales
    absolute_error = error.abs()

    log_mse = torch.mean((prediction_log_sales - target_log_sales) ** 2)
    mae = absolute_error.mean()
    rmse = torch.sqrt(torch.mean(error**2))

    wape = absolute_error.sum() / target_sales.abs().sum().clamp_min(epsilon)
    smape = torch.mean(
        2.0
        * absolute_error
        / (prediction_sales.abs() + target_sales.abs() + epsilon)
    )

    return ForecastMetrics(
        log_mse=float(log_mse.item()),
        mae=float(mae.item()),
        rmse=float(rmse.item()),
        wape_percent=float((100.0 * wape).item()),
        smape_percent=float((100.0 * smape).item()),
    )
