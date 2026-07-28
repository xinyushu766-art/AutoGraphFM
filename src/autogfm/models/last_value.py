"""Last-observation forecasting baseline."""

from __future__ import annotations

from torch import Tensor, nn


class LastValueForecaster(nn.Module):
    """Repeat the most recent observed log-sales over the forecast horizon."""

    def __init__(self, horizon: int, sales_feature_index: int = 0) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        self.horizon = horizon
        self.sales_feature_index = sales_feature_index

    def forward(
        self,
        x: Tensor,
        adjacency: Tensor | None = None,
    ) -> Tensor:
        """Return predictions with shape `[batch, horizon, vehicles]`."""
        del adjacency

        if x.ndim != 4:
            raise ValueError(
                "x must have shape [batch, history, vehicles, features]."
            )

        last_log_sales = x[:, -1, :, self.sales_feature_index]
        return last_log_sales.unsqueeze(1).repeat(1, self.horizon, 1)
