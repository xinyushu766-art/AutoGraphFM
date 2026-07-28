"""Temporal-only GRU baseline."""

from __future__ import annotations

from torch import Tensor, nn


class TemporalOnlyGRU(nn.Module):
    """Forecast each vehicle from its own history with shared GRU parameters."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        horizon: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.temporal_encoder = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Linear(hidden_dim, horizon)

    def forward(
        self,
        x: Tensor,
        adjacency: Tensor | None = None,
    ) -> Tensor:
        """Return future log-sales `[batch, horizon, vehicles]`."""
        del adjacency

        if x.ndim != 4:
            raise ValueError(
                "x must have shape [batch, history, vehicles, features]."
            )

        batch_size, history, num_vehicles, num_features = x.shape
        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {num_features}."
            )

        node_sequences = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_vehicles,
            history,
            num_features,
        )
        _, final_hidden = self.temporal_encoder(node_sequences)

        temporal_state = final_hidden[-1].reshape(
            batch_size,
            num_vehicles,
            self.hidden_dim,
        )
        temporal_state = self.dropout(temporal_state)

        forecast = self.forecast_head(temporal_state)
        return forecast.permute(0, 2, 1)
