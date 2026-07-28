"""Residual temporal-only GRU baseline."""

from __future__ import annotations

from torch import Tensor, nn


class ResidualTemporalGRU(nn.Module):
    """Learn a correction on top of the Last Value forecast.

    The model predicts:

    `future_log_sales = last_log_sales + delta`.

    The delta head starts at zero, so the untrained model exactly reproduces
    Last Value. Training therefore optimizes improvements rather than learning
    the absolute sales level from scratch.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        horizon: int,
        sales_feature_index: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        if not 0 <= sales_feature_index < num_features:
            raise ValueError("sales_feature_index is out of range.")

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.sales_feature_index = sales_feature_index

        self.temporal_encoder = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.delta_head = nn.Linear(hidden_dim, horizon)

        # Exact Last Value initialization.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

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

        delta = self.delta_head(temporal_state).permute(0, 2, 1)
        last_log_sales = x[
            :,
            -1,
            :,
            self.sales_feature_index,
        ].unsqueeze(1)

        return last_log_sales + delta
