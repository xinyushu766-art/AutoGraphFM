"""Residual GRU with a zero-initialized graph adapter."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from autogfm.graph.ops import normalize_adjacency


class SafeGraphResidualGRU(nn.Module):
    """Add graph context conservatively to a temporal representation.

    The model uses two nested residual ideas:

    1. the final forecast starts from Last Value;
    2. the graph branch starts from zero influence.

    At initialization:

    `graph_scale = 0` and `delta_head = 0`.

    Therefore the complete model initially equals Last Value. During training,
    it can first learn a temporal correction and only use graph information when
    the validation signal supports it.
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
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
        )

        # One bounded scale per hidden dimension. Zero means no graph effect.
        self.graph_scale = nn.Parameter(torch.zeros(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.delta_head = nn.Linear(hidden_dim, horizon)

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def encode_temporal(self, x: Tensor) -> Tensor:
        """Encode every vehicle history with a shared GRU."""
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

        return final_hidden[-1].reshape(
            batch_size,
            num_vehicles,
            self.hidden_dim,
        )

    def forward(
        self,
        x: Tensor,
        adjacency: Tensor | None,
    ) -> Tensor:
        """Return future log-sales `[batch, horizon, vehicles]`."""
        if adjacency is None:
            raise ValueError(
                "SafeGraphResidualGRU requires an adjacency matrix."
            )

        temporal_state = self.encode_temporal(x)
        num_vehicles = temporal_state.shape[1]

        if adjacency.shape != (num_vehicles, num_vehicles):
            raise ValueError(
                "adjacency shape must match the vehicle dimension in x."
            )

        normalized_adjacency = normalize_adjacency(adjacency)
        neighbor_state = torch.einsum(
            "ij,bjd->bid",
            normalized_adjacency,
            temporal_state,
        )

        # The correction is relational rather than a replacement for H_t.
        graph_correction = self.graph_projection(
            neighbor_state - temporal_state
        )
        bounded_scale = torch.tanh(self.graph_scale).view(
            1,
            1,
            self.hidden_dim,
        )
        fused_state = temporal_state + bounded_scale * graph_correction
        fused_state = self.dropout(fused_state)

        delta = self.delta_head(fused_state).permute(0, 2, 1)
        last_log_sales = x[
            :,
            -1,
            :,
            self.sales_feature_index,
        ].unsqueeze(1)

        return last_log_sales + delta
