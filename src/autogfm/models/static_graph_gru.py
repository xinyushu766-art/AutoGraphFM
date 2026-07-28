"""Static graph-temporal GRU baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from autogfm.graph.ops import normalize_adjacency


class StaticGraphGRU(nn.Module):
    """Forecast every vehicle from temporal state and graph context."""

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
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fusion_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.forecast_head = nn.Linear(hidden_dim, horizon)

    def forward(
        self,
        x: Tensor,
        adjacency: Tensor | None,
    ) -> Tensor:
        """Return future log-sales `[batch, horizon, vehicles]`."""
        if x.ndim != 4:
            raise ValueError(
                "x must have shape [batch, history, vehicles, features]."
            )
        if adjacency is None:
            raise ValueError("StaticGraphGRU requires an adjacency matrix.")

        batch_size, history, num_vehicles, num_features = x.shape

        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {num_features}."
            )
        if adjacency.shape != (num_vehicles, num_vehicles):
            raise ValueError(
                "adjacency shape must match the vehicle dimension in x."
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

        normalized_adjacency = normalize_adjacency(adjacency)
        graph_state = torch.einsum(
            "ij,bjd->bid",
            normalized_adjacency,
            temporal_state,
        )
        graph_state = self.graph_projection(graph_state)

        concatenated = torch.cat(
            [temporal_state, graph_state],
            dim=-1,
        )
        gate = torch.sigmoid(self.fusion_gate(concatenated))

        fused_state = gate * temporal_state + (1.0 - gate) * graph_state
        fused_state = self.dropout(fused_state)

        forecast = self.forecast_head(fused_state)
        return forecast.permute(0, 2, 1)
