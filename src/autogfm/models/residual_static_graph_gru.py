"""Residual static graph-temporal GRU baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from autogfm.graph.ops import normalize_adjacency


class ResidualStaticGraphGRU(nn.Module):
    """Learn graph-conditioned corrections to the Last Value forecast."""

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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fusion_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.delta_head = nn.Linear(hidden_dim, horizon)

        # The initial prediction is exactly Last Value for every graph variant.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

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
            raise ValueError(
                "ResidualStaticGraphGRU requires an adjacency matrix."
            )

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

        gate = torch.sigmoid(
            self.fusion_gate(
                torch.cat(
                    [temporal_state, graph_state],
                    dim=-1,
                )
            )
        )
        fused_state = (
            gate * temporal_state
            + (1.0 - gate) * graph_state
        )
        fused_state = self.dropout(fused_state)

        delta = self.delta_head(fused_state).permute(0, 2, 1)
        last_log_sales = x[
            :,
            -1,
            :,
            self.sales_feature_index,
        ].unsqueeze(1)

        return last_log_sales + delta
