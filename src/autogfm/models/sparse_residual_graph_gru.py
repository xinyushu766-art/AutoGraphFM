"""Residual GRU with a regularized direct sparse residual graph."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from autogfm.graph.ops import normalize_adjacency
from autogfm.graph.sparse_residual import DirectSparseResidualGraphLearner


class SparseResidualGraphGRU(nn.Module):
    """Learn sparse vehicle relations while retaining a static prior.

    Forecasting starts at Last Value.  The graph correction and output head are
    zero-initialized, so the learned graph cannot damage the initial prediction.
    The direct graph has enough rank to represent arbitrary sparse relations,
    while the residual gate and explicit regularization keep it conservative.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        horizon: int,
        num_nodes: int,
        *,
        top_k: int = 5,
        use_prior: bool = True,
        sales_feature_index: int = 0,
        dropout: float = 0.1,
        initial_residual_weight: float = 0.15,
        l1_strength: float = 2e-4,
        deviation_strength: float = 2e-3,
        gate_strength: float = 2e-4,
        candidate_mask: Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_features <= 0 or hidden_dim <= 0 or horizon <= 0:
            raise ValueError("num_features, hidden_dim and horizon must be positive.")
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2.")
        if not 0 <= sales_feature_index < num_features:
            raise ValueError("sales_feature_index is out of range.")

        self.num_features = int(num_features)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.num_nodes = int(num_nodes)
        self.sales_feature_index = int(sales_feature_index)
        self.use_prior = bool(use_prior)

        self.temporal_encoder = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.graph_learner = DirectSparseResidualGraphLearner(
            num_nodes,
            min(top_k, num_nodes - 1),
            initial_residual_weight=initial_residual_weight,
            l1_strength=l1_strength,
            deviation_strength=deviation_strength,
            gate_strength=gate_strength,
            candidate_mask=candidate_mask,
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
        )
        self.graph_scale = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.delta_head = nn.Linear(hidden_dim, horizon)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def encode_temporal(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, history, nodes, features].")
        batch_size, history, num_nodes, num_features = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}.")
        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {num_features}."
            )
        sequences = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes,
            history,
            num_features,
        )
        _, hidden = self.temporal_encoder(sequences)
        return hidden[-1].reshape(batch_size, num_nodes, self.hidden_dim)

    def learned_adjacency(self, prior: Tensor | None = None) -> Tensor:
        if self.use_prior and prior is None:
            raise ValueError("A prior adjacency is required when use_prior=True.")
        if not self.use_prior:
            prior = None
        return self.graph_learner(prior)

    def regularization_loss(self, prior: Tensor | None = None) -> Tensor:
        if not self.use_prior:
            prior = None
        return self.graph_learner.regularization_loss(prior)

    def residual_graph_weight(self) -> float:
        if not self.use_prior:
            return 1.0
        return float(self.graph_learner.residual_weight().detach().cpu())

    def forward(self, x: Tensor, adjacency: Tensor | None = None) -> Tensor:
        temporal_state = self.encode_temporal(x)
        graph = self.learned_adjacency(adjacency)
        normalized = normalize_adjacency(graph)
        neighbor_state = torch.einsum("ij,bjd->bid", normalized, temporal_state)
        correction = self.graph_projection(neighbor_state - temporal_state)
        scale = torch.tanh(self.graph_scale).view(1, 1, self.hidden_dim)
        fused = self.dropout(temporal_state + scale * correction)
        delta = self.delta_head(fused).permute(0, 2, 1)
        last_log_sales = x[
            :, -1, :, self.sales_feature_index
        ].unsqueeze(1)
        return last_log_sales + delta
