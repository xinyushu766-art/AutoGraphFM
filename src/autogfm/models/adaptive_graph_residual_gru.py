"""Residual GRU with a trainable AGCRN/MTGNN-style graph learner."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from autogfm.graph.adaptive import AdaptiveGraphLearner, blend_adjacencies
from autogfm.graph.ops import normalize_adjacency


class AdaptiveGraphResidualGRU(nn.Module):
    """Forecast node-level log sales with a conservative learned graph branch.

    The temporal residual path is shared with :class:`SafeGraphResidualGRU`.
    The relation matrix is learned from node embeddings using either the
    AGCRN similarity mechanism or the MTGNN directed Top-K mechanism.  A
    supplied prior can be blended with the learned graph through a trainable
    gate.  Both the forecast correction and graph influence start at zero, so
    initialization remains equivalent to Last Value.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        horizon: int,
        num_nodes: int,
        *,
        graph_mode: str = "agcrn",
        embedding_dim: int = 8,
        top_k: int | None = None,
        use_prior: bool = False,
        sales_feature_index: int = 0,
        dropout: float = 0.1,
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
        self.graph_learner = AdaptiveGraphLearner(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            mode=graph_mode,
            top_k=top_k,
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
        )
        self.graph_scale = nn.Parameter(torch.zeros(hidden_dim))
        self.prior_logit = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.delta_head = nn.Linear(hidden_dim, horizon)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def encode_temporal(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, history, nodes, features].")
        batch_size, history, num_nodes, num_features = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, got {num_nodes}."
            )
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
        learned = self.graph_learner()
        if not self.use_prior:
            return learned
        if prior is None:
            raise ValueError("A prior adjacency is required when use_prior=True.")
        if prior.shape != learned.shape:
            raise ValueError("prior adjacency shape does not match learned graph.")
        learned_weight = torch.sigmoid(self.prior_logit)
        return blend_adjacencies(learned, prior, learned_weight)

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
