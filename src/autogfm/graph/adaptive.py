"""Adaptive graph learners distilled from AGCRN and MTGNN mechanisms."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveGraphLearner(nn.Module):
    """Learn a row-normalized adjacency without requiring a fixed graph.

    ``agcrn`` uses a softmax of ReLU node-embedding similarities.
    ``mtgnn`` uses an anti-symmetric directed score and row-wise Top-K sparsity.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 8,
        *,
        mode: str = "agcrn",
        top_k: int | None = None,
        alpha: float = 3.0,
    ) -> None:
        super().__init__()
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        if mode not in {"agcrn", "mtgnn"}:
            raise ValueError("mode must be agcrn or mtgnn.")
        self.num_nodes = int(num_nodes)
        self.mode = mode
        self.top_k = min(top_k or num_nodes, num_nodes)
        self.alpha = float(alpha)
        self.embedding_1 = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        if mode == "mtgnn":
            self.embedding_2 = nn.Parameter(torch.randn(num_nodes, embedding_dim))
            self.linear_1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        else:
            self.register_parameter("embedding_2", None)
            self.linear_1 = nn.Identity()
            self.linear_2 = nn.Identity()

    def forward(self) -> torch.Tensor:
        if self.mode == "agcrn":
            scores = F.relu(self.embedding_1 @ self.embedding_1.transpose(0, 1))
            return F.softmax(scores, dim=-1)

        node_1 = torch.tanh(self.alpha * self.linear_1(self.embedding_1))
        assert self.embedding_2 is not None
        node_2 = torch.tanh(self.alpha * self.linear_2(self.embedding_2))
        anti_symmetric = node_1 @ node_2.transpose(0, 1)
        anti_symmetric = anti_symmetric - node_2 @ node_1.transpose(0, 1)
        scores = F.relu(torch.tanh(self.alpha * anti_symmetric))
        if self.top_k < self.num_nodes:
            _, indices = torch.topk(scores, self.top_k, dim=-1)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, indices, 1.0)
            scores = scores * mask
        row_sum = scores.sum(dim=-1, keepdim=True)
        identity = torch.eye(
            self.num_nodes,
            dtype=scores.dtype,
            device=scores.device,
        )
        normalized = scores / row_sum.clamp_min(1e-8)
        empty_rows = (row_sum <= 1e-8).to(scores.dtype)
        return normalized * (1.0 - empty_rows) + identity * empty_rows


def blend_adjacencies(
    learned: torch.Tensor,
    prior: torch.Tensor,
    learned_weight: float | torch.Tensor,
) -> torch.Tensor:
    """Blend a learned graph with a normalized prior and renormalize rows."""
    if learned.shape != prior.shape or learned.ndim != 2:
        raise ValueError("learned and prior adjacency matrices must match.")
    weight = torch.as_tensor(
        learned_weight,
        dtype=learned.dtype,
        device=learned.device,
    ).clamp(0.0, 1.0)
    prior = prior.clamp_min(0.0)
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    mixed = weight * learned + (1.0 - weight) * prior
    return mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-8)
