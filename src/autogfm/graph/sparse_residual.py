"""Regularized direct sparse residual graph learning.

The learner keeps an interpretable static prior and learns only a bounded,
row-sparse residual relation matrix.  It is intentionally higher-rank than the
AGCRN/MTGNN embedding parameterizations because Stage 11 showed that the
low-rank learners could not reliably recover a known dependency graph.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _row_normalize(adjacency: Tensor) -> Tensor:
    graph = adjacency.clamp_min(0.0)
    row_sum = graph.sum(dim=-1, keepdim=True)
    identity = torch.eye(
        graph.shape[0], dtype=graph.dtype, device=graph.device
    )
    normalized = graph / row_sum.clamp_min(1e-8)
    empty = (row_sum <= 1e-8).to(graph.dtype)
    return normalized * (1.0 - empty) + identity * empty


class DirectSparseResidualGraphLearner(nn.Module):
    """Learn a direct Top-K residual graph around an optional static prior.

    The dense residual score matrix is not factorized through node embeddings.
    A bounded gate controls how far the final adjacency may move away from the
    prior.  The selected residual edges are row-normalized and the final graph
    remains non-negative and row-stochastic.
    """

    def __init__(
        self,
        num_nodes: int,
        top_k: int,
        *,
        initial_residual_weight: float = 0.15,
        l1_strength: float = 2e-4,
        deviation_strength: float = 2e-3,
        gate_strength: float = 2e-4,
        candidate_mask: Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2.")
        if not 1 <= top_k < num_nodes:
            raise ValueError("top_k must be in [1, num_nodes).")
        if not 0.0 < initial_residual_weight < 1.0:
            raise ValueError("initial_residual_weight must be in (0, 1).")
        if min(l1_strength, deviation_strength, gate_strength) < 0.0:
            raise ValueError("regularization strengths must be non-negative.")

        self.num_nodes = int(num_nodes)
        self.top_k = int(top_k)
        self.l1_strength = float(l1_strength)
        self.deviation_strength = float(deviation_strength)
        self.gate_strength = float(gate_strength)
        self.residual_logits = nn.Parameter(
            torch.randn(num_nodes, num_nodes) * 0.02 - 2.0
        )
        initial_logit = torch.logit(torch.tensor(initial_residual_weight))
        self.residual_weight_logit = nn.Parameter(initial_logit)
        if candidate_mask is None:
            mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
            mask.fill_diagonal_(False)
        else:
            mask = candidate_mask.detach().to(dtype=torch.bool, device='cpu')
            if mask.shape != (num_nodes, num_nodes):
                raise ValueError('candidate_mask has the wrong shape.')
            mask = mask.clone()
            mask.fill_diagonal_(False)
            if torch.any(mask.sum(dim=1) < self.top_k):
                raise ValueError(
                    'Each candidate_mask row must contain at least top_k edges.'
                )
        self.register_buffer('candidate_mask', mask)

    def residual_scores(self) -> Tensor:
        scores = F.softplus(self.residual_logits)
        diagonal = torch.eye(
            self.num_nodes, dtype=torch.bool, device=scores.device
        )
        valid = self.candidate_mask.to(scores.device) & ~diagonal
        return scores.masked_fill(~valid, 0.0)

    def residual_adjacency(self) -> Tensor:
        scores = self.residual_scores()
        masked_scores = scores.masked_fill(
            ~self.candidate_mask.to(scores.device), float('-inf')
        )
        values, indices = torch.topk(masked_scores, self.top_k, dim=-1)
        sparse = torch.zeros_like(scores)
        sparse.scatter_(1, indices, values)
        return _row_normalize(sparse)

    def residual_weight(self) -> Tensor:
        return torch.sigmoid(self.residual_weight_logit)

    def forward(self, prior: Tensor | None = None) -> Tensor:
        residual = self.residual_adjacency()
        if prior is None:
            return residual
        if prior.shape != residual.shape:
            raise ValueError("prior adjacency shape does not match residual graph.")
        normalized_prior = _row_normalize(prior)
        weight = self.residual_weight()
        mixed = (1.0 - weight) * normalized_prior + weight * residual
        return _row_normalize(mixed)

    def regularization_loss(self, prior: Tensor | None = None) -> Tensor:
        scores = self.residual_scores()
        loss = self.l1_strength * scores.mean()
        weight = self.residual_weight()
        if prior is not None:
            loss = loss + self.gate_strength * weight
        if prior is not None and self.deviation_strength > 0.0:
            normalized_prior = _row_normalize(prior)
            learned = self.forward(prior)
            loss = loss + self.deviation_strength * torch.mean(
                torch.abs(learned - normalized_prior)
            )
        return loss

    @torch.no_grad()
    def edge_mask(self) -> Tensor:
        scores = self.residual_scores()
        masked_scores = scores.masked_fill(
            ~self.candidate_mask.to(scores.device), float("-inf")
        )
        _, indices = torch.topk(masked_scores, self.top_k, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        return mask
