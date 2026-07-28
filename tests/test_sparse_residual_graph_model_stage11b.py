from __future__ import annotations

import pytest
import torch

from autogfm.models import SparseResidualGraphGRU


def test_sparse_residual_graph_gru_shapes_and_prior() -> None:
    model = SparseResidualGraphGRU(
        num_features=5,
        hidden_dim=8,
        horizon=3,
        num_nodes=6,
        top_k=2,
        use_prior=True,
    )
    x = torch.randn(4, 12, 6, 5)
    with pytest.raises(ValueError, match="prior adjacency"):
        model(x)
    output = model(x, torch.eye(6))
    graph = model.learned_adjacency(torch.eye(6))
    assert output.shape == (4, 3, 6)
    assert torch.allclose(graph.sum(-1), torch.ones(6), atol=1e-6)
    assert model.regularization_loss(torch.eye(6)).item() > 0.0


def test_direct_sparse_graph_does_not_require_prior() -> None:
    model = SparseResidualGraphGRU(
        num_features=3,
        hidden_dim=4,
        horizon=1,
        num_nodes=5,
        top_k=2,
        use_prior=False,
    )
    output = model(torch.randn(2, 6, 5, 3))
    assert output.shape == (2, 1, 5)
    assert model.residual_graph_weight() == 1.0
