from __future__ import annotations

import pytest
import torch

from autogfm.models import AdaptiveGraphResidualGRU


def test_adaptive_graph_residual_gru_shapes_and_rows() -> None:
    model = AdaptiveGraphResidualGRU(
        num_features=5,
        hidden_dim=8,
        horizon=3,
        num_nodes=6,
        graph_mode="mtgnn",
        top_k=2,
    )
    x = torch.randn(4, 12, 6, 5)
    output = model(x)
    adjacency = model.learned_adjacency()
    assert output.shape == (4, 3, 6)
    assert adjacency.shape == (6, 6)
    assert torch.allclose(adjacency.sum(-1), torch.ones(6), atol=1e-5)
    assert int((adjacency > 0).sum(-1).max()) <= 2


def test_hybrid_adaptive_graph_requires_matching_prior() -> None:
    model = AdaptiveGraphResidualGRU(
        num_features=3,
        hidden_dim=4,
        horizon=1,
        num_nodes=5,
        graph_mode="agcrn",
        use_prior=True,
    )
    x = torch.randn(2, 6, 5, 3)
    with pytest.raises(ValueError, match="prior adjacency"):
        model(x)
    with pytest.raises(ValueError, match="shape"):
        model(x, torch.eye(4))
    output = model(x, torch.eye(5))
    assert output.shape == (2, 1, 5)
