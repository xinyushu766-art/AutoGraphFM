from __future__ import annotations

import torch

from autogfm.graph.adaptive import AdaptiveGraphLearner, blend_adjacencies


def test_agcrn_adjacency_is_row_normalized() -> None:
    torch.manual_seed(1)
    adjacency = AdaptiveGraphLearner(9, 4, mode="agcrn")()
    assert adjacency.shape == (9, 9)
    assert torch.all(adjacency >= 0)
    assert torch.allclose(adjacency.sum(-1), torch.ones(9), atol=1e-6)


def test_mtgnn_adjacency_is_directed_sparse_and_normalized() -> None:
    torch.manual_seed(2)
    adjacency = AdaptiveGraphLearner(10, 5, mode="mtgnn", top_k=3)()
    assert adjacency.shape == (10, 10)
    assert torch.all(adjacency >= 0)
    assert torch.allclose(adjacency.sum(-1), torch.ones(10), atol=1e-6)
    assert int((adjacency > 0).sum(-1).max()) <= 3
    assert not torch.allclose(adjacency, adjacency.transpose(0, 1))


def test_prior_blend_is_row_normalized() -> None:
    torch.manual_seed(3)
    learned = AdaptiveGraphLearner(6, 3, mode="agcrn")()
    prior = torch.eye(6)
    blended = blend_adjacencies(learned, prior, 0.4)
    assert torch.allclose(blended.sum(-1), torch.ones(6), atol=1e-6)
    assert torch.all(blended >= 0)
