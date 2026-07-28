from __future__ import annotations

import numpy as np
import torch

from autogfm.graph import DirectSparseResidualGraphLearner
from autogfm.graph.recovery import corrupted_prior_graph, directed_ring_graph


def test_sparse_residual_graph_rows_and_regularization() -> None:
    learner = DirectSparseResidualGraphLearner(
        num_nodes=6,
        top_k=2,
        initial_residual_weight=0.2,
    )
    prior = torch.eye(6)
    graph = learner(prior)
    assert graph.shape == (6, 6)
    assert torch.allclose(graph.sum(-1), torch.ones(6), atol=1e-6)
    assert 0.0 < float(learner.residual_weight().detach()) < 1.0
    assert learner.regularization_loss(prior).item() > 0.0


def test_corrupted_prior_retains_partial_truth() -> None:
    truth = directed_ring_graph(8, 2)
    prior = corrupted_prior_graph(truth, seed=9)
    assert prior.shape == truth.shape
    assert np.allclose(prior.sum(axis=1), 1.0)
    overlap = 0
    for row in range(truth.shape[0]):
        overlap += len(
            set(np.flatnonzero(prior[row] > 0))
            & set(np.flatnonzero(truth[row] > 0))
        )
    assert overlap == truth.shape[0]
