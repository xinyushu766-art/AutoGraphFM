from __future__ import annotations

import numpy as np
import torch

from autogfm.graph.sparse_residual import DirectSparseResidualGraphLearner
from autogfm.graph.stability_selection import (
    consensus_adjacency,
    edge_frequency,
    lagged_ridge_scores,
    select_candidate_mask,
    top_k_edge_mask,
)


def test_lagged_ridge_screening_recovers_directed_dependency() -> None:
    rng = np.random.default_rng(7)
    values = np.zeros((180, 4), dtype=np.float32)
    values[0] = rng.normal(size=4)
    for index in range(1, len(values)):
        values[index, 0] = 0.7 * values[index - 1, 0] + rng.normal(0, 0.05)
        values[index, 1] = 0.8 * values[index - 1, 0] + rng.normal(0, 0.05)
        values[index, 2] = 0.8 * values[index - 1, 1] + rng.normal(0, 0.05)
        values[index, 3] = 0.8 * values[index - 1, 2] + rng.normal(0, 0.05)
    scores = lagged_ridge_scores(
        values,
        139,
        ridge_alpha=0.5,
        use_differences=False,
    )
    assert int(np.argmax(scores[1])) == 0
    assert int(np.argmax(scores[2])) == 1
    assert int(np.argmax(scores[3])) == 2


def test_stability_selection_produces_fixed_row_sparse_support() -> None:
    graphs = [
        np.asarray(
            [
                [0.0, 0.8, 0.2],
                [0.7, 0.0, 0.3],
                [0.6, 0.4, 0.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [0.0, 0.7, 0.3],
                [0.6, 0.0, 0.4],
                [0.7, 0.3, 0.0],
            ],
            dtype=np.float32,
        ),
    ]
    frequency, diagnostics = edge_frequency(graphs, top_k=1)
    lagged = graphs[0]
    mask, _ = select_candidate_mask(frequency, lagged, top_k=1)
    consensus = consensus_adjacency(graphs, mask)
    assert np.all(mask.sum(axis=1) == 1)
    assert np.allclose(consensus.sum(axis=1), 1.0)
    assert diagnostics.mean_pairwise_jaccard == 1.0
    assert np.array_equal(top_k_edge_mask(consensus, 1), mask)


def test_candidate_mask_locks_sparse_learner_edge_set() -> None:
    mask = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [True, False, False, False],
        ]
    )
    learner = DirectSparseResidualGraphLearner(
        4,
        1,
        candidate_mask=mask,
    )
    with torch.no_grad():
        learner.residual_logits.normal_(mean=0.0, std=5.0)
    assert torch.equal(learner.edge_mask().cpu(), mask)
    adjacency = learner.residual_adjacency()
    assert torch.allclose(adjacency.sum(dim=1), torch.ones(4))
