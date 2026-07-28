import torch

from autogfm.graph import make_identity_graph, make_permuted_graph


def test_identity_graph_contains_no_cross_node_edges() -> None:
    adjacency = torch.ones(5, 5) - torch.eye(5)
    identity = make_identity_graph(adjacency)

    assert torch.allclose(identity, torch.eye(5))


def test_permuted_graph_preserves_global_statistics() -> None:
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 0.0, 2.0],
            [1.0, 0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [2.0, 0.0, 4.0, 0.0],
        ]
    )
    permuted = make_permuted_graph(adjacency, seed=7)

    assert torch.allclose(permuted, permuted.T)
    assert torch.allclose(
        torch.sort(permuted.flatten()).values,
        torch.sort(adjacency.flatten()).values,
    )
