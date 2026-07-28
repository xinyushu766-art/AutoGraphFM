import torch

from autogfm.graph import normalize_adjacency


def test_normalized_adjacency_is_symmetric() -> None:
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
        ]
    )

    normalized = normalize_adjacency(adjacency)

    assert normalized.shape == (3, 3)
    assert torch.allclose(normalized, normalized.T)
    assert torch.all(torch.diag(normalized) > 0)


def test_identity_normalizes_to_identity() -> None:
    identity = torch.eye(4)
    normalized = normalize_adjacency(identity)

    assert torch.allclose(normalized, torch.eye(4))
