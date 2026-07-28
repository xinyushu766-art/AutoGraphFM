import pytest
import torch

from autogfm.models import StaticGraphGRU


def test_static_graph_gru_output_shape() -> None:
    model = StaticGraphGRU(
        num_features=3,
        hidden_dim=16,
        horizon=3,
        dropout=0.0,
    )
    x = torch.randn(4, 12, 10, 3)
    adjacency = torch.rand(10, 10)
    adjacency = 0.5 * (adjacency + adjacency.T)

    output = model(x, adjacency)

    assert output.shape == (4, 3, 10)


def test_static_graph_gru_requires_graph() -> None:
    model = StaticGraphGRU(
        num_features=3,
        hidden_dim=16,
        horizon=3,
    )
    x = torch.randn(4, 12, 10, 3)

    with pytest.raises(ValueError, match="requires an adjacency matrix"):
        model(x, None)
