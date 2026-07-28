import torch

from autogfm.models import (
    LastValueForecaster,
    ResidualStaticGraphGRU,
    ResidualTemporalGRU,
)


def test_untrained_residual_temporal_equals_last_value() -> None:
    x = torch.randn(3, 8, 5, 3)

    residual_model = ResidualTemporalGRU(
        num_features=3,
        hidden_dim=12,
        horizon=2,
        dropout=0.0,
    )
    last_value = LastValueForecaster(horizon=2)

    residual_model.eval()

    assert torch.allclose(
        residual_model(x),
        last_value(x),
    )


def test_untrained_residual_graph_equals_last_value() -> None:
    x = torch.randn(3, 8, 5, 3)
    adjacency = torch.rand(5, 5)
    adjacency = 0.5 * (adjacency + adjacency.T)

    residual_model = ResidualStaticGraphGRU(
        num_features=3,
        hidden_dim=12,
        horizon=2,
        dropout=0.0,
    )
    last_value = LastValueForecaster(horizon=2)

    residual_model.eval()

    assert torch.allclose(
        residual_model(x, adjacency),
        last_value(x),
    )


def test_residual_models_output_shape() -> None:
    x = torch.randn(4, 12, 6, 3)
    adjacency = torch.eye(6)

    temporal_model = ResidualTemporalGRU(
        num_features=3,
        hidden_dim=16,
        horizon=3,
    )
    graph_model = ResidualStaticGraphGRU(
        num_features=3,
        hidden_dim=16,
        horizon=3,
    )

    assert temporal_model(x).shape == (4, 3, 6)
    assert graph_model(x, adjacency).shape == (4, 3, 6)
