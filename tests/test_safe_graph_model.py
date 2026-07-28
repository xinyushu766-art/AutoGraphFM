import torch

from autogfm.models import (
    LastValueForecaster,
    SafeGraphResidualGRU,
)


def test_safe_graph_starts_as_last_value() -> None:
    x = torch.randn(3, 8, 5, 3)
    adjacency = torch.rand(5, 5)
    adjacency = 0.5 * (adjacency + adjacency.T)

    model = SafeGraphResidualGRU(
        num_features=3,
        hidden_dim=12,
        horizon=2,
        dropout=0.0,
    )
    baseline = LastValueForecaster(horizon=2)

    model.eval()

    assert torch.allclose(
        model(x, adjacency),
        baseline(x),
    )


def test_safe_graph_scale_starts_at_zero() -> None:
    model = SafeGraphResidualGRU(
        num_features=3,
        hidden_dim=12,
        horizon=2,
    )

    assert torch.all(model.graph_scale == 0.0)


def test_identity_graph_produces_zero_graph_correction() -> None:
    x = torch.randn(2, 8, 5, 3)
    adjacency = torch.eye(5)

    model = SafeGraphResidualGRU(
        num_features=3,
        hidden_dim=12,
        horizon=2,
        dropout=0.0,
    )
    model.eval()

    temporal_state = model.encode_temporal(x)
    neighbor_state = torch.einsum(
        "ij,bjd->bid",
        adjacency,
        temporal_state,
    )
    graph_correction = model.graph_projection(
        neighbor_state - temporal_state
    )

    assert torch.allclose(
        graph_correction,
        torch.zeros_like(graph_correction),
    )
