"""Trace every important tensor shape in the Stage 0.2 graph model."""

import torch
from torch.utils.data import DataLoader

from autogfm.config import SyntheticMarketConfig
from autogfm.data import SyntheticAutomotiveDataset
from autogfm.graph import normalize_adjacency
from autogfm.models import StaticGraphGRU


def print_shape(name: str, tensor: torch.Tensor) -> None:
    """Print a tensor name and its shape."""
    print(f"{name:<24} {list(tensor.shape)}")


def main() -> None:
    config = SyntheticMarketConfig()
    dataset = SyntheticAutomotiveDataset(config)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    x, y = next(iter(loader))
    adjacency = dataset.adjacency

    model = StaticGraphGRU(
        num_features=config.num_features,
        hidden_dim=32,
        horizon=config.horizon,
        dropout=0.0,
    )
    model.eval()

    print("\n=== Original tensors ===")
    print_shape("x", x)
    print_shape("y", y)
    print_shape("adjacency", adjacency)

    batch_size, history, num_vehicles, num_features = x.shape
    permuted = x.permute(0, 2, 1, 3)
    node_sequences = permuted.reshape(
        batch_size * num_vehicles,
        history,
        num_features,
    )

    print("\n=== Temporal encoding ===")
    print_shape("permuted", permuted)
    print_shape("node_sequences", node_sequences)

    _, final_hidden = model.temporal_encoder(node_sequences)
    temporal_state = final_hidden[-1].reshape(
        batch_size,
        num_vehicles,
        model.hidden_dim,
    )

    print_shape("final_hidden", final_hidden)
    print_shape("temporal_state", temporal_state)

    print("\n=== Graph message passing ===")
    normalized_adjacency = normalize_adjacency(adjacency)
    graph_state = torch.einsum(
        "ij,bjd->bid",
        normalized_adjacency,
        temporal_state,
    )
    graph_state = model.graph_projection(graph_state)

    print_shape("normalized_adjacency", normalized_adjacency)
    print_shape("graph_state", graph_state)

    print("\n=== Gated fusion ===")
    concatenated = torch.cat([temporal_state, graph_state], dim=-1)
    gate = torch.sigmoid(model.fusion_gate(concatenated))
    fused_state = gate * temporal_state + (1.0 - gate) * graph_state

    print_shape("concatenated", concatenated)
    print_shape("gate", gate)
    print_shape("fused_state", fused_state)

    print("\n=== Forecast head ===")
    forecast_before_permute = model.forecast_head(fused_state)
    forecast = forecast_before_permute.permute(0, 2, 1)

    print_shape("forecast_before_permute", forecast_before_permute)
    print_shape("forecast", forecast)

    with torch.no_grad():
        model_output = model(x, adjacency)

    print_shape("model_output", model_output)

    assert forecast.shape == y.shape
    assert torch.allclose(forecast, model_output)

    print("\nShape trace completed successfully.")


if __name__ == "__main__":
    main()
