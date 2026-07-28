import torch

from autogfm.models import TemporalOnlyGRU


def test_temporal_gru_output_shape() -> None:
    model = TemporalOnlyGRU(
        num_features=3,
        hidden_dim=16,
        horizon=3,
        dropout=0.0,
    )
    x = torch.randn(4, 12, 10, 3)

    output = model(x)

    assert output.shape == (4, 3, 10)
