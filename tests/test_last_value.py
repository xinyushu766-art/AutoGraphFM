import torch

from autogfm.models import LastValueForecaster


def test_last_value_repeats_last_observation() -> None:
    model = LastValueForecaster(horizon=3)

    x = torch.zeros(2, 4, 5, 3)
    x[:, -1, :, 0] = 7.0

    prediction = model(x)

    assert prediction.shape == (2, 3, 5)
    assert torch.all(prediction == 7.0)
