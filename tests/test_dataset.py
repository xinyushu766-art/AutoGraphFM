from autogfm.config import SyntheticMarketConfig
from autogfm.data import SyntheticAutomotiveDataset


def test_dataset_tensor_contract() -> None:
    config = SyntheticMarketConfig(
        num_steps=40,
        num_vehicles=8,
        history=6,
        horizon=2,
    )
    dataset = SyntheticAutomotiveDataset(config)

    x, y = dataset[0]

    assert x.shape == (6, 8, 3)
    assert y.shape == (2, 8)
    assert dataset.adjacency.shape == (8, 8)


def test_dataset_window_count() -> None:
    config = SyntheticMarketConfig(
        num_steps=40,
        num_vehicles=8,
        history=6,
        horizon=2,
    )
    dataset = SyntheticAutomotiveDataset(config)

    assert len(dataset) == 40 - 6 - 2 + 1
