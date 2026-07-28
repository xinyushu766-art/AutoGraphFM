import torch

from autogfm.config import SyntheticMarketConfig
from autogfm.data import SyntheticAutomotiveDataset
from autogfm.graph import make_permuted_graph


def test_fixed_data_seed_reproduces_identical_market() -> None:
    config_a = SyntheticMarketConfig(seed=42)
    config_b = SyntheticMarketConfig(seed=42)

    dataset_a = SyntheticAutomotiveDataset(config_a)
    dataset_b = SyntheticAutomotiveDataset(config_b)

    assert torch.equal(
        dataset_a.market.features,
        dataset_b.market.features,
    )
    assert torch.equal(
        dataset_a.adjacency,
        dataset_b.adjacency,
    )


def test_fixed_permutation_seed_reproduces_control_graph() -> None:
    dataset = SyntheticAutomotiveDataset(
        SyntheticMarketConfig(seed=42)
    )

    first = make_permuted_graph(dataset.adjacency, seed=1042)
    second = make_permuted_graph(dataset.adjacency, seed=1042)

    assert torch.equal(first, second)
