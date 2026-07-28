from autogfm.config import SyntheticMarketConfig
from autogfm.data import SyntheticAutomotiveDataset
from autogfm.experiments.safe_graph_runner import (
    chronological_three_way_split,
)


def _target_interval(
    dataset: SyntheticAutomotiveDataset,
    subset_index: int,
) -> tuple[int, int]:
    start = dataset.start_indices[subset_index]
    target_start = start + dataset.config.history
    target_end = target_start + dataset.config.horizon - 1
    return target_start, target_end


def test_three_way_split_has_no_target_overlap() -> None:
    config = SyntheticMarketConfig(
        num_steps=60,
        num_vehicles=6,
        history=6,
        horizon=3,
        seed=7,
    )
    dataset = SyntheticAutomotiveDataset(config)
    split = chronological_three_way_split(dataset)

    _, train_target_end = _target_interval(
        dataset,
        split.train.indices[-1],
    )
    validation_target_start, validation_target_end = _target_interval(
        dataset,
        split.validation.indices[0],
    )
    test_target_start, _ = _target_interval(
        dataset,
        split.test.indices[0],
    )

    assert train_target_end < validation_target_start
    assert validation_target_end < test_target_start
