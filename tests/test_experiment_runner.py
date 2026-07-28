from autogfm.config import SyntheticMarketConfig, TrainingConfig
from autogfm.experiments import run_stage0_2_experiments


def test_stage0_2_experiment_smoke(tmp_path) -> None:
    market_config = SyntheticMarketConfig(
        num_steps=20,
        num_vehicles=4,
        history=3,
        horizon=1,
        seed=3,
    )
    training_config = TrainingConfig(
        hidden_dim=4,
        batch_size=16,
        epochs=1,
        train_ratio=0.75,
        seed=3,
    )

    results = run_stage0_2_experiments(
        market_config=market_config,
        training_config=training_config,
        output_dir=tmp_path,
    )

    assert len(results) == 5
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "experiment_summary.md").exists()
