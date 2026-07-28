from autogfm.config import SyntheticMarketConfig, TrainingConfig
from autogfm.experiments import run_stage0_2d_residual_experiments


def test_residual_experiment_smoke(tmp_path) -> None:
    market_config = SyntheticMarketConfig(
        num_steps=28,
        num_vehicles=5,
        history=4,
        horizon=2,
        seed=5,
    )
    training_config = TrainingConfig(
        hidden_dim=6,
        batch_size=8,
        epochs=1,
        train_ratio=0.7,
        seed=5,
    )

    results = run_stage0_2d_residual_experiments(
        market_config=market_config,
        training_config=training_config,
        output_dir=tmp_path,
    )

    assert len(results) == 5
    assert results[0].model_name == "last_value"
    assert results[0].relative_wape_improvement_percent == 0.0
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "experiment_summary.md").exists()
