from autogfm.config import SyntheticMarketConfig, TrainingConfig
from autogfm.experiments import run_stage0_2f_multiseed_experiments


def test_multiseed_runner_keeps_data_fixed(tmp_path) -> None:
    market_config = SyntheticMarketConfig(
        num_steps=24,
        num_vehicles=4,
        history=4,
        horizon=2,
        seed=11,
    )
    training_template = TrainingConfig(
        hidden_dim=4,
        batch_size=32,
        epochs=1,
        learning_rate=1e-3,
        seed=21,
    )

    bundle = run_stage0_2f_multiseed_experiments(
        market_config=market_config,
        training_template=training_template,
        model_seeds=[21, 22],
        graph_permutation_seed=1011,
        output_dir=tmp_path,
        patience=1,
        train_ratio=0.65,
        validation_ratio=0.15,
    )

    assert len(bundle.per_seed_results) == 10
    assert len(bundle.aggregate_metrics) == 5
    assert len(bundle.paired_comparisons) == 4

    last_value_wapes = {
        result.test_wape_percent
        for result in bundle.per_seed_results
        if result.model_name == "last_value"
    }
    assert len(last_value_wapes) == 1

    assert (tmp_path / "per_seed_metrics.csv").exists()
    assert (tmp_path / "aggregate_metrics.csv").exists()
    assert (tmp_path / "paired_comparisons.csv").exists()
    assert (tmp_path / "experiment_summary.md").exists()
