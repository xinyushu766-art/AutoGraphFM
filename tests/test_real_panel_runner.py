from autogfm.experiments import RealPanelExperimentSpec, run_real_panel_experiments


def test_real_panel_runner_emits_all_graph_controls(stage3_artifacts: dict[str, object], tmp_path) -> None:
    result = run_real_panel_experiments(
        stage3_artifacts["panel"].output_dir,
        stage3_artifacts["graph"].output_dir,
        tmp_path / "neural",
        RealPanelExperimentSpec(
            history=12,
            horizon=3,
            hidden_dim=8,
            batch_size=4,
            max_epochs=2,
            patience=1,
            seeds=(3,),
        ),
        overwrite=True,
    )
    assert result.manifest["winner"]
    assert set(result.manifest["claim_gate"]) == {
        "beats_last_value",
        "beats_temporal",
        "beats_identity",
        "beats_permuted",
    }
    assert (result.output_dir / "aggregate_metrics.csv").is_file()
