from __future__ import annotations

from autogfm.experiments import (
    AdaptiveGraphExperimentSpec,
    run_adaptive_graph_experiments,
)


def test_adaptive_graph_runner_emits_falsification_controls(
    stage3_artifacts: dict[str, object], tmp_path
) -> None:
    result = run_adaptive_graph_experiments(
        stage3_artifacts["panel"].output_dir,
        stage3_artifacts["graph"].output_dir,
        tmp_path / "stage11_vehicle",
        AdaptiveGraphExperimentSpec(
            history=12,
            horizon=3,
            hidden_dim=8,
            embedding_dim=4,
            top_k=3,
            batch_size=4,
            max_epochs=2,
            patience=1,
            seeds=(3,),
            bootstrap_samples=100,
        ),
        overwrite=True,
    )
    assert result.manifest["engineering_status"] == "pass"
    assert result.manifest["graph_research_claim"] in {"go", "no-claim"}
    assert result.manifest["best_adaptive_model"].startswith(("adaptive_", "hybrid_"))
    assert set(result.manifest["claim_checks"]) == {
        "beats_residual_temporal_gru",
        "beats_safe_static_graph_gru",
        "beats_safe_identity_graph_gru",
        "beats_safe_permuted_graph_gru",
    }
    assert (result.output_dir / "paired_bootstrap.csv").is_file()
