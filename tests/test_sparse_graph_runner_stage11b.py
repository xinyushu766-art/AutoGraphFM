from __future__ import annotations

from autogfm.experiments import (
    SparseGraphExperimentSpec,
    run_sparse_graph_experiments,
)


def test_sparse_graph_runner_emits_consensus_and_controls(
    stage3_artifacts: dict[str, object], tmp_path
) -> None:
    result = run_sparse_graph_experiments(
        stage3_artifacts["panel"].output_dir,
        stage3_artifacts["graph"].output_dir,
        tmp_path / "stage11b_vehicle",
        SparseGraphExperimentSpec(
            history=12,
            horizon=3,
            hidden_dim=8,
            embedding_dim=4,
            top_k=3,
            batch_size=4,
            max_epochs=2,
            patience=1,
            seeds=(3, 5),
            bootstrap_samples=100,
            consensus_ratio=1.0,
            min_pairwise_jaccard=0.0,
            min_stable_edge_ratio=0.0,
        ),
        evidence_scope="demo_only",
        overwrite=True,
    )
    assert result.manifest["engineering_status"] == "pass"
    assert result.manifest["graph_research_claim"] in {"go", "no-claim"}
    assert result.manifest["best_sparse_model"] in {
        "direct_sparse_graph_gru",
        "residual_sparse_static_graph_gru",
    }
    assert set(result.manifest["performance_checks"]) == {
        "beats_residual_temporal_gru",
        "beats_safe_static_graph_gru",
        "beats_safe_identity_graph_gru",
        "beats_safe_permuted_graph_gru",
        "beats_adaptive_mtgnn_graph_gru",
    }
    assert (result.output_dir / "edge_stability.csv").is_file()
    assert (
        result.output_dir
        / "consensus_graphs/residual_sparse_static_graph_gru_consensus.npy"
    ).is_file()
