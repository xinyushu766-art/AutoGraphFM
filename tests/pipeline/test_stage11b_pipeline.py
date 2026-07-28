from __future__ import annotations

from autogfm.experiments import SparseGraphExperimentSpec
from autogfm.graph import GraphRecoverySpec
from autogfm.pipeline import Stage11BSpec, run_stage11b_pipeline


def test_stage11b_pipeline_promotes_architecture_not_demo_claim(
    stage3_artifacts: dict[str, object], tmp_path
) -> None:
    result = run_stage11b_pipeline(
        tmp_path / "stage11b",
        Stage11BSpec(
            recovery=GraphRecoverySpec(
                steps=120,
                train_steps=75,
                validation_steps=20,
                max_epochs=100,
                patience=20,
                seeds=(5,),
            ),
            vehicle_experiment=SparseGraphExperimentSpec(
                hidden_dim=8,
                embedding_dim=4,
                top_k=3,
                max_epochs=2,
                patience=1,
                seeds=(5, 7),
                bootstrap_samples=100,
                consensus_ratio=1.0,
                min_pairwise_jaccard=0.0,
                min_stable_edge_ratio=0.0,
            ),
        ),
        panel_dir=stage3_artifacts["panel"].output_dir,
        graph_dir=stage3_artifacts["graph"].output_dir,
        evidence_scope="demo_only",
        overwrite=True,
    )
    assert result.manifest["engineering_status"] == "pass"
    assert result.manifest["sparse_recovery_status"] in {"go", "no-go"}
    assert result.manifest["architecture_decision"] in {
        "sparse_residual_graph_eligible_for_real_vehicle_challenge",
        "refine_sparse_residual_regularization",
    }
    assert result.manifest["graph_claim_authorized"] is False
    assert result.manifest["final_graph_research_claim"] == "no-claim"
    assert (result.output_dir / "STAGE11B_REPORT.md").is_file()
