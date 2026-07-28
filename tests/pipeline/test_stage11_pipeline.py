from __future__ import annotations

from autogfm.experiments import AdaptiveGraphExperimentSpec
from autogfm.graph import GraphRecoverySpec
from autogfm.pipeline import Stage11Spec, run_stage11_pipeline


def test_stage11_pipeline_never_promotes_demo_evidence(
    stage3_artifacts: dict[str, object], tmp_path
) -> None:
    result = run_stage11_pipeline(
        tmp_path / "stage11",
        Stage11Spec(
            recovery=GraphRecoverySpec(
                steps=120,
                train_steps=75,
                validation_steps=20,
                max_epochs=80,
                patience=15,
                seeds=(5,),
            ),
            vehicle_experiment=AdaptiveGraphExperimentSpec(
                hidden_dim=8,
                embedding_dim=4,
                top_k=3,
                max_epochs=2,
                patience=1,
                seeds=(5,),
                bootstrap_samples=100,
            ),
        ),
        panel_dir=stage3_artifacts["panel"].output_dir,
        graph_dir=stage3_artifacts["graph"].output_dir,
        overwrite=True,
    )
    assert result.manifest["engineering_status"] == "pass"
    assert result.manifest["graph_claim_authorized"] is False
    assert result.manifest["final_graph_research_claim"] == "no-claim"
    assert (result.output_dir / "STAGE11_REPORT.md").is_file()
