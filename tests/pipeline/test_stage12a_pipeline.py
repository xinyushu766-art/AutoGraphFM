from __future__ import annotations

from autogfm.graph import StabilityRecoverySpec
from autogfm.graph.recovery import GraphRecoverySpec
from autogfm.pipeline import Stage12ASpec, run_stage12a_pipeline


def test_stage12a_recovery_only_pipeline(tmp_path) -> None:
    result = run_stage12a_pipeline(
        tmp_path / "stage12a",
        Stage12ASpec(
            recovery=StabilityRecoverySpec(
                base=GraphRecoverySpec(
                    steps=100,
                    train_steps=60,
                    validation_steps=18,
                    max_epochs=120,
                    patience=20,
                    seeds=(42, 43),
                ),
                minimum_edge_precision=0.45,
                maximum_mse_ratio_to_direct=1.30,
            )
        ),
        overwrite=True,
    )
    assert result.manifest["engineering_status"] == "pass"
    assert result.manifest["final_graph_research_claim"] == "no-claim"
    assert result.manifest["vehicle_stability_graph_status"] == "not_evaluated"
    assert (result.output_dir / "stage12a_manifest.json").is_file()
