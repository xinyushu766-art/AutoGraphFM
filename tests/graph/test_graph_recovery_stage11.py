from __future__ import annotations

from pathlib import Path

from autogfm.graph import GraphRecoverySpec, run_graph_recovery_benchmark


def test_graph_recovery_benchmark_has_identifiability_control(tmp_path: Path) -> None:
    result = run_graph_recovery_benchmark(
        tmp_path / "recovery",
        GraphRecoverySpec(
            steps=120,
            train_steps=75,
            validation_steps=20,
            max_epochs=80,
            patience=15,
            seeds=(5,),
        ),
        overwrite=True,
    )
    names = {row["model_name"] for row in result.manifest["aggregate"]}
    assert "direct_sparse_adaptive" in names
    assert "mtgnn_adaptive" in names
    assert "regularized_sparse_residual" in names
    assert result.manifest["sparse_residual_recovery_status"] in {"go", "no-go"}
    assert result.manifest["evidence_scope"] == "synthetic_known_graph_only"
    assert result.manifest["vehicle_graph_claim_authorized"] is False
    assert (result.output_dir / "aggregate_metrics.csv").is_file()
