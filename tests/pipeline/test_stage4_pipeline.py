from datetime import date

from autogfm.forecasting import BacktestSpec
from autogfm.graph import GraphSpec
from autogfm.pipeline import (
    Stage3PipelineSpec,
    Stage4PipelineSpec,
    run_stage4_pipeline,
)
from autogfm.realdata import CanonicalizationConfig, PanelSpec
from autogfm.reporting import Stage4ReportSpec, build_stage4_audit_report
from autogfm.realdata.demo import generate_stage3_demo_data


def test_stage4_pipeline_builds_monitoring_and_audit(tmp_path) -> None:
    demo = generate_stage3_demo_data(
        tmp_path / "raw", num_series=8, num_months=60, seed=31
    )
    stage3 = Stage3PipelineSpec(
        canonicalization=CanonicalizationConfig(
            source_name="pipeline_demo", data_version="v1"
        ),
        panel=PanelSpec(
            start_month=date(2020, 1, 1),
            train_end=date(2023, 12, 1),
            validation_end=date(2024, 6, 1),
            test_end=date(2024, 12, 1),
            min_history_months=36,
            max_missing_ratio=0.15,
            max_trailing_missing_months=1,
        ),
        graph=GraphSpec(top_k=3),
        backtest=BacktestSpec(
            horizons=(1, 3), origin_step=3, bootstrap_samples=10
        ),
        run_neural=False,
    )
    result = run_stage4_pipeline(
        demo.sales_csv,
        demo.mapping_csv,
        demo.vehicles_csv,
        tmp_path / "run",
        Stage4PipelineSpec(stage3=stage3),
        overwrite=True,
    )
    assert result.manifest["engineering_status"] == "pass"
    assert result.manifest["evidence_scope"] == "demo_only"
    assert result.manifest["release_ready"] is False
    assert (result.output_dir / "07_monitoring/manifest.json").is_file()
    assert (result.output_dir / "08_report/AUDIT_REPORT.html").is_file()
    assert (result.output_dir / "stage4_manifest.json").is_file()

    relabeled = build_stage4_audit_report(
        result.output_dir,
        tmp_path / "synthetic_report",
        Stage4ReportSpec(evidence_scope="synthetic_only"),
        overwrite=True,
    )
    assert relabeled.decision["evidence_scope"] == "synthetic_only"
    assert relabeled.decision["release_ready"] is False


def test_release_policy_never_deploys_demo_stage4(tmp_path) -> None:
    from autogfm.governance import ReleaseSpec, prepare_release_candidate

    demo = generate_stage3_demo_data(
        tmp_path / "raw_release", num_series=8, num_months=60, seed=37
    )
    stage3 = Stage3PipelineSpec(
        canonicalization=CanonicalizationConfig(
            source_name="release_demo", data_version="v1"
        ),
        panel=PanelSpec(
            start_month=date(2020, 1, 1),
            train_end=date(2023, 12, 1),
            validation_end=date(2024, 6, 1),
            test_end=date(2024, 12, 1),
            min_history_months=36,
            max_missing_ratio=0.15,
            max_trailing_missing_months=1,
        ),
        graph=GraphSpec(top_k=3),
        backtest=BacktestSpec(
            horizons=(1, 3), origin_step=3, bootstrap_samples=10
        ),
        run_neural=False,
    )
    run = run_stage4_pipeline(
        demo.sales_csv,
        demo.mapping_csv,
        demo.vehicles_csv,
        tmp_path / "release_run",
        Stage4PipelineSpec(stage3=stage3),
        overwrite=True,
    )
    production = prepare_release_candidate(
        run.output_dir,
        tmp_path / "production_release",
        ReleaseSpec(mode="production"),
        overwrite=True,
    )
    shadow = prepare_release_candidate(
        run.output_dir,
        tmp_path / "shadow_release",
        ReleaseSpec(mode="shadow"),
        overwrite=True,
    )
    assert production.status == "blocked"
    assert production.deployable is False
    assert not (production.output_dir / "deployment_bundle").exists()
    assert shadow.status == "shadow"
    assert shadow.deployable is False
    assert (shadow.output_dir / "shadow_bundle/SHA256SUMS.txt").is_file()
