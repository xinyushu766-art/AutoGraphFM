"""Execute and validate the complete Stage 4 release workflow."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from autogfm.experiments import RealPanelExperimentSpec
from autogfm.forecasting import BacktestSpec
from autogfm.graph import GraphSpec
from autogfm.pipeline import (
    Stage3PipelineSpec,
    Stage4PipelineSpec,
    run_stage4_pipeline,
)
from autogfm.realdata import CanonicalizationConfig, PanelSpec
from autogfm.realdata.common import read_csv_rows, write_json
from autogfm.realdata.demo import generate_stage3_demo_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage4_validation"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Validation output exists: {args.output_dir}. Use --overwrite."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    demo = generate_stage3_demo_data(args.output_dir / "demo_data")
    if args.full:
        backtest = BacktestSpec(origin_step=1, bootstrap_samples=500)
        neural = RealPanelExperimentSpec(
            max_epochs=30,
            patience=6,
            seeds=(42, 43, 44),
        )
    else:
        backtest = BacktestSpec(origin_step=3, bootstrap_samples=100)
        neural = RealPanelExperimentSpec(
            hidden_dim=16,
            max_epochs=8,
            patience=3,
            seeds=(42,),
        )
    stage3 = Stage3PipelineSpec(
        canonicalization=CanonicalizationConfig(
            source_name="stage4_demo",
            data_version="v1",
        ),
        panel=PanelSpec(
            start_month=date(2020, 1, 1),
            train_end=date(2023, 12, 1),
            validation_end=date(2024, 12, 1),
            test_end=date(2025, 12, 1),
            min_history_months=36,
            max_missing_ratio=0.10,
            max_trailing_missing_months=1,
        ),
        graph=GraphSpec(top_k=5),
        backtest=backtest,
        neural=neural,
        run_neural=True,
    )
    run = run_stage4_pipeline(
        demo.sales_csv,
        demo.mapping_csv,
        demo.vehicles_csv,
        args.output_dir / "run",
        Stage4PipelineSpec(stage3=stage3),
        overwrite=True,
    )

    required = [
        "run/01_canonical/manifest.json",
        "run/02_panel/panel.npz",
        "run/03_graph/adjacency.npy",
        "run/04_backtest/leaderboard.csv",
        "run/04_backtest/selected_model_test_metrics.csv",
        "run/05_neural/aggregate_metrics.csv",
        "run/06_forecast/production_forecast.csv",
        "run/06_forecast/MODEL_CARD.md",
        "run/07_monitoring/monitoring_report.json",
        "run/07_monitoring/MONITORING_REPORT.md",
        "run/08_report/AUDIT_REPORT.html",
        "run/08_report/AUDIT_SUMMARY.md",
        "run/STAGE4_RUN_REPORT.md",
        "run/stage4_manifest.json",
    ]
    missing = [path for path in required if not (args.output_dir / path).is_file()]
    _, forecast_alerts, _ = read_csv_rows(
        args.output_dir / "run/07_monitoring/forecast_alerts.csv"
    )
    audit_html = (args.output_dir / "run/08_report/AUDIT_REPORT.html").read_text(
        encoding="utf-8"
    )
    quality = run.stage3.manifest["quality"]
    panel_quality = run.stage3.manifest["panel_quality"]
    graph = run.stage3.manifest["graph_diagnostics"]
    checks = {
        "required_artifacts_present": not missing,
        "malformed_row_quarantined": quality["rejected_sales_rows"] >= 1,
        "duplicate_key_audited": quality["duplicate_monthly_keys_aggregated"] >= 1,
        "panel_has_multiple_series": panel_quality["included_series"] >= 8,
        "panel_observed_ratio": panel_quality["observed_ratio"] > 0.90,
        "graph_has_no_isolates": graph["isolated_nodes"] == 0,
        "graph_is_mostly_connected": graph["largest_component_ratio"] >= 0.90,
        "validation_selected_models_exist": bool(run.manifest["selected_models"]),
        "production_forecast_built": bool(
            run.manifest["production_forecast_id"]
        ),
        "monitoring_status_is_explicit": run.manifest["deployment_status"]
        in {"pass", "warn", "block"},
        "forecast_sanity_has_no_alerts": not forecast_alerts,
        "demo_evidence_is_not_release_ready": (
            run.manifest["evidence_scope"] == "demo_only"
            and run.manifest["release_ready"] is False
        ),
        "graph_claim_gate_is_conservative": (
            run.manifest["graph_research_claim"] == "no-claim"
        ),
        "standalone_html_is_complete": (
            "AutoGraphFM Stage 4 Audit" in audit_html
            and "Graph falsification controls" in audit_html
            and "Production forecast sample" in audit_html
        ),
    }
    passed = all(checks.values())
    payload = {
        "status": "passed" if passed else "failed",
        "run_id": run.run_id,
        "stage3_run_id": run.stage3.run_id,
        "checks": checks,
        "missing": missing,
        "deployment_status": run.manifest["deployment_status"],
        "block_reasons": run.manifest["block_reasons"],
        "warn_reasons": run.manifest["warn_reasons"],
        "selected_models": run.manifest["selected_models"],
        "graph_research_claim": run.manifest["graph_research_claim"],
        "evidence_scope": run.manifest["evidence_scope"],
    }
    write_json(args.output_dir / "validation_result.json", payload)
    lines = [
        "# Stage 4 Validation Report",
        "",
        f"- Validation status: **{payload['status'].upper()}**",
        f"- Stage 4 run ID: `{run.run_id}`",
        f"- Deployment gate: **{payload['deployment_status'].upper()}**",
        f"- Graph claim: **{payload['graph_research_claim'].upper()}**",
        f"- Evidence scope: **{payload['evidence_scope'].upper()}**",
        "",
        "## Acceptance checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## Gate outcomes",
            "",
            f"- Blocking reasons: `{payload['block_reasons']}`",
            f"- Warning reasons: `{payload['warn_reasons']}`",
            f"- Validation-selected models: `{payload['selected_models']}`",
            "",
            "The validation checks that adverse demo outcomes are surfaced rather than suppressed. Demo rankings are not evidence about the user's real automotive data.",
            "",
        ]
    )
    (args.output_dir / "STAGE4_VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
