"""Create demo data, run Stage 3, and enforce release acceptance checks."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from autogfm.experiments import RealPanelExperimentSpec
from autogfm.forecasting import BacktestSpec
from autogfm.graph import GraphSpec
from autogfm.pipeline import Stage3PipelineSpec, run_stage3_pipeline
from autogfm.realdata import CanonicalizationConfig, PanelSpec
from autogfm.realdata.demo import generate_stage3_demo_data
from autogfm.realdata.common import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage3_validation"))
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
        neural = RealPanelExperimentSpec(max_epochs=30, patience=6, seeds=(42, 43, 44))
    else:
        backtest = BacktestSpec(origin_step=3, bootstrap_samples=100)
        neural = RealPanelExperimentSpec(
            hidden_dim=16,
            max_epochs=8,
            patience=3,
            seeds=(42,),
        )
    spec = Stage3PipelineSpec(
        canonicalization=CanonicalizationConfig(
            source_name="stage3_demo",
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
    run = run_stage3_pipeline(
        demo.sales_csv,
        demo.mapping_csv,
        demo.vehicles_csv,
        args.output_dir / "run",
        spec,
        overwrite=True,
    )

    required = [
        "run/01_canonical/manifest.json",
        "run/02_panel/panel.npz",
        "run/03_graph/adjacency.npy",
        "run/04_backtest/leaderboard.csv",
        "run/05_neural/aggregate_metrics.csv",
        "run/06_forecast/production_forecast.csv",
        "run/06_forecast/MODEL_CARD.md",
        "run/STAGE3_RUN_REPORT.md",
        "run/manifest.json",
    ]
    missing = [path for path in required if not (args.output_dir / path).is_file()]
    quality = run.manifest["quality"]
    panel_quality = run.manifest["panel_quality"]
    graph = run.manifest["graph_diagnostics"]
    checks = {
        "required_artifacts_present": not missing,
        "malformed_row_quarantined": quality["rejected_sales_rows"] >= 1,
        "duplicate_key_audited": quality["duplicate_monthly_keys_aggregated"] >= 1,
        "panel_has_multiple_series": panel_quality["included_series"] >= 8,
        "panel_observed_ratio": panel_quality["observed_ratio"] > 0.90,
        "graph_has_no_isolates": graph["isolated_nodes"] == 0,
        "graph_is_mostly_connected": graph["largest_component_ratio"] >= 0.90,
        "backtest_has_winners": bool(run.manifest["backtest_winners"]),
        "backtest_selected_models": bool(run.manifest["backtest_selected_models"]),
        "production_forecast_built": bool(run.manifest["production_selected_models"]),
        "neural_has_winner": bool(run.manifest["neural_winner"]),
    }
    passed = all(checks.values())
    payload = {
        "status": "passed" if passed else "failed",
        "run_id": run.run_id,
        "checks": checks,
        "missing": missing,
        "backtest_selected_models": run.manifest["backtest_selected_models"],
        "backtest_winners": run.manifest["backtest_winners"],
        "backtest_graph_claim_checks": run.manifest["backtest_graph_claim_checks"],
        "neural_winner": run.manifest["neural_winner"],
        "neural_claim_gate": run.manifest["neural_claim_gate"],
    }
    write_json(args.output_dir / "validation_result.json", payload)
    lines = [
        "# Stage 3 Validation Report",
        "",
        f"- Status: **{payload['status'].upper()}**",
        f"- Run ID: `{run.run_id}`",
        "",
        "## Acceptance checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## Experimental outcomes",
            "",
            f"- Validation-selected models: `{payload['backtest_selected_models']}`",
            f"- Oracle rolling winners: `{payload['backtest_winners']}`",
            f"- Rolling graph controls: `{payload['backtest_graph_claim_checks']}`",
            f"- Neural winner: `{payload['neural_winner']}`",
            f"- Neural graph claim gate: `{payload['neural_claim_gate']}`",
            "",
            "The validation proves the pipeline and controls execute. Demo-data model rankings are not evidence about the user's real automotive dataset.",
            "",
        ]
    )
    (args.output_dir / "STAGE3_VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
