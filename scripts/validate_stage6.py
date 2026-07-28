"""Validate governed release and post-deployment outcome feedback loops."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import date
from pathlib import Path

from autogfm.experiments import RealPanelExperimentSpec
from autogfm.forecasting import BacktestSpec
from autogfm.governance import ReleaseSpec, prepare_release_candidate
from autogfm.graph import GraphSpec
from autogfm.monitoring import evaluate_forecast_outcomes
from autogfm.pipeline import (
    Stage3PipelineSpec,
    Stage4PipelineSpec,
    Stage5PipelineSpec,
    run_stage5_pipeline,
)
from autogfm.realdata import CanonicalizationConfig, PanelSpec
from autogfm.realdata.common import read_csv_rows, write_csv_rows, write_json
from autogfm.realdata.demo import generate_stage3_demo_data


def _verify_checksums(bundle_dir: Path) -> bool:
    checksum_path = bundle_dir / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        return False
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        digest = hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest()
        if digest != expected:
            return False
    return True


def _build_actuals(
    forecast_csv: Path,
    output_csv: Path,
    *,
    stress: bool,
) -> None:
    _, forecasts, _ = read_csv_rows(forecast_csv)
    fields = [
        "series_key",
        "brand_name",
        "series_name",
        "month",
        "sales",
        "sales_rank",
        "source",
        "data_version",
    ]
    rows = []
    for index, row in enumerate(forecasts):
        prediction = float(row["prediction"])
        if stress:
            actual = max(1.0, prediction * (0.18 + 0.02 * (index % 3)))
        else:
            actual = max(0.0, prediction * (1.0 + 0.015 * math.sin(index)))
        rows.append(
            {
                "series_key": row["series_key"],
                "brand_name": row["brand_name"],
                "series_name": row["series_name"],
                "month": row["target_month"],
                "sales": round(actual, 3),
                "sales_rank": index + 1,
                "source": "stage6_simulated_actuals",
                "data_version": "healthy" if not stress else "stress",
            }
        )
    write_csv_rows(output_csv, fields, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage6_validation"),
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
            source_name="stage6_demo",
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
    run = run_stage5_pipeline(
        demo.sales_csv,
        demo.mapping_csv,
        demo.vehicles_csv,
        args.output_dir / "run",
        Stage5PipelineSpec(
            stage4=Stage4PipelineSpec(stage3=stage3),
            release=ReleaseSpec(mode="shadow"),
        ),
        overwrite=True,
    )

    production_check = prepare_release_candidate(
        run.output_dir,
        args.output_dir / "production_release_check",
        ReleaseSpec(mode="production"),
        overwrite=True,
    )
    forecast_csv = run.output_dir / "06_forecast/production_forecast.csv"
    healthy_actuals = args.output_dir / "healthy_actuals.csv"
    stress_actuals = args.output_dir / "stress_actuals.csv"
    _build_actuals(forecast_csv, healthy_actuals, stress=False)
    _build_actuals(forecast_csv, stress_actuals, stress=True)
    healthy = evaluate_forecast_outcomes(
        forecast_csv,
        healthy_actuals,
        args.output_dir / "healthy_outcomes",
        overwrite=True,
    )
    stress = evaluate_forecast_outcomes(
        forecast_csv,
        stress_actuals,
        args.output_dir / "stress_outcomes",
        overwrite=True,
    )

    shadow_bundle = run.output_dir / "09_release/shadow_bundle"
    required = [
        run.output_dir / "stage5_manifest.json",
        run.output_dir / "09_release/RELEASE_READINESS.md",
        shadow_bundle / "SHA256SUMS.txt",
        args.output_dir / "production_release_check/release_decision.json",
        args.output_dir / "healthy_outcomes/OUTCOME_REPORT.md",
        args.output_dir / "stress_outcomes/OUTCOME_REPORT.md",
    ]
    checks = {
        "required_artifacts_present": all(path.is_file() for path in required),
        "shadow_release_is_not_deployable": (
            run.manifest["release_status"] == "shadow"
            and run.manifest["deployable"] is False
        ),
        "shadow_checksums_verify": _verify_checksums(shadow_bundle),
        "production_demo_release_is_blocked": (
            production_check.status == "blocked"
            and production_check.deployable is False
            and not (production_check.output_dir / "deployment_bundle").exists()
        ),
        "healthy_outcomes_retain_champion": (
            healthy.status == "pass" and healthy.action == "retain_champion"
        ),
        "stress_outcomes_trigger_rollback": (
            stress.status == "block"
            and stress.action == "retrain_revalidate_or_rollback"
        ),
        "stage4_gate_remains_authoritative": (
            run.stage4.manifest["deployment_status"] == "block"
            and run.stage4.manifest["evidence_scope"] == "demo_only"
        ),
    }
    passed = all(checks.values())
    payload = {
        "status": "passed" if passed else "failed",
        "stage5_run_id": run.run_id,
        "checks": checks,
        "stage4_deployment_status": run.stage4.manifest["deployment_status"],
        "shadow_release_status": run.manifest["release_status"],
        "production_release_status": production_check.status,
        "healthy_outcome_status": healthy.status,
        "healthy_action": healthy.action,
        "stress_outcome_status": stress.status,
        "stress_action": stress.action,
    }
    write_json(args.output_dir / "validation_result.json", payload)
    lines = [
        "# Stage 6 Validation Report",
        "",
        f"- Status: **{payload['status'].upper()}**",
        f"- Stage 5 run ID: `{run.run_id}`",
        f"- Shadow release: **{payload['shadow_release_status'].upper()}**",
        f"- Production release check: **{payload['production_release_status'].upper()}**",
        f"- Healthy feedback action: **{payload['healthy_action']}**",
        f"- Stress feedback action: **{payload['stress_action']}**",
        "",
        "## Acceptance checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "This validation uses simulated future actuals only to prove the feedback mechanism. It is not a real forecast-accuracy result.",
            "",
        ]
    )
    (args.output_dir / "STAGE6_VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
