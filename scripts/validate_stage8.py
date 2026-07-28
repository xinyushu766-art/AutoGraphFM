"""Validate SRNI authenticity gates without claiming real-data performance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import date
from pathlib import Path

from autogfm.forecasting import BacktestSpec
from autogfm.governance import ReleaseSpec, prepare_release_candidate
from autogfm.graph import GraphSpec
from autogfm.monitoring import MonitoringSpec
from autogfm.pipeline import Stage3PipelineSpec, Stage4PipelineSpec, run_stage4_pipeline
from autogfm.realdata import (
    CanonicalizationConfig,
    PanelSpec,
    RealDataError,
    SrniCarPreparationConfig,
    inspect_srni_car_sales,
    prepare_srni_car_sales,
)
from autogfm.realdata.common import write_json
from autogfm.reporting import Stage4ReportSpec


def _generate_official_schema_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Car series",
        "Brand",
        "Year",
        "Month",
        "Car model type",
        "Brand energy type",
        "Size",
        "Brand country of origin",
        "Model launch date",
        "Brand establishment date",
        "Brand entered China date",
        "Sales",
    ]
    rows: list[list[object]] = []
    for series_index in range(18):
        brand_index = series_index // 3 + 1
        for month_index in range(84):
            year = 2016 + month_index // 12
            month = month_index % 12 + 1
            trend = 12 * month_index
            seasonality = 90 * math.sin(2 * math.pi * month / 12)
            level = 450 + 110 * series_index
            sales = max(1, round(level + trend + seasonality))
            rows.append(
                [
                    f"Series{series_index + 1:02d}",
                    f"Brand{brand_index}",
                    year,
                    month,
                    "SUV" if series_index % 2 == 0 else "Sedan",
                    "NEV" if series_index % 3 == 0 else "ICE",
                    "Compact" if series_index % 2 == 0 else "Mid-size",
                    "China",
                    2010 + series_index % 8,
                    1998 + brand_index,
                    2002 + brand_index,
                    sales,
                ]
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage8_validation"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Validation output exists: {args.output_dir}. Use --overwrite."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    fixture = args.output_dir / "fixture" / "Sales.csv"
    _generate_official_schema_fixture(fixture)
    inspection = inspect_srni_car_sales(
        fixture,
        args.output_dir / "inspection",
        overwrite=True,
    )

    strict_rejected = False
    try:
        prepare_srni_car_sales(
            fixture,
            args.output_dir / "strict_prepare",
            config=SrniCarPreparationConfig(
                required_evidence_scope="official_real_data"
            ),
            overwrite=True,
        )
    except RealDataError:
        strict_rejected = True

    prepared = prepare_srni_car_sales(
        fixture,
        args.output_dir / "prepared",
        overwrite=True,
    )
    stage3 = Stage3PipelineSpec(
        canonicalization=CanonicalizationConfig(
            source_name="srni_car",
            data_version="stage8-synthetic-validation",
            strict_unmatched=False,
            duplicate_policy="sum",
        ),
        panel=PanelSpec(
            train_end=date(2020, 12, 1),
            validation_end=date(2021, 12, 1),
            test_end=date(2022, 12, 1),
            min_history_months=36,
            max_missing_ratio=0.20,
            max_trailing_missing_months=3,
            imputation="linear",
        ),
        graph=GraphSpec(top_k=5),
        backtest=BacktestSpec(
            horizons=(1, 3, 6),
            origin_step=3,
            bootstrap_samples=20,
        ),
        run_neural=False,
    )
    run = run_stage4_pipeline(
        prepared.sales_csv,
        prepared.mapping_csv,
        prepared.vehicles_csv,
        args.output_dir / "run",
        Stage4PipelineSpec(
            stage3=stage3,
            monitoring=MonitoringSpec(recent_months=6),
            report=Stage4ReportSpec(evidence_scope=inspection.evidence_scope),
        ),
        overwrite=True,
    )
    production = prepare_release_candidate(
        run.output_dir,
        args.output_dir / "production_release",
        ReleaseSpec(mode="production"),
        overwrite=True,
    )
    audit_html = (run.output_dir / "08_report" / "AUDIT_REPORT.html").read_text(
        encoding="utf-8"
    )
    checks = {
        "synthetic_fixture_detected": inspection.evidence_scope == "synthetic_only",
        "strict_official_preparation_rejected": strict_rejected,
        "engineering_mode_completed": run.manifest["engineering_status"] == "pass",
        "stage4_scope_remains_synthetic": (
            run.manifest["evidence_scope"] == "synthetic_only"
        ),
        "synthetic_run_not_release_ready": run.manifest["release_ready"] is False,
        "production_release_blocked": (
            production.status == "blocked" and production.deployable is False
        ),
        "audit_does_not_claim_user_data": (
            "SYNTHETIC_ONLY" in audit_html and "USER_DATA" not in audit_html
        ),
    }
    passed = all(checks.values())
    payload = {
        "status": "passed" if passed else "failed",
        "package_stage": "8.0",
        "fixture_scope": inspection.evidence_scope,
        "run_id": run.run_id,
        "checks": checks,
        "claim_boundary": (
            "This validation proves authenticity and release gates only. "
            "It is not a real SRNI-CAR accuracy result."
        ),
    }
    write_json(args.output_dir / "validation_result.json", payload)
    lines = [
        "# AutoGraphFM Stage 8 Validation Report",
        "",
        f"- Status: **{payload['status'].upper()}**",
        f"- Fixture evidence scope: **{inspection.evidence_scope.upper()}**",
        f"- Stage 4 run ID: `{run.run_id}`",
        "",
        "## Acceptance checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "The fixture follows the official schema but is synthetic. Its model "
            "rankings are not real SRNI-CAR evidence and must not be cited.",
            "",
        ]
    )
    (args.output_dir / "STAGE8_VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
