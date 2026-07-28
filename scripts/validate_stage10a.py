"""Validate Stage 10A leakage controls and real multi-window benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

import autogfm
from autogfm.pipeline.stage10a import run_stage10a_real_market
from autogfm.realdata.lifecycle import profile_series_lifecycle


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    return tuple(int(part) for part in parts[:3])  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage10a_validation"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)

    source = Path("data/china_auto_market/ev_sales_monthly.csv")
    result = run_stage10a_real_market(
        source,
        args.output_dir,
        overwrite=args.overwrite,
    )
    manifest = result.manifest

    months = tuple(f"2020-{month:02d}-01" for month in range(1, 13)) + tuple(
        f"2021-{month:02d}-01" for month in range(1, 13)
    )
    future_launch = np.full(24, np.nan)
    future_launch[15:] = 50.0
    lifecycle = profile_series_lifecycle(
        future_launch,
        months,
        train_end_index=11,
        test_end_index=23,
    )

    checks = {
        "package_version_at_least_0_14_0": (
            _version_tuple(autogfm.__version__) >= (0, 14, 0)
        ),
        "source_rows_76": (
            manifest["ev_multiwindow"]["source_rows"] == 76
            and manifest["total_multiwindow"]["source_rows"] == 76
        ),
        "three_annual_folds": (
            len(manifest["ev_multiwindow"]["folds"]) == 3
            and len(manifest["total_multiwindow"]["folds"]) == 3
        ),
        "horizons_1_3_6": {
            row["horizon"]
            for row in manifest["ev_multiwindow"]["aggregate_selected_metrics"]
        }
        == {1, 3, 6},
        "aggregate_metrics_finite": all(
            math.isfinite(float(row[metric]))
            for target in ("ev_multiwindow", "total_multiwindow")
            for row in manifest[target]["aggregate_selected_metrics"]
            for metric in ("wape_percent", "mae", "rmse", "bias")
        ),
        "selection_instability_exposed": (
            manifest["model_selection_stability"] == "warn"
        ),
        "graph_claim_blocked": manifest["graph_research_claim"] == "no-claim",
        "future_launch_not_train_observed": (
            lifecycle["lifecycle_at_train_end"] == "not_observed_by_train"
            and lifecycle["outcome_status_at_test_end"] == "launched_after_train"
        ),
        "reports_exist": (
            (args.output_dir / "STAGE10A_REPORT.md").is_file()
            and (args.output_dir / "STAGE10A_REPORT.html").is_file()
            and (
                args.output_dir
                / "03_ev_multiwindow"
                / "MULTIWINDOW_MARKET_REPORT.html"
            ).is_file()
        ),
        "engineering_pass": manifest["engineering_status"] == "pass",
    }
    passed = sum(checks.values())
    validation = {
        "package_version": autogfm.__version__,
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "status": "passed" if passed == len(checks) else "failed",
        "stage10a_run_id": result.run_id,
    }
    (args.output_dir / "STAGE10A_VALIDATION.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# AutoGraphFM Stage 10A Compatibility Validation",
        "",
        f"- Status: **{validation['status'].upper()}**",
        f"- Checks: **{passed}/{len(checks)} passed**",
        f"- Run ID: `{result.run_id}`",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {'PASS' if value else 'FAIL'} |"
        for name, value in checks.items()
    )
    lines.extend(
        [
            "",
            "Engineering acceptance does not authorize a vehicle-graph claim. ",
            "The executed real evidence remains aggregate-market only.",
            "",
        ]
    )
    (args.output_dir / "STAGE10A_VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False))
    if validation["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
