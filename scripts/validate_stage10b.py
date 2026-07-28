"""Validate Stage 10B literature models and adaptive graph scaffolds."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import autogfm
from autogfm.pipeline.stage10b import run_stage10b_literature_market


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    return tuple(int(part) for part in parts[:3])  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage10b_validation"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    result = run_stage10b_literature_market(
        Path("data/china_auto_market/ev_sales_monthly.csv"),
        args.output_dir,
        overwrite=args.overwrite,
    )
    manifest = result.manifest
    graph = manifest["adaptive_graph_audit"]
    checks = {
        "package_version_at_least_0_15_0": (
            _version_tuple(autogfm.__version__) >= (0, 15, 0)
        ),
        "engineering_pass": manifest["engineering_status"] == "pass",
        "six_comparison_cells": len(manifest["comparison"]) == 6,
        "comparison_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in manifest["comparison"]
            for key in (
                "stage10a_wape_percent",
                "stage10b_wape_percent",
                "relative_improvement_percent",
            )
        ),
        "literature_models_registered": {
            "nlinear_ridge",
            "dlinear_ridge",
            "regime_error_moe",
        }.issubset(set(manifest["stage10b_models"])),
        "regime_diagnostics_present": (
            len(manifest["top_regime_changes"]) == 12
            and (args.output_dir / "regime_diagnostics.csv").is_file()
        ),
        "adaptive_graph_rows_normalized": (
            graph["agcrn_row_sum_max_error"] < 1e-5
            and graph["mtgnn_row_sum_max_error"] < 1e-5
            and graph["blended_row_sum_max_error"] < 1e-5
        ),
        "mtgnn_topk_respected": graph["mtgnn_max_nonzero_per_row"] <= 4,
        "graph_claim_blocked": manifest["graph_research_claim"] == "no-claim",
        "reports_exist": (
            (args.output_dir / "STAGE10B_REPORT.md").is_file()
            and (args.output_dir / "STAGE10B_REPORT.html").is_file()
            and (args.output_dir / "stage10a_vs_stage10b.csv").is_file()
        ),
    }
    passed = sum(checks.values())
    validation = {
        "package_version": autogfm.__version__,
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "status": "passed" if passed == len(checks) else "failed",
        "run_id": result.run_id,
    }
    (args.output_dir / "STAGE10B_VALIDATION.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# AutoGraphFM Stage 10B Compatibility Validation",
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
    (args.output_dir / "STAGE10B_VALIDATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False))
    if validation["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
