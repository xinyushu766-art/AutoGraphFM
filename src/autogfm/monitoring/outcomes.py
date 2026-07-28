"""Post-deployment forecast outcome reconciliation and champion health gates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autogfm.realdata import RealDataError
from autogfm.realdata.common import (
    read_csv_rows,
    sha256_file,
    stable_payload_id,
    write_csv_rows,
    write_json,
)


@dataclass(frozen=True)
class OutcomeSpec:
    """Thresholds for realized-forecast performance monitoring."""

    min_match_ratio_warn: float = 0.90
    min_match_ratio_block: float = 0.70
    wape_warn_percent: float = 20.0
    wape_block_percent: float = 35.0
    coverage80_warn: float = 0.70
    coverage80_block: float = 0.55
    coverage95_warn: float = 0.85
    coverage95_block: float = 0.70

    def validate(self) -> None:
        if not 0 <= self.min_match_ratio_block <= self.min_match_ratio_warn <= 1:
            raise RealDataError("Invalid match-ratio thresholds.")
        if not 0 <= self.wape_warn_percent <= self.wape_block_percent:
            raise RealDataError("Invalid WAPE thresholds.")
        if not 0 <= self.coverage80_block <= self.coverage80_warn <= 1:
            raise RealDataError("Invalid 80% coverage thresholds.")
        if not 0 <= self.coverage95_block <= self.coverage95_warn <= 1:
            raise RealDataError("Invalid 95% coverage thresholds.")


@dataclass(frozen=True)
class OutcomeResult:
    output_dir: Path
    outcome_id: str
    status: str
    action: str
    manifest: dict[str, Any]


def _severity_high(value: float, warn: float, block: float) -> str:
    if not np.isfinite(value) or value >= block:
        return "block"
    if value >= warn:
        return "warn"
    return "pass"


def _severity_low(value: float, warn: float, block: float) -> str:
    if not np.isfinite(value) or value <= block:
        return "block"
    if value <= warn:
        return "warn"
    return "pass"


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    actual = np.asarray([row["actual"] for row in rows], dtype=np.float64)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=np.float64)
    error = prediction - actual
    absolute = np.abs(error)
    denominator = max(float(np.abs(actual).sum()), 1e-9)
    return {
        "cells": int(len(rows)),
        "wape_percent": float(100.0 * absolute.sum() / denominator),
        "mae": float(absolute.mean()),
        "rmse": float(math.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "coverage_80": float(
            np.mean(
                [
                    row["lower_80"] <= row["actual"] <= row["upper_80"]
                    for row in rows
                ]
            )
        ),
        "coverage_95": float(
            np.mean(
                [
                    row["lower_95"] <= row["actual"] <= row["upper_95"]
                    for row in rows
                ]
            )
        ),
    }


def evaluate_forecast_outcomes(
    forecast_csv: str | Path,
    actual_canonical_sales_csv: str | Path,
    output_dir: str | Path,
    spec: OutcomeSpec | None = None,
    *,
    overwrite: bool = False,
) -> OutcomeResult:
    """Match realized sales to forecasts and issue retain/investigate/retrain action."""
    spec = spec or OutcomeSpec()
    spec.validate()
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    _, forecast_rows, _ = read_csv_rows(forecast_csv)
    _, actual_rows, _ = read_csv_rows(actual_canonical_sales_csv)
    actual_index: dict[tuple[str, str], float] = {}
    duplicate_actual_keys: set[tuple[str, str]] = set()
    for row in actual_rows:
        key = (row.get("series_key", ""), row.get("month", ""))
        if not key[0] or not key[1]:
            continue
        if key in actual_index:
            duplicate_actual_keys.add(key)
        try:
            actual_index[key] = float(row["sales"])
        except (KeyError, TypeError, ValueError):
            continue
    if duplicate_actual_keys:
        raise RealDataError(
            "Actual canonical sales contain duplicate series/month keys."
        )

    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for row in forecast_rows:
        key = (row.get("series_key", ""), row.get("target_month", ""))
        actual = actual_index.get(key)
        if actual is None:
            unmatched.append(
                {
                    "series_key": key[0],
                    "target_month": key[1],
                    "horizon": row.get("horizon", ""),
                    "selected_model": row.get("selected_model", ""),
                    "reason": "actual_not_available",
                }
            )
            continue
        parsed = {
            "series_index": int(row["series_index"]),
            "series_key": row["series_key"],
            "brand_name": row.get("brand_name", ""),
            "series_name": row.get("series_name", ""),
            "origin_month": row["origin_month"],
            "target_month": row["target_month"],
            "horizon": int(row["horizon"]),
            "selected_model": row["selected_model"],
            "prediction": float(row["prediction"]),
            "actual": actual,
            "error": float(row["prediction"]) - actual,
            "absolute_error": abs(float(row["prediction"]) - actual),
            "lower_80": float(row["lower_80"]),
            "upper_80": float(row["upper_80"]),
            "lower_95": float(row["lower_95"]),
            "upper_95": float(row["upper_95"]),
        }
        parsed["covered_80"] = parsed["lower_80"] <= actual <= parsed["upper_80"]
        parsed["covered_95"] = parsed["lower_95"] <= actual <= parsed["upper_95"]
        matched.append(parsed)

    total = len(forecast_rows)
    match_ratio = len(matched) / total if total else 0.0
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in matched:
        grouped.setdefault((row["horizon"], row["selected_model"]), []).append(row)

    metric_rows: list[dict[str, Any]] = []
    severity_order = {"pass": 0, "warn": 1, "block": 2}
    overall_severity = _severity_low(
        match_ratio,
        spec.min_match_ratio_warn,
        spec.min_match_ratio_block,
    )
    for (horizon, model), rows in sorted(grouped.items()):
        values = _metrics(rows)
        wape_severity = _severity_high(
            values["wape_percent"],
            spec.wape_warn_percent,
            spec.wape_block_percent,
        )
        coverage80_severity = _severity_low(
            values["coverage_80"],
            spec.coverage80_warn,
            spec.coverage80_block,
        )
        coverage95_severity = _severity_low(
            values["coverage_95"],
            spec.coverage95_warn,
            spec.coverage95_block,
        )
        severity = max(
            (wape_severity, coverage80_severity, coverage95_severity),
            key=lambda value: severity_order[value],
        )
        overall_severity = max(
            overall_severity,
            severity,
            key=lambda value: severity_order[value],
        )
        metric_rows.append(
            {
                "horizon": horizon,
                "selected_model": model,
                **values,
                "wape_severity": wape_severity,
                "coverage80_severity": coverage80_severity,
                "coverage95_severity": coverage95_severity,
                "severity": severity,
            }
        )

    if not matched:
        overall_severity = "block"
    action = {
        "pass": "retain_champion",
        "warn": "investigate_and_shadow_challengers",
        "block": "retrain_revalidate_or_rollback",
    }[overall_severity]

    matched_fields = [
        "series_index",
        "series_key",
        "brand_name",
        "series_name",
        "origin_month",
        "target_month",
        "horizon",
        "selected_model",
        "prediction",
        "actual",
        "error",
        "absolute_error",
        "lower_80",
        "upper_80",
        "lower_95",
        "upper_95",
        "covered_80",
        "covered_95",
    ]
    metric_fields = [
        "horizon",
        "selected_model",
        "cells",
        "wape_percent",
        "mae",
        "rmse",
        "bias",
        "coverage_80",
        "coverage_95",
        "wape_severity",
        "coverage80_severity",
        "coverage95_severity",
        "severity",
    ]
    write_csv_rows(destination / "outcome_rows.csv", matched_fields, matched)
    write_csv_rows(destination / "outcome_metrics.csv", metric_fields, metric_rows)
    write_csv_rows(
        destination / "unmatched_forecasts.csv",
        ["series_key", "target_month", "horizon", "selected_model", "reason"],
        unmatched,
    )

    report = {
        "status": overall_severity,
        "action": action,
        "forecast_rows": total,
        "matched_rows": len(matched),
        "unmatched_rows": len(unmatched),
        "match_ratio": match_ratio,
        "metric_groups": len(metric_rows),
    }
    write_json(destination / "outcome_report.json", report)
    identity = {
        "schema_version": "autogfm-forecast-outcomes-v1",
        "forecast_sha256": sha256_file(forecast_csv),
        "actual_sales_sha256": sha256_file(actual_canonical_sales_csv),
        "spec": asdict(spec),
        "report": report,
    }
    outcome_id = stable_payload_id(identity)

    lines = [
        "# AutoGraphFM Forecast Outcome Report",
        "",
        f"- Outcome ID: `{outcome_id}`",
        f"- Status: **{overall_severity.upper()}**",
        f"- Recommended action: **{action}**",
        f"- Forecast rows matched: `{len(matched)}/{total}` ({match_ratio:.2%})",
        "",
        "## Horizon/model metrics",
        "",
        "| Horizon | Model | Cells | WAPE | MAE | Bias | Coverage 80 | Coverage 95 | Status |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['horizon']} | {row['selected_model']} | {row['cells']} | "
            f"{row['wape_percent']:.3f}% | {row['mae']:.3f} | "
            f"{row['bias']:.3f} | {row['coverage_80']:.3f} | "
            f"{row['coverage_95']:.3f} | {row['severity'].upper()} |"
        )
    if not metric_rows:
        lines.append("| - | - | 0 | - | - | - | - | - | BLOCK |")
    lines.extend(
        [
            "",
            "Outcome monitoring is authoritative only after the corresponding target months have closed and canonical actual sales are available.",
            "",
        ]
    )
    report_path = destination / "OUTCOME_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        **identity,
        "outcome_id": outcome_id,
        "status": overall_severity,
        "action": action,
        "files": {
            name: {"path": name, "sha256": sha256_file(destination / name)}
            for name in (
                "outcome_rows.csv",
                "outcome_metrics.csv",
                "unmatched_forecasts.csv",
                "outcome_report.json",
                "OUTCOME_REPORT.md",
            )
        },
    }
    write_json(destination / "manifest.json", manifest)
    return OutcomeResult(
        destination,
        outcome_id,
        overall_severity,
        action,
        manifest,
    )
