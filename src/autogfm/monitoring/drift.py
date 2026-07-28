"""Production-oriented data drift and forecast sanity monitoring."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autogfm.realdata import RealDataError, load_panel
from autogfm.realdata.common import read_csv_rows, sha256_file, stable_payload_id, write_csv_rows, write_json


@dataclass(frozen=True)
class MonitoringSpec:
    """Thresholds for operational data and forecast gates."""

    recent_months: int = 6
    psi_bins: int = 10
    psi_warn: float = 0.10
    psi_block: float = 0.25
    recent_missing_warn: float = 0.10
    recent_missing_block: float = 0.25
    series_shift_warn_z: float = 2.5
    series_shift_block_z: float = 4.0
    forecast_ratio_min: float = 0.20
    forecast_ratio_max: float = 5.0
    max_blocking_series_ratio: float = 0.10

    def validate(self) -> None:
        if self.recent_months < 1:
            raise RealDataError("recent_months must be positive.")
        if self.psi_bins < 3:
            raise RealDataError("psi_bins must be at least 3.")
        if not 0 <= self.psi_warn <= self.psi_block:
            raise RealDataError("Expected 0 <= psi_warn <= psi_block.")
        if not 0 <= self.recent_missing_warn <= self.recent_missing_block < 1:
            raise RealDataError("Invalid recent missingness thresholds.")
        if not 0 < self.series_shift_warn_z <= self.series_shift_block_z:
            raise RealDataError("Invalid series shift thresholds.")
        if not 0 <= self.forecast_ratio_min < self.forecast_ratio_max:
            raise RealDataError("Invalid forecast ratio bounds.")
        if not 0 <= self.max_blocking_series_ratio <= 1:
            raise RealDataError("max_blocking_series_ratio must be in [0, 1].")


@dataclass(frozen=True)
class MonitoringResult:
    output_dir: Path
    monitoring_id: str
    status: str
    manifest: dict[str, Any]


def _psi(reference: np.ndarray, current: np.ndarray, bins: int) -> float:
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        return float("nan")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    expected, _ = np.histogram(reference, bins=edges)
    actual, _ = np.histogram(current, bins=edges)
    expected_rate = np.clip(expected / max(expected.sum(), 1), 1e-6, None)
    actual_rate = np.clip(actual / max(actual.sum(), 1), 1e-6, None)
    return float(np.sum((actual_rate - expected_rate) * np.log(actual_rate / expected_rate)))


def _severity(value: float, warn: float, block: float) -> str:
    if not np.isfinite(value):
        return "block"
    if value >= block:
        return "block"
    if value >= warn:
        return "warn"
    return "pass"


def run_monitoring(
    panel_dir: str | Path,
    backtest_dir: str | Path,
    production_dir: str | Path,
    output_dir: str | Path,
    spec: MonitoringSpec | None = None,
    *,
    neural_dir: str | Path | None = None,
    overwrite: bool = False,
) -> MonitoringResult:
    """Evaluate drift, coverage, model controls, and forecast sanity."""
    spec = spec or MonitoringSpec()
    spec.validate()
    panel = load_panel(panel_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    backtest_manifest = json.loads(
        (Path(backtest_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    production_manifest = json.loads(
        (Path(production_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    neural_manifest = None
    if neural_dir is not None and (Path(neural_dir) / "manifest.json").is_file():
        neural_manifest = json.loads(
            (Path(neural_dir) / "manifest.json").read_text(encoding="utf-8")
        )

    train_end = int(panel.manifest["boundaries"]["train_end_index"])
    recent_start = max(0, panel.sales.shape[0] - spec.recent_months)
    train_log = np.log1p(np.clip(panel.sales[: train_end + 1], 0.0, None))
    recent_log = np.log1p(np.clip(panel.sales[recent_start:], 0.0, None))
    global_psi = _psi(train_log.ravel(), recent_log.ravel(), spec.psi_bins)
    recent_missing_ratio = float(1.0 - panel.observed_mask[recent_start:].mean())

    series_rows: list[dict[str, Any]] = []
    blocking_series = 0
    warning_series = 0
    for index, key in enumerate(panel.series_keys):
        train_values = train_log[:, index]
        recent_values = recent_log[:, index]
        train_mean = float(train_values.mean())
        train_std = float(train_values.std())
        if train_std < 1e-6:
            train_std = 1.0
        recent_mean = float(recent_values.mean())
        shift_z = (recent_mean - train_mean) / train_std
        missing_ratio = float(1.0 - panel.observed_mask[recent_start:, index].mean())
        shift_severity = _severity(
            abs(shift_z), spec.series_shift_warn_z, spec.series_shift_block_z
        )
        missing_severity = _severity(
            missing_ratio, spec.recent_missing_warn, spec.recent_missing_block
        )
        severity_order = {"pass": 0, "warn": 1, "block": 2}
        severity = max(
            (shift_severity, missing_severity), key=lambda value: severity_order[value]
        )
        if severity == "block":
            blocking_series += 1
        elif severity == "warn":
            warning_series += 1
        series_rows.append(
            {
                "series_index": index,
                "series_key": key,
                "brand_name": panel.catalog[index].get("brand_name", ""),
                "series_name": panel.catalog[index].get("series_name", ""),
                "train_log_mean": train_mean,
                "recent_log_mean": recent_mean,
                "level_shift_z": shift_z,
                "recent_missing_ratio": missing_ratio,
                "severity": severity,
            }
        )

    _, forecast_rows, _ = read_csv_rows(
        Path(production_dir) / "production_forecast.csv"
    )
    forecast_alerts: list[dict[str, Any]] = []
    for row in forecast_rows:
        series_index = int(row["series_index"])
        last_sales = float(panel.sales[-1, series_index])
        prediction = float(row["prediction"])
        ratio = prediction / max(last_sales, 1.0)
        reasons = []
        if prediction < 0 or not np.isfinite(prediction):
            reasons.append("invalid_prediction")
        if ratio < spec.forecast_ratio_min:
            reasons.append("forecast_ratio_too_low")
        if ratio > spec.forecast_ratio_max:
            reasons.append("forecast_ratio_too_high")
        for level in (80, 95):
            lower = float(row[f"lower_{level}"])
            upper = float(row[f"upper_{level}"])
            if not (0.0 <= lower <= prediction <= upper):
                reasons.append(f"invalid_interval_{level}")
        if reasons:
            forecast_alerts.append(
                {
                    "series_index": series_index,
                    "series_key": row["series_key"],
                    "horizon": int(row["horizon"]),
                    "selected_model": row["selected_model"],
                    "last_sales": last_sales,
                    "prediction": prediction,
                    "forecast_to_last_ratio": ratio,
                    "reasons": "|".join(reasons),
                }
            )

    global_checks = {
        "panel_latest_month_matches_test_end": (
            panel.months[-1] == panel.manifest["boundaries"]["test_end"]
        ),
        "global_psi": {
            "value": global_psi,
            "severity": _severity(global_psi, spec.psi_warn, spec.psi_block),
        },
        "recent_missing_ratio": {
            "value": recent_missing_ratio,
            "severity": _severity(
                recent_missing_ratio,
                spec.recent_missing_warn,
                spec.recent_missing_block,
            ),
        },
        "blocking_series_ratio": blocking_series / len(panel.series_keys),
        "forecast_alerts": len(forecast_alerts),
        "backtest_selected_models": backtest_manifest["selected_models"],
        "backtest_graph_claim_checks": backtest_manifest["graph_claim_checks"],
        "neural_claim_gate": (
            neural_manifest["claim_gate"] if neural_manifest is not None else None
        ),
    }

    block_reasons: list[str] = []
    warn_reasons: list[str] = []
    if not global_checks["panel_latest_month_matches_test_end"]:
        block_reasons.append("panel_not_current_to_declared_test_end")
    if global_checks["global_psi"]["severity"] == "block":
        block_reasons.append("global_sales_distribution_drift")
    elif global_checks["global_psi"]["severity"] == "warn":
        warn_reasons.append("moderate_global_sales_distribution_drift")
    if global_checks["recent_missing_ratio"]["severity"] == "block":
        block_reasons.append("excessive_recent_missingness")
    elif global_checks["recent_missing_ratio"]["severity"] == "warn":
        warn_reasons.append("elevated_recent_missingness")
    if global_checks["blocking_series_ratio"] > spec.max_blocking_series_ratio:
        block_reasons.append("too_many_blocking_series")
    elif warning_series > 0 or blocking_series > 0:
        warn_reasons.append("series_level_shift_or_missingness")
    if forecast_alerts:
        block_reasons.append("forecast_sanity_alerts")

    status = "block" if block_reasons else "warn" if warn_reasons else "pass"
    write_csv_rows(
        destination / "series_monitoring.csv",
        list(series_rows[0].keys()),
        sorted(series_rows, key=lambda row: ({"block": 0, "warn": 1, "pass": 2}[row["severity"]], -abs(row["level_shift_z"]))),
    )
    write_csv_rows(
        destination / "forecast_alerts.csv",
        [
            "series_index", "series_key", "horizon", "selected_model", "last_sales",
            "prediction", "forecast_to_last_ratio", "reasons",
        ],
        forecast_alerts,
    )
    report_payload = {
        "status": status,
        "block_reasons": block_reasons,
        "warn_reasons": warn_reasons,
        "global_checks": global_checks,
        "series_summary": {
            "total": len(panel.series_keys),
            "pass": len(panel.series_keys) - warning_series - blocking_series,
            "warn": warning_series,
            "block": blocking_series,
        },
    }
    write_json(destination / "monitoring_report.json", report_payload)

    lines = [
        "# AutoGraphFM Monitoring Report",
        "",
        f"- Status: **{status.upper()}**",
        f"- Global PSI: `{global_psi:.4f}`",
        f"- Recent missing ratio: `{recent_missing_ratio:.4f}`",
        f"- Series pass/warn/block: `{report_payload['series_summary']}`",
        f"- Forecast alerts: `{len(forecast_alerts)}`",
        "",
        "## Block reasons",
        "",
    ]
    lines.extend([f"- {reason}" for reason in block_reasons] or ["- None"])
    lines.extend(["", "## Warning reasons", ""])
    lines.extend([f"- {reason}" for reason in warn_reasons] or ["- None"])
    lines.extend(
        [
            "",
            "## Research-control status",
            "",
            f"- Rolling graph checks: `{backtest_manifest['graph_claim_checks']}`",
            f"- Neural graph gate: `{neural_manifest['claim_gate'] if neural_manifest else 'not_run'}`",
            "",
            "A PASS/WARN monitoring status does not authorize a graph-effect research claim; that decision remains controlled by the held-out graph comparisons.",
            "",
        ]
    )
    markdown_path = destination / "MONITORING_REPORT.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    identity = {
        "schema_version": "autogfm-monitoring-v1",
        "panel_id": panel.manifest["panel_id"],
        "backtest_id": backtest_manifest["backtest_id"],
        "forecast_id": production_manifest["forecast_id"],
        "spec": asdict(spec),
        "report_sha256": sha256_file(destination / "monitoring_report.json"),
    }
    monitoring_id = stable_payload_id(identity)
    manifest = {
        **identity,
        "monitoring_id": monitoring_id,
        **report_payload,
        "files": {
            name: {"path": name, "sha256": sha256_file(destination / name)}
            for name in (
                "series_monitoring.csv",
                "forecast_alerts.csv",
                "monitoring_report.json",
                "MONITORING_REPORT.md",
            )
        },
    }
    write_json(destination / "manifest.json", manifest)
    return MonitoringResult(destination, monitoring_id, status, manifest)
