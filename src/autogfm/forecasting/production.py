"""Validation-selected production forecasts with empirical uncertainty bands."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from autogfm.forecasting.baselines import (
    PooledResidualRidge,
    damped_trend_forecast,
    drift_forecast,
    last_value_forecast,
    moving_average_forecast,
    regime_guarded_forecast,
    seasonal_growth_forecast,
    seasonal_naive_forecast,
)
from autogfm.realdata import RealDataError, load_panel
from autogfm.realdata.common import read_csv_rows, sha256_file, stable_payload_id, write_csv_rows, write_json


@dataclass(frozen=True)
class ProductionForecastSpec:
    history: int = 12
    ridge_alpha: float = 10.0
    interval_levels: tuple[float, ...] = (0.80, 0.95)

    def validate(self) -> None:
        if self.history < 2:
            raise RealDataError("history must be at least 2.")
        if self.ridge_alpha < 0:
            raise RealDataError("ridge_alpha cannot be negative.")
        if not self.interval_levels:
            raise RealDataError("at least one interval level is required.")
        if any(not 0.0 < value < 1.0 for value in self.interval_levels):
            raise RealDataError("interval levels must be in (0, 1).")


@dataclass(frozen=True)
class ProductionForecastResult:
    output_dir: Path
    forecast_id: str
    manifest: dict[str, Any]


def _add_months(value: date, count: int) -> date:
    ordinal = value.year * 12 + value.month - 1 + count
    return date(ordinal // 12, ordinal % 12 + 1, 1)


def _ridge_adjacency(model_name: str, graph_dir: Path) -> np.ndarray | None:
    mapping = {
        "pooled_ridge": None,
        "seasonal_residual_ridge": None,
        "graph_ridge": np.load(graph_dir / "adjacency.npy"),
        "identity_graph_ridge": np.load(graph_dir / "identity_adjacency.npy"),
        "permuted_graph_ridge": np.load(graph_dir / "permuted_adjacency.npy"),
    }
    if model_name not in mapping:
        raise RealDataError(f"Unsupported ridge model: {model_name}")
    return mapping[model_name]


def _forecast_model(
    model_name: str,
    horizon: int,
    sales: np.ndarray,
    static_features: np.ndarray,
    month_numbers: np.ndarray,
    graph_dir: Path,
    spec: ProductionForecastSpec,
) -> np.ndarray:
    origin = sales.shape[0] - 1
    horizons = (horizon,)
    if model_name == "last_value":
        return last_value_forecast(sales, origin, horizons)[0]
    if model_name == "seasonal_naive_12":
        return seasonal_naive_forecast(sales, origin, horizons)[0]
    if model_name == "moving_average_3":
        return moving_average_forecast(sales, origin, horizons, 3)[0]
    if model_name == "moving_average_6":
        return moving_average_forecast(sales, origin, horizons, 6)[0]
    if model_name == "drift":
        return drift_forecast(sales, origin, horizons, spec.history)[0]
    if model_name == "damped_trend":
        return damped_trend_forecast(
            sales, origin, horizons, spec.history
        )[0]
    if model_name == "seasonal_growth":
        return seasonal_growth_forecast(sales, origin, horizons)[0]
    if model_name == "regime_guarded":
        return regime_guarded_forecast(
            sales, origin, horizons, max(24, spec.history)
        )[0]
    if model_name.endswith("ridge"):
        model = PooledResidualRidge(
            alpha=spec.ridge_alpha,
            adjacency=_ridge_adjacency(model_name, graph_dir),
            month_numbers=month_numbers,
            anchor=(
                "seasonal_naive_12"
                if model_name == "seasonal_residual_ridge"
                else "last_value"
            ),
        )
        model.fit(
            sales,
            static_features,
            origin,
            history=spec.history,
            horizons=horizons,
        )
        return model.predict(sales, static_features, origin, horizons)[0]
    raise RealDataError(f"Unsupported selected production model: {model_name}")


def build_production_forecast(
    panel_dir: str | Path,
    graph_dir: str | Path,
    backtest_dir: str | Path,
    output_dir: str | Path,
    spec: ProductionForecastSpec | None = None,
    *,
    overwrite: bool = False,
) -> ProductionForecastResult:
    """Refit validation-selected models through the latest month and forecast."""
    spec = spec or ProductionForecastSpec()
    spec.validate()
    panel = load_panel(panel_dir)
    graph_path = Path(graph_dir)
    backtest_path = Path(backtest_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    backtest_manifest = json.loads(
        (backtest_path / "manifest.json").read_text(encoding="utf-8")
    )
    selected_models = {
        int(horizon): model for horizon, model in backtest_manifest["selected_models"].items()
    }
    _, prediction_rows, _ = read_csv_rows(backtest_path / "predictions.csv")
    last_month = date.fromisoformat(panel.months[-1])
    max_horizon = max(selected_models)
    future_months = [_add_months(last_month, value) for value in range(1, max_horizon + 1)]
    extended_month_numbers = np.array(
        [int(value[5:7]) for value in panel.months]
        + [value.month for value in future_months],
        dtype=np.int64,
    )

    forecast_rows: list[dict[str, Any]] = []
    calibration: dict[str, Any] = {}
    for horizon, model_name in sorted(selected_models.items()):
        point = _forecast_model(
            model_name,
            horizon,
            panel.sales,
            panel.static_features,
            extended_month_numbers,
            graph_path,
            spec,
        )
        validation_errors = np.array(
            [
                abs(float(row["target"]) - float(row["prediction"]))
                for row in prediction_rows
                if row["split"] == "validation"
                and int(row["horizon"]) == horizon
                and row["model_name"] == model_name
            ],
            dtype=np.float64,
        )
        if validation_errors.size == 0:
            raise RealDataError(
                f"No validation residuals for selected model {model_name}, horizon {horizon}."
            )
        radii = {
            level: float(np.quantile(validation_errors, level, method="higher"))
            for level in spec.interval_levels
        }
        calibration[str(horizon)] = {
            "model_name": model_name,
            "validation_cells": int(validation_errors.size),
            "absolute_error_quantiles": {
                str(level): radius for level, radius in radii.items()
            },
        }
        target_month = _add_months(last_month, horizon).isoformat()
        for series_index, prediction in enumerate(point):
            row = {
                "series_index": series_index,
                "series_key": panel.series_keys[series_index],
                "brand_name": panel.catalog[series_index].get("brand_name", ""),
                "series_name": panel.catalog[series_index].get("series_name", ""),
                "origin_month": last_month.isoformat(),
                "target_month": target_month,
                "horizon": horizon,
                "selected_model": model_name,
                "prediction": float(prediction),
                "calibration_cells": int(validation_errors.size),
            }
            for level, radius in radii.items():
                label = int(round(level * 100))
                row[f"lower_{label}"] = max(0.0, float(prediction) - radius)
                row[f"upper_{label}"] = float(prediction) + radius
            forecast_rows.append(row)

    fieldnames = list(forecast_rows[0].keys())
    write_csv_rows(destination / "production_forecast.csv", fieldnames, forecast_rows)
    write_json(destination / "calibration.json", calibration)

    identity = {
        "schema_version": "autogfm-production-forecast-v1",
        "panel_id": panel.manifest["panel_id"],
        "backtest_id": backtest_manifest["backtest_id"],
        "selected_models": selected_models,
        "spec": asdict(spec),
        "forecast_sha256": sha256_file(destination / "production_forecast.csv"),
    }
    forecast_id = stable_payload_id(identity)
    manifest = {
        **identity,
        "forecast_id": forecast_id,
        "origin_month": last_month.isoformat(),
        "forecast_rows": len(forecast_rows),
        "calibration": calibration,
        "files": {
            name: {"path": name, "sha256": sha256_file(destination / name)}
            for name in ("production_forecast.csv", "calibration.json")
        },
    }
    write_json(destination / "manifest.json", manifest)

    lines = [
        "# Production Forecast Model Card",
        "",
        f"- Forecast ID: `{forecast_id}`",
        f"- Origin month: `{last_month.isoformat()}`",
        f"- Selected models: `{selected_models}`",
        "",
        "## Selection protocol",
        "",
        "Models were selected independently for each horizon using validation WAPE. The selected model was then refit through the latest available month.",
        "",
        "## Uncertainty protocol",
        "",
        "Intervals are empirical symmetric bands calibrated from absolute validation residuals. They are diagnostic uncertainty bands, not guaranteed probabilistic coverage.",
        "",
        "## Operational caveats",
        "",
        "Do not treat forecasts for newly launched, discontinued, heavily imputed, or structurally shocked series as equivalent to stable-series forecasts. Review the panel exclusions and per-series backtest metrics before operational use.",
        "",
    ]
    (destination / "MODEL_CARD.md").write_text("\n".join(lines), encoding="utf-8")
    return ProductionForecastResult(destination, forecast_id, manifest)
