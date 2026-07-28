"""Rolling-origin backtesting with controls, bootstrap intervals, and reports."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

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
from autogfm.realdata import PanelData, RealDataError, load_panel
from autogfm.realdata.common import sha256_file, stable_payload_id, write_csv_rows, write_json


@dataclass(frozen=True)
class BacktestSpec:
    history: int = 12
    horizons: tuple[int, ...] = (1, 3, 6)
    origin_step: int = 1
    ridge_alpha: float = 10.0
    bootstrap_samples: int = 500
    seed: int = 42
    score_observed_only: bool = True

    def validate(self) -> None:
        if self.history < 2:
            raise RealDataError("history must be at least 2.")
        if not self.horizons or any(value < 1 for value in self.horizons):
            raise RealDataError("horizons must contain positive values.")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise RealDataError("horizons must be sorted and unique.")
        if self.origin_step < 1:
            raise RealDataError("origin_step must be positive.")
        if self.ridge_alpha < 0:
            raise RealDataError("ridge_alpha cannot be negative.")
        if self.bootstrap_samples < 0:
            raise RealDataError("bootstrap_samples cannot be negative.")


@dataclass(frozen=True)
class BacktestResult:
    output_dir: Path
    backtest_id: str
    manifest: dict[str, Any]


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "wape_percent": float("nan"),
            "smape_percent": float("nan"),
            "bias": float("nan"),
        }
    error = prediction - target
    absolute = np.abs(error)
    denominator = max(float(np.abs(target).sum()), 1e-8)
    smape = np.mean(2.0 * absolute / (np.abs(prediction) + np.abs(target) + 1e-8))
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape_percent": float(100.0 * absolute.sum() / denominator),
        "smape_percent": float(100.0 * smape),
        "bias": float(error.mean()),
    }


def _origin_split(panel: PanelData, target_index: int) -> str:
    validation_end = int(panel.manifest["boundaries"]["validation_end_index"])
    train_end = int(panel.manifest["boundaries"]["train_end_index"])
    if target_index <= train_end:
        return "train"
    if target_index <= validation_end:
        return "validation"
    return "test"


def _bootstrap_delta(
    model_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if samples <= 0 or not model_rows or not baseline_rows:
        return float("nan"), float("nan"), float("nan")
    baseline_lookup = {
        (row["origin_index"], row["horizon"], row["series_index"]): row
        for row in baseline_rows
    }
    paired = [
        (row, baseline_lookup[(row["origin_index"], row["horizon"], row["series_index"])])
        for row in model_rows
        if (row["origin_index"], row["horizon"], row["series_index"]) in baseline_lookup
    ]
    if not paired:
        return float("nan"), float("nan"), float("nan")
    by_origin: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for pair in paired:
        by_origin.setdefault(int(pair[0]["origin_index"]), []).append(pair)
    origins = np.array(sorted(by_origin), dtype=int)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        sampled = rng.choice(origins, size=len(origins), replace=True)
        model_abs = baseline_abs = target_sum = 0.0
        for origin in sampled:
            for model_row, baseline_row in by_origin[int(origin)]:
                target = float(model_row["target"])
                model_abs += abs(float(model_row["prediction"]) - target)
                baseline_abs += abs(float(baseline_row["prediction"]) - target)
                target_sum += abs(target)
        if target_sum > 0:
            deltas.append(100.0 * (model_abs - baseline_abs) / target_sum)
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    values = np.array(deltas)
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def run_rolling_backtest(
    panel_dir: str | Path,
    graph_dir: str | Path,
    output_dir: str | Path,
    spec: BacktestSpec | None = None,
    *,
    overwrite: bool = False,
) -> BacktestResult:
    """Run expanding-window validation and held-out test evaluation."""
    spec = spec or BacktestSpec()
    spec.validate()
    panel = load_panel(panel_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    adjacency = np.load(Path(graph_dir) / "adjacency.npy")
    identity = np.load(Path(graph_dir) / "identity_adjacency.npy")
    permuted = np.load(Path(graph_dir) / "permuted_adjacency.npy")
    if adjacency.shape != (len(panel.series_keys), len(panel.series_keys)):
        raise RealDataError("Graph node count does not match panel series count.")

    train_end = int(panel.manifest["boundaries"]["train_end_index"])
    test_end = int(panel.manifest["boundaries"]["test_end_index"])
    max_horizon = max(spec.horizons)
    first_origin = max(train_end, spec.history - 1)
    origins = list(range(first_origin, test_end - max_horizon + 1, spec.origin_step))
    if not origins:
        raise RealDataError("No rolling origins fit inside the panel boundaries.")

    prediction_rows: list[dict[str, Any]] = []
    for origin in origins:
        naive_predictions = {
            "last_value": last_value_forecast(panel.sales, origin, spec.horizons),
            "seasonal_naive_12": seasonal_naive_forecast(panel.sales, origin, spec.horizons),
            "moving_average_3": moving_average_forecast(panel.sales, origin, spec.horizons, 3),
            "moving_average_6": moving_average_forecast(
                panel.sales, origin, spec.horizons, 6
            ),
            "drift": drift_forecast(
                panel.sales, origin, spec.horizons, spec.history
            ),
            "damped_trend": damped_trend_forecast(
                panel.sales, origin, spec.horizons, spec.history
            ),
            "seasonal_growth": seasonal_growth_forecast(
                panel.sales, origin, spec.horizons
            ),
            "regime_guarded": regime_guarded_forecast(
                panel.sales, origin, spec.horizons, max(24, spec.history)
            ),
        }
        month_numbers = np.array(
            [int(value[5:7]) for value in panel.months], dtype=np.int64
        )
        ridge_models = {
            "pooled_ridge": PooledResidualRidge(
                alpha=spec.ridge_alpha, month_numbers=month_numbers
            ),
            "seasonal_residual_ridge": PooledResidualRidge(
                alpha=spec.ridge_alpha,
                month_numbers=month_numbers,
                anchor="seasonal_naive_12",
            ),
            "graph_ridge": PooledResidualRidge(
                alpha=spec.ridge_alpha, adjacency=adjacency, month_numbers=month_numbers
            ),
            "identity_graph_ridge": PooledResidualRidge(
                alpha=spec.ridge_alpha, adjacency=identity, month_numbers=month_numbers
            ),
            "permuted_graph_ridge": PooledResidualRidge(
                alpha=spec.ridge_alpha, adjacency=permuted, month_numbers=month_numbers
            ),
        }
        for model in ridge_models.values():
            model.fit(
                panel.sales,
                panel.static_features,
                origin,
                history=spec.history,
                horizons=spec.horizons,
            )
        predictions = {
            **naive_predictions,
            **{
                name: model.predict(
                    panel.sales, panel.static_features, origin, spec.horizons
                )
                for name, model in ridge_models.items()
            },
        }

        for horizon_position, horizon in enumerate(spec.horizons):
            target_index = origin + horizon
            split = _origin_split(panel, target_index)
            target = panel.sales[target_index]
            mask = (
                panel.observed_mask[target_index] > 0.5
                if spec.score_observed_only
                else panel.availability_mask[target_index] > 0.5
            )
            for model_name, matrix in predictions.items():
                for series_index in np.flatnonzero(mask):
                    prediction_rows.append(
                        {
                            "model_name": model_name,
                            "split": split,
                            "origin_index": origin,
                            "origin_month": panel.months[origin],
                            "target_index": target_index,
                            "target_month": panel.months[target_index],
                            "horizon": horizon,
                            "series_index": int(series_index),
                            "series_key": panel.series_keys[int(series_index)],
                            "target": float(target[series_index]),
                            "prediction": float(matrix[horizon_position, series_index]),
                            "error": float(matrix[horizon_position, series_index] - target[series_index]),
                        }
                    )

    fieldnames = [
        "model_name", "split", "origin_index", "origin_month", "target_index",
        "target_month", "horizon", "series_index", "series_key", "target",
        "prediction", "error",
    ]
    write_csv_rows(destination / "predictions.csv", fieldnames, prediction_rows)

    summary_rows: list[dict[str, Any]] = []
    model_names = sorted({row["model_name"] for row in prediction_rows})
    for split in ("validation", "test"):
        for horizon in spec.horizons:
            baseline_rows = [
                row for row in prediction_rows
                if row["split"] == split and row["horizon"] == horizon
                and row["model_name"] == "last_value"
            ]
            for model_name in model_names:
                rows = [
                    row for row in prediction_rows
                    if row["split"] == split and row["horizon"] == horizon
                    and row["model_name"] == model_name
                ]
                target = np.array([row["target"] for row in rows], dtype=np.float64)
                prediction = np.array([row["prediction"] for row in rows], dtype=np.float64)
                metrics = _metrics(target, prediction)
                delta_mean, delta_low, delta_high = _bootstrap_delta(
                    rows, baseline_rows, spec.bootstrap_samples,
                    spec.seed + horizon + (0 if split == "validation" else 1000),
                )
                summary_rows.append(
                    {
                        "split": split,
                        "horizon": horizon,
                        "model_name": model_name,
                        "forecast_cells": len(rows),
                        "origins": len({row["origin_index"] for row in rows}),
                        **metrics,
                        "delta_wape_vs_last_value_pp": delta_mean,
                        "delta_wape_ci95_low_pp": delta_low,
                        "delta_wape_ci95_high_pp": delta_high,
                    }
                )
    summary_rows.sort(key=lambda row: (row["split"], row["horizon"], row["wape_percent"]))
    write_csv_rows(
        destination / "leaderboard.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )

    per_series_rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for model_name in model_names:
            for series_index, series_key in enumerate(panel.series_keys):
                rows = [
                    row for row in prediction_rows
                    if row["split"] == split and row["model_name"] == model_name
                    and row["series_index"] == series_index
                ]
                if not rows:
                    continue
                target = np.array([row["target"] for row in rows], dtype=np.float64)
                prediction = np.array([row["prediction"] for row in rows], dtype=np.float64)
                per_series_rows.append(
                    {
                        "split": split,
                        "model_name": model_name,
                        "series_index": series_index,
                        "series_key": series_key,
                        "forecast_cells": len(rows),
                        **_metrics(target, prediction),
                    }
                )
    write_csv_rows(
        destination / "per_series_metrics.csv",
        list(per_series_rows[0].keys()),
        per_series_rows,
    )

    validation_rows = [row for row in summary_rows if row["split"] == "validation"]
    test_rows = [row for row in summary_rows if row["split"] == "test"]
    selected_models: dict[str, str] = {}
    selected_test_rows: list[dict[str, Any]] = []
    for horizon in spec.horizons:
        validation_candidates = [
            row for row in validation_rows if row["horizon"] == horizon
        ]
        selected = min(validation_candidates, key=lambda row: row["wape_percent"])
        selected_models[str(horizon)] = str(selected["model_name"])
        test_result = next(
            row for row in test_rows
            if row["horizon"] == horizon
            and row["model_name"] == selected["model_name"]
        )
        selected_test_rows.append(
            {
                "horizon": horizon,
                "selected_model": selected["model_name"],
                "validation_wape_percent": selected["wape_percent"],
                "test_wape_percent": test_result["wape_percent"],
                "test_mae": test_result["mae"],
                "test_rmse": test_result["rmse"],
                "test_bias": test_result["bias"],
                "test_delta_wape_vs_last_value_pp": test_result["delta_wape_vs_last_value_pp"],
                "test_delta_wape_ci95_low_pp": test_result["delta_wape_ci95_low_pp"],
                "test_delta_wape_ci95_high_pp": test_result["delta_wape_ci95_high_pp"],
            }
        )
    write_csv_rows(
        destination / "selected_model_test_metrics.csv",
        list(selected_test_rows[0].keys()),
        selected_test_rows,
    )

    lines = [
        "# Stage 2 Rolling-Origin Backtest",
        "",
        "All ridge models are re-fit at each origin using data available at that origin.",
        "Targets are scored only when the original monthly value was observed."
        if spec.score_observed_only else
        "Targets are scored across the available series lifespan.",
        "",
    ]
    for split in ("validation", "test"):
        lines.extend([f"## {split.title()}", ""])
        for horizon in spec.horizons:
            ranked = [
                row for row in summary_rows
                if row["split"] == split and row["horizon"] == horizon
            ]
            lines.append(f"### Horizon {horizon} month(s)")
            lines.append("")
            lines.append("| Rank | Model | WAPE | MAE | RMSE | Bias | Δ WAPE vs LV | 95% CI |")
            lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
            for rank, row in enumerate(ranked, start=1):
                lines.append(
                    f"| {rank} | {row['model_name']} | {row['wape_percent']:.3f}% | "
                    f"{row['mae']:.3f} | {row['rmse']:.3f} | {row['bias']:.3f} | "
                    f"{row['delta_wape_vs_last_value_pp']:.3f} pp | "
                    f"[{row['delta_wape_ci95_low_pp']:.3f}, {row['delta_wape_ci95_high_pp']:.3f}] |"
                )
            lines.append("")
    lines.extend(["## Validation-selected models on held-out test", ""])
    lines.append("| Horizon | Selected model | Validation WAPE | Test WAPE | Test Δ vs LV | 95% CI |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for row in selected_test_rows:
        lines.append(
            f"| {row['horizon']} | {row['selected_model']} | "
            f"{row['validation_wape_percent']:.3f}% | {row['test_wape_percent']:.3f}% | "
            f"{row['test_delta_wape_vs_last_value_pp']:.3f} pp | "
            f"[{row['test_delta_wape_ci95_low_pp']:.3f}, {row['test_delta_wape_ci95_high_pp']:.3f}] |"
        )
    lines.append("")
    lines.extend(
        [
            "## Decision checks",
            "",
            "- Negative Δ WAPE means improvement over Last Value.",
            "- A graph claim requires graph_ridge to beat pooled, identity, and permuted controls.",
            "- Validation chooses a model; held-out test measures the final claim.",
            "",
        ]
    )
    (destination / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    oracle_test_winners = {
        str(horizon): min(
            (row for row in test_rows if row["horizon"] == horizon),
            key=lambda row: row["wape_percent"],
        )["model_name"]
        for horizon in spec.horizons
    }
    graph_checks = {}
    for horizon in spec.horizons:
        rows = {
            row["model_name"]: row for row in test_rows if row["horizon"] == horizon
        }
        graph_checks[str(horizon)] = {
            "beats_pooled": rows["graph_ridge"]["wape_percent"] < rows["pooled_ridge"]["wape_percent"],
            "beats_identity": rows["graph_ridge"]["wape_percent"] < rows["identity_graph_ridge"]["wape_percent"],
            "beats_permuted": rows["graph_ridge"]["wape_percent"] < rows["permuted_graph_ridge"]["wape_percent"],
        }

    identity_payload = {
        "schema_version": "autogfm-rolling-backtest-v2",
        "panel_id": panel.manifest["panel_id"],
        "graph_sha256": sha256_file(Path(graph_dir) / "adjacency.npy"),
        "spec": asdict(spec),
        "predictions_sha256": sha256_file(destination / "predictions.csv"),
    }
    backtest_id = stable_payload_id(identity_payload)
    manifest = {
        **identity_payload,
        "backtest_id": backtest_id,
        "origins": len(origins),
        "models": model_names,
        "selected_models": selected_models,
        "selected_model_test_metrics": selected_test_rows,
        "oracle_test_winners": oracle_test_winners,
        "test_winners": oracle_test_winners,
        "graph_claim_checks": graph_checks,
        "files": {
            name: {"path": name, "sha256": sha256_file(destination / name)}
            for name in (
                "predictions.csv", "leaderboard.csv", "per_series_metrics.csv",
                "selected_model_test_metrics.csv", "summary.md",
            )
        },
    }
    write_json(destination / "manifest.json", manifest)
    return BacktestResult(destination, backtest_id, manifest)
