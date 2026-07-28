"""Leakage-safe aggregate-market benchmarks with strong non-graph baselines."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from autogfm.forecasting.baselines import (
    _fit_ridge,
    _predict_ridge,
    damped_trend_forecast,
    regime_guarded_forecast,
    seasonal_growth_forecast,
)
from autogfm.forecasting.literature_models import (
    dlinear_ridge_forecast,
    nlinear_ridge_forecast,
    regime_error_moe_forecast,
)
from autogfm.forecasting.regime import (
    regime_feature_vector,
    regime_snapshot,
)
from autogfm.realdata.common import sha256_file, stable_payload_id


STAGE10A_MODEL_NAMES = (
    "last_value",
    "seasonal_naive_12",
    "moving_average_3",
    "moving_average_6",
    "drift",
    "damped_trend",
    "seasonal_growth",
    "autoregressive_ridge",
    "seasonal_residual_ridge",
    "regime_guarded",
    "robust_ensemble",
)

LITERATURE_MODEL_NAMES = (
    "nlinear_ridge",
    "dlinear_ridge",
    "regime_error_moe",
)

STAGE10B_MODEL_NAMES = STAGE10A_MODEL_NAMES + LITERATURE_MODEL_NAMES
MODEL_NAMES = STAGE10A_MODEL_NAMES
REGIME_EXPERT_NAMES = (
    "seasonal_naive_12",
    "damped_trend",
    "seasonal_residual_ridge",
    "nlinear_ridge",
    "dlinear_ridge",
)


@dataclass(frozen=True)
class MarketBenchmarkResult:
    output_dir: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class MarketFoldSpec:
    name: str
    train_end: str
    validation_end: str
    test_end: str

    def validate(self) -> None:
        if not self.train_end < self.validation_end < self.test_end:
            raise ValueError(
                "Expected train_end < validation_end < test_end for each fold."
            )


@dataclass(frozen=True)
class MultiWindowMarketBenchmarkResult:
    output_dir: Path
    benchmark_id: str
    manifest: dict[str, Any]


def _read_market_series(
    source_csv: str | Path,
    value_column: str,
) -> tuple[list[dict[str, str]], list[str], np.ndarray]:
    with Path(source_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Market CSV is empty.")
    required = {"year_month", value_column}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Missing required market columns: {missing}")
    months = [str(row["year_month"]) for row in rows]
    values = np.asarray([float(row[value_column]) for row in rows], dtype=float)
    if months != sorted(months) or len(set(months)) != len(months):
        raise ValueError("Months must be sorted and unique.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Market values must be finite and non-negative.")
    return rows, months, values


def _month_number(month: str) -> int:
    return int(month[5:7])


def _anchor(values: np.ndarray, origin: int, horizon: int, kind: str) -> float:
    if kind == "last_value":
        return float(values[origin])
    reference = origin + horizon - 12
    return float(values[reference if reference >= 0 else origin])


def _direct_ridge_feature(
    values: np.ndarray,
    months: list[str],
    origin: int,
    horizon: int,
    history: int,
    anchor_kind: str,
) -> np.ndarray:
    log_values = np.log1p(np.clip(values, 0.0, None))
    start = max(0, origin - history + 1)
    window = log_values[start : origin + 1]
    if len(window) < history:
        window = np.pad(window, (history - len(window), 0), mode="edge")
    target_index = origin + horizon
    month = _month_number(months[target_index])
    angle = 2.0 * math.pi * (month - 1) / 12.0
    recent_diff = np.diff(window)
    anchor_value = math.log1p(_anchor(values, origin, horizon, anchor_kind))
    seasonal_reference = math.log1p(
        _anchor(values, origin, horizon, "seasonal_naive_12")
    )
    return np.concatenate(
        [
            window,
            np.array(
                [
                    window[-1],
                    window[-3:].mean(),
                    window[-6:].mean(),
                    window[-12:].mean(),
                    np.median(recent_diff[-6:]) if len(recent_diff) else 0.0,
                    np.std(recent_diff[-6:]) if len(recent_diff) else 0.0,
                    np.sin(angle),
                    np.cos(angle),
                    origin / max(len(values) - 1, 1),
                    anchor_value,
                    seasonal_reference,
                    seasonal_reference - window[-1],
                ],
                dtype=float,
            ),
        ]
    )


def _direct_ridge_forecast(
    values: np.ndarray,
    months: list[str],
    origin: int,
    horizon: int,
    *,
    anchor_kind: str,
    history: int = 18,
    alpha: float = 8.0,
) -> float:
    first_origin = max(history - 1, 12)
    training_origins = range(first_origin, origin - horizon + 1)
    x_rows: list[np.ndarray] = []
    targets: list[float] = []
    log_values = np.log1p(np.clip(values, 0.0, None))
    for training_origin in training_origins:
        target_index = training_origin + horizon
        if target_index > origin:
            continue
        x_rows.append(
            _direct_ridge_feature(
                values,
                months,
                training_origin,
                horizon,
                history,
                anchor_kind,
            )
        )
        anchor_value = math.log1p(
            _anchor(values, training_origin, horizon, anchor_kind)
        )
        targets.append(float(log_values[target_index] - anchor_value))
    if len(x_rows) < 8:
        return _anchor(values, origin, horizon, anchor_kind)
    state = _fit_ridge(np.stack(x_rows), np.asarray(targets), alpha)
    feature = _direct_ridge_feature(
        values,
        months,
        origin,
        horizon,
        history,
        anchor_kind,
    )[None, :]
    residual = float(_predict_ridge(state, feature)[0])
    anchor_value = math.log1p(_anchor(values, origin, horizon, anchor_kind))
    return float(max(0.0, np.expm1(np.clip(anchor_value + residual, 0.0, 20.0))))


def _forecast(
    values: np.ndarray,
    months: list[str],
    origin: int,
    horizon: int,
    model: str,
) -> float:
    if model == "last_value":
        return float(values[origin])
    if model == "seasonal_naive_12":
        return _anchor(values, origin, horizon, "seasonal_naive_12")
    if model.startswith("moving_average_"):
        window = int(model.rsplit("_", 1)[1])
        return float(values[max(0, origin - window + 1) : origin + 1].mean())
    if model == "drift":
        history = min(24, origin)
        start = origin - history
        slope = (values[origin] - values[start]) / max(origin - start, 1)
        return float(max(0.0, values[origin] + horizon * slope))

    panel = values[:, None]
    if model == "damped_trend":
        return float(
            damped_trend_forecast(panel, origin, (horizon,), history=18)[0, 0]
        )
    if model == "seasonal_growth":
        return float(
            seasonal_growth_forecast(panel, origin, (horizon,))[0, 0]
        )
    if model == "autoregressive_ridge":
        return _direct_ridge_forecast(
            values,
            months,
            origin,
            horizon,
            anchor_kind="last_value",
        )
    if model == "seasonal_residual_ridge":
        return _direct_ridge_forecast(
            values,
            months,
            origin,
            horizon,
            anchor_kind="seasonal_naive_12",
        )
    if model == "nlinear_ridge":
        return nlinear_ridge_forecast(values, months, origin, horizon)
    if model == "dlinear_ridge":
        return dlinear_ridge_forecast(values, months, origin, horizon)
    if model == "regime_error_moe":
        raise ValueError("regime_error_moe is generated with historical gate state.")
    if model == "regime_guarded":
        return float(
            regime_guarded_forecast(panel, origin, (horizon,), history=24)[0, 0]
        )
    if model == "robust_ensemble":
        components = (
            _forecast(values, months, origin, horizon, "seasonal_naive_12"),
            _forecast(values, months, origin, horizon, "damped_trend"),
            _forecast(values, months, origin, horizon, "autoregressive_ridge"),
            _forecast(values, months, origin, horizon, "regime_guarded"),
        )
        return float(np.median(np.asarray(components, dtype=float)))
    raise ValueError(f"Unknown model: {model}")


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "wape_percent": float("nan"),
            "smape_percent": float("nan"),
            "bias": float("nan"),
        }
    target = np.asarray([row["target"] for row in rows], dtype=float)
    pred = np.asarray([row["prediction"] for row in rows], dtype=float)
    error = pred - target
    absolute = np.abs(error)
    smape = np.mean(2.0 * absolute / (np.abs(pred) + np.abs(target) + 1e-9))
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape_percent": float(
            100.0 * absolute.sum() / max(np.abs(target).sum(), 1e-9)
        ),
        "smape_percent": float(100.0 * smape),
        "bias": float(error.mean()),
    }


def _generate_predictions(
    values: np.ndarray,
    months: list[str],
    fold: MarketFoldSpec,
    horizons: Iterable[int],
    models: tuple[str, ...] = MODEL_NAMES,
) -> list[dict[str, Any]]:
    fold.validate()
    requested_models = tuple(models)
    unknown = sorted(set(requested_models) - set(STAGE10B_MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown benchmark models: {unknown}")
    predictions: list[dict[str, Any]] = []
    use_moe = "regime_error_moe" in requested_models
    calculation_models = [
        model for model in requested_models if model != "regime_error_moe"
    ]
    if use_moe:
        for expert in REGIME_EXPERT_NAMES:
            if expert not in calculation_models:
                calculation_models.append(expert)

    for horizon in horizons:
        history_records: list[dict[str, Any]] = []
        for origin in range(11, len(values) - int(horizon)):
            target_index = origin + int(horizon)
            target_month = months[target_index]
            if target_month > fold.test_end:
                continue
            split = (
                "train"
                if target_month <= fold.train_end
                else "validation"
                if target_month <= fold.validation_end
                else "test"
            )
            model_predictions = {
                model: _forecast(
                    values,
                    months,
                    origin,
                    int(horizon),
                    model,
                )
                for model in calculation_models
            }
            feature = regime_feature_vector(
                values,
                months,
                origin,
                int(horizon),
            )
            weights: dict[str, float] = {}
            if use_moe:
                known_history = [
                    row for row in history_records if row["target_index"] <= origin
                ]
                expert_predictions = {
                    name: model_predictions[name] for name in REGIME_EXPERT_NAMES
                }
                moe_prediction, weights = regime_error_moe_forecast(
                    expert_predictions,
                    known_history,
                    feature,
                )
                model_predictions["regime_error_moe"] = moe_prediction

            snapshot = regime_snapshot(values, origin)
            target = float(values[target_index])
            for model in requested_models:
                prediction = float(model_predictions[model])
                predictions.append(
                    {
                        "fold": fold.name,
                        "model_name": model,
                        "split": split,
                        "origin_month": months[origin],
                        "target_month": target_month,
                        "horizon": int(horizon),
                        "target": target,
                        "prediction": prediction,
                        "error": prediction - target,
                        "regime_label": snapshot.label,
                        "change_score": snapshot.change_score,
                        "expert_weights": (
                            json.dumps(weights, sort_keys=True)
                            if model == "regime_error_moe"
                            else ""
                        ),
                    }
                )
            if use_moe:
                history_records.append(
                    {
                        "target_index": target_index,
                        "feature": feature,
                        "target": target,
                        "predictions": {
                            name: model_predictions[name]
                            for name in REGIME_EXPERT_NAMES
                        },
                    }
                )
    return predictions


def _summarize_predictions(
    predictions: list[dict[str, Any]],
    horizons: Iterable[int],
    models: tuple[str, ...] = MODEL_NAMES,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    fold_names = sorted({str(row["fold"]) for row in predictions})
    for fold in fold_names:
        for split in ("validation", "test"):
            for horizon in horizons:
                for model in models:
                    selected = [
                        row
                        for row in predictions
                        if row["fold"] == fold
                        and row["split"] == split
                        and row["horizon"] == int(horizon)
                        and row["model_name"] == model
                    ]
                    if selected:
                        summary.append(
                            {
                                "fold": fold,
                                "split": split,
                                "horizon": int(horizon),
                                "model_name": model,
                                "n": len(selected),
                                **_metrics(selected),
                            }
                        )
    return summary


def _select_fold_models(
    predictions: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    horizons: Iterable[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_metrics: list[dict[str, Any]] = []
    selected_predictions: list[dict[str, Any]] = []
    fold_names = sorted({str(row["fold"]) for row in predictions})
    for fold in fold_names:
        for horizon in horizons:
            candidates = [
                row
                for row in summary
                if row["fold"] == fold
                and row["split"] == "validation"
                and row["horizon"] == int(horizon)
            ]
            if not candidates:
                continue
            winner = min(
                candidates,
                key=lambda row: (row["wape_percent"], row["model_name"]),
            )
            test_rows = [
                row
                for row in predictions
                if row["fold"] == fold
                and row["split"] == "test"
                and row["horizon"] == int(horizon)
                and row["model_name"] == winner["model_name"]
            ]
            if not test_rows:
                continue
            selected_predictions.extend(test_rows)
            selected_metrics.append(
                {
                    "fold": fold,
                    "horizon": int(horizon),
                    "selected_model": winner["model_name"],
                    "validation_n": winner["n"],
                    "validation_wape_percent": winner["wape_percent"],
                    "test_n": len(test_rows),
                    **{
                        f"test_{key}": value
                        for key, value in _metrics(test_rows).items()
                    },
                }
            )
    return selected_metrics, selected_predictions


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _default_folds(months: list[str]) -> tuple[MarketFoldSpec, ...]:
    first_year = int(months[0][:4])
    final_year = int(months[-1][:4])
    final_month = months[-1]
    folds: list[MarketFoldSpec] = []
    for test_year in range(first_year + 4, final_year + 1):
        train_end = f"{test_year - 2:04d}-12-01"
        validation_end = f"{test_year - 1:04d}-12-01"
        test_end = min(f"{test_year:04d}-12-01", final_month)
        if validation_end >= final_month or test_end <= validation_end:
            continue
        folds.append(
            MarketFoldSpec(
                name=f"test_{test_year}",
                train_end=train_end,
                validation_end=validation_end,
                test_end=test_end,
            )
        )
    if not folds:
        raise ValueError("The series is too short to construct rolling annual folds.")
    return tuple(folds)


def run_market_benchmark(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    value_column: str = "EV_Sales",
    train_end: str = "2022-12-01",
    validation_end: str = "2023-12-01",
    test_end: str | None = None,
    horizons: tuple[int, ...] = (1, 3),
    models: tuple[str, ...] = MODEL_NAMES,
    overwrite: bool = False,
) -> MarketBenchmarkResult:
    """Run the backward-compatible single-split aggregate benchmark."""
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    rows, months, values = _read_market_series(source_csv, value_column)
    fold = MarketFoldSpec(
        name="single_split",
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end or months[-1],
    )
    predictions = _generate_predictions(values, months, fold, horizons, models)
    summary = _summarize_predictions(predictions, horizons, models)
    selected_metrics, selected_predictions = _select_fold_models(
        predictions,
        summary,
        horizons,
    )
    _write_csv(destination / "predictions.csv", predictions)
    _write_csv(destination / "metrics.csv", summary)
    _write_csv(destination / "selected_test_predictions.csv", selected_predictions)

    ev_share = np.asarray(
        [float(row["EV_Sales"]) / float(row["Total_Sales"]) for row in rows]
    )
    selected_models = {
        str(row["horizon"]): str(row["selected_model"])
        for row in selected_metrics
    }
    test_metrics = [
        {
            "split": "test",
            "horizon": row["horizon"],
            "model_name": row["selected_model"],
            "n": row["test_n"],
            "mae": row["test_mae"],
            "rmse": row["test_rmse"],
            "wape_percent": row["test_wape_percent"],
            "smape_percent": row["test_smape_percent"],
            "bias": row["test_bias"],
        }
        for row in selected_metrics
    ]
    manifest = {
        "evidence_scope": "public_real_aggregate_market_series",
        "source_rows": len(rows),
        "month_min": months[0],
        "month_max": months[-1],
        "value_column": value_column,
        "train_end": train_end,
        "validation_end": validation_end,
        "test_end": fold.test_end,
        "models": list(models),
        "selected_models": selected_models,
        "test_metrics": test_metrics,
        "ev_share": {
            "first_percent": float(100 * ev_share[0]),
            "last_percent": float(100 * ev_share[-1]),
            "peak_percent": float(100 * ev_share.max()),
            "peak_month": months[int(ev_share.argmax())],
        },
        "claim_boundary": (
            "This benchmark validates aggregate-market forecasting only; it "
            "does not evaluate model-level graphs or graph research claims."
        ),
    }
    (destination / "BENCHMARK_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_market_html(destination, months, selected_metrics, ev_share)
    return MarketBenchmarkResult(destination, manifest)


def _write_market_html(
    destination: Path,
    months: list[str],
    selected_metrics: list[dict[str, Any]],
    ev_share: np.ndarray,
) -> None:
    html_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{row[key]}</td>"
            for key in (
                "horizon",
                "selected_model",
                "test_n",
                "test_wape_percent",
                "test_mae",
                "test_rmse",
                "test_bias",
            )
        )
        + "</tr>"
        for row in selected_metrics
    )
    (destination / "REAL_MARKET_BENCHMARK.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>AutoGraphFM Real Market Benchmark</title>"
        "<style>body{font-family:Arial;max-width:1080px;margin:40px auto;"
        "line-height:1.55}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #bbb;padding:8px}th{background:#eee}"
        ".warning{padding:12px;background:#fff3cd}</style>"
        "<h1>AutoGraphFM Stage 10A — Strong Real-Market Baselines</h1>"
        f"<p>Coverage: {months[0]} to {months[-1]} · {len(months)} months.</p>"
        "<div class='warning'><b>Claim boundary:</b> Aggregate market evidence "
        "only. No model-level graph claim is permitted.</div>"
        "<h2>Validation-selected test performance</h2><table><tr>"
        "<th>Horizon</th><th>Model</th><th>N</th><th>WAPE %</th><th>MAE</th>"
        "<th>RMSE</th><th>Bias</th></tr>"
        + html_rows
        + "</table>"
        f"<h2>EV share</h2><p>Start: {100 * ev_share[0]:.2f}% · "
        f"End: {100 * ev_share[-1]:.2f}% · Peak: {100 * ev_share.max():.2f}% "
        f"({months[int(ev_share.argmax())]})</p>",
        encoding="utf-8",
    )


def run_multiwindow_market_benchmark(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    value_column: str = "EV_Sales",
    horizons: tuple[int, ...] = (1, 3, 6),
    folds: tuple[MarketFoldSpec, ...] | None = None,
    models: tuple[str, ...] = MODEL_NAMES,
    overwrite: bool = False,
) -> MultiWindowMarketBenchmarkResult:
    """Run annual expanding-window selection and rolling held-out evaluation."""
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    _, months, values = _read_market_series(source_csv, value_column)
    selected_folds = folds or _default_folds(months)
    for fold in selected_folds:
        fold.validate()
        if fold.test_end > months[-1]:
            raise ValueError(f"Fold {fold.name} exceeds source coverage.")

    predictions: list[dict[str, Any]] = []
    for fold in selected_folds:
        predictions.extend(
            _generate_predictions(values, months, fold, horizons, models)
        )
    summary = _summarize_predictions(predictions, horizons, models)
    selected_metrics, selected_predictions = _select_fold_models(
        predictions,
        summary,
        horizons,
    )

    aggregate_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        rows = [
            row for row in selected_predictions if row["horizon"] == int(horizon)
        ]
        if rows:
            aggregate_rows.append(
                {
                    "horizon": int(horizon),
                    "policy": "validation_selected_per_fold",
                    "folds": len({row["fold"] for row in rows}),
                    "n": len(rows),
                    **_metrics(rows),
                }
            )

    stability_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        counts = Counter(
            str(row["selected_model"])
            for row in selected_metrics
            if row["horizon"] == int(horizon)
        )
        total = sum(counts.values())
        ordered = sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
        for model, count in ordered:
            stability_rows.append(
                {
                    "horizon": int(horizon),
                    "model_name": model,
                    "selected_folds": count,
                    "total_folds": total,
                    "selection_share": count / max(total, 1),
                }
            )

    _write_csv(destination / "predictions.csv", predictions)
    _write_csv(destination / "fold_leaderboard.csv", summary)
    _write_csv(destination / "selected_fold_metrics.csv", selected_metrics)
    _write_csv(destination / "selected_test_predictions.csv", selected_predictions)
    _write_csv(destination / "aggregate_selected_metrics.csv", aggregate_rows)
    _write_csv(destination / "selection_stability.csv", stability_rows)

    identity = {
        "schema_version": "autogfm-market-multiwindow-v1",
        "source_sha256": sha256_file(source_csv),
        "value_column": value_column,
        "horizons": horizons,
        "folds": [asdict(fold) for fold in selected_folds],
        "models": list(models),
    }
    benchmark_id = stable_payload_id(identity)
    manifest = {
        **identity,
        "benchmark_id": benchmark_id,
        "evidence_scope": "public_real_aggregate_market_multiwindow",
        "source_rows": len(values),
        "month_min": months[0],
        "month_max": months[-1],
        "selected_fold_metrics": selected_metrics,
        "aggregate_selected_metrics": aggregate_rows,
        "selection_stability": stability_rows,
        "claim_boundary": (
            "Multi-window aggregate evidence improves model-selection reliability "
            "but cannot establish a vehicle-graph effect."
        ),
    }
    (destination / "MULTIWINDOW_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_multiwindow_html(destination, manifest)
    return MultiWindowMarketBenchmarkResult(destination, benchmark_id, manifest)


def _write_multiwindow_html(destination: Path, manifest: dict[str, Any]) -> None:
    fold_rows = "".join(
        "<tr>"
        f"<td>{row['fold']}</td><td>{row['horizon']}</td>"
        f"<td>{row['selected_model']}</td>"
        f"<td>{row['validation_wape_percent']:.2f}%</td>"
        f"<td>{row['test_wape_percent']:.2f}%</td>"
        f"<td>{row['test_n']}</td></tr>"
        for row in manifest["selected_fold_metrics"]
    )
    aggregate_rows = "".join(
        "<tr>"
        f"<td>{row['horizon']}</td><td>{row['folds']}</td><td>{row['n']}</td>"
        f"<td>{row['wape_percent']:.2f}%</td><td>{row['mae']:.0f}</td>"
        f"<td>{row['rmse']:.0f}</td><td>{row['bias']:.0f}</td></tr>"
        for row in manifest["aggregate_selected_metrics"]
    )
    (destination / "MULTIWINDOW_MARKET_REPORT.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Stage 10A Multi-window "
        "Benchmark</title><style>body{font-family:Arial;max-width:1120px;"
        "margin:40px auto;line-height:1.55}table{border-collapse:collapse;"
        "width:100%;margin-bottom:26px}th,td{border:1px solid #bbb;padding:8px}"
        "th{background:#eee}.warning{background:#fff3cd;padding:12px}</style>"
        "<h1>AutoGraphFM Stage 10A — Multi-window Real-Market Benchmark</h1>"
        f"<p>Benchmark ID: <code>{manifest['benchmark_id']}</code></p>"
        "<div class='warning'>Aggregate market evidence only; graph claims remain "
        "blocked until the vehicle-level panel is executed.</div>"
        "<h2>Fold-level validation selection</h2><table><tr><th>Fold</th>"
        "<th>Horizon</th><th>Selected model</th><th>Validation WAPE</th>"
        "<th>Test WAPE</th><th>Test N</th></tr>"
        + fold_rows
        + "</table><h2>Aggregate selected-policy performance</h2><table><tr>"
        "<th>Horizon</th><th>Folds</th><th>N</th><th>WAPE</th><th>MAE</th>"
        "<th>RMSE</th><th>Bias</th></tr>"
        + aggregate_rows
        + "</table>",
        encoding="utf-8",
    )
