"""Lightweight forecasting ideas adapted from published forecasting models.

These are short-series adaptations, not exact benchmark reproductions.  They
retain the central mechanisms of NLinear (last-value normalization), DLinear
(series decomposition), and gated expert selection while using deterministic
ridge fitting suitable for 76 monthly observations.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from autogfm.forecasting.baselines import _fit_ridge, _predict_ridge
from autogfm.forecasting.regime import regime_feature_vector


def _history_window(values: np.ndarray, origin: int, history: int) -> np.ndarray:
    start = max(0, origin - history + 1)
    window = values[start : origin + 1]
    if len(window) < history:
        window = np.pad(window, (history - len(window), 0), mode="edge")
    return window.astype(np.float64, copy=False)


def _moving_average(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer of at least 3.")
    half = kernel_size // 2
    padded = np.pad(values, (half, half), mode="edge")
    kernel = np.full(kernel_size, 1.0 / kernel_size, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def _fit_direct_ridge(
    features: list[np.ndarray],
    targets: list[float],
    current_feature: np.ndarray,
    *,
    alpha: float,
    fallback: float,
) -> float:
    if len(features) < 8:
        return fallback
    state = _fit_ridge(np.stack(features), np.asarray(targets), alpha)
    return float(_predict_ridge(state, current_feature[None, :])[0])


def nlinear_ridge_forecast(
    values: np.ndarray,
    months: list[str],
    origin: int,
    horizon: int,
    *,
    history: int = 18,
    alpha: float = 6.0,
) -> float:
    """NLinear-inspired direct forecast with last-value normalization."""
    array = np.asarray(values, dtype=np.float64)
    log_values = np.log1p(np.clip(array, 0.0, None))
    first_origin = max(history - 1, 12)
    features: list[np.ndarray] = []
    targets: list[float] = []
    for training_origin in range(first_origin, origin - horizon + 1):
        target_index = training_origin + horizon
        if target_index > origin:
            continue
        window = _history_window(log_values, training_origin, history)
        last = float(window[-1])
        feature = np.concatenate(
            [
                window - last,
                regime_feature_vector(
                    array,
                    months,
                    training_origin,
                    horizon,
                ),
            ]
        )
        features.append(feature)
        targets.append(float(log_values[target_index] - last))
    current_window = _history_window(log_values, origin, history)
    current_last = float(current_window[-1])
    current_feature = np.concatenate(
        [
            current_window - current_last,
            regime_feature_vector(array, months, origin, horizon),
        ]
    )
    normalized = _fit_direct_ridge(
        features,
        targets,
        current_feature,
        alpha=alpha,
        fallback=0.0,
    )
    return float(max(0.0, np.expm1(np.clip(current_last + normalized, 0.0, 20.0))))


def dlinear_ridge_forecast(
    values: np.ndarray,
    months: list[str],
    origin: int,
    horizon: int,
    *,
    history: int = 24,
    kernel_size: int = 5,
    alpha: float = 10.0,
) -> float:
    """DLinear-inspired trend/remainder decomposition with linear readout."""
    array = np.asarray(values, dtype=np.float64)
    log_values = np.log1p(np.clip(array, 0.0, None))
    first_origin = max(history - 1, 12)
    features: list[np.ndarray] = []
    targets: list[float] = []
    for training_origin in range(first_origin, origin - horizon + 1):
        target_index = training_origin + horizon
        if target_index > origin:
            continue
        window = _history_window(log_values, training_origin, history)
        trend = _moving_average(window, kernel_size)
        seasonal = window - trend
        features.append(
            np.concatenate(
                [
                    seasonal,
                    trend,
                    regime_feature_vector(
                        array,
                        months,
                        training_origin,
                        horizon,
                    ),
                ]
            )
        )
        targets.append(float(log_values[target_index]))
    current_window = _history_window(log_values, origin, history)
    current_trend = _moving_average(current_window, kernel_size)
    current_feature = np.concatenate(
        [
            current_window - current_trend,
            current_trend,
            regime_feature_vector(array, months, origin, horizon),
        ]
    )
    fallback = float(log_values[origin])
    prediction = _fit_direct_ridge(
        features,
        targets,
        current_feature,
        alpha=alpha,
        fallback=fallback,
    )
    return float(max(0.0, np.expm1(np.clip(prediction, 0.0, 20.0))))


def regime_error_moe_forecast(
    current_predictions: dict[str, float],
    history_records: list[dict[str, Any]],
    current_feature: np.ndarray,
    *,
    alpha: float = 3.0,
) -> tuple[float, dict[str, float]]:
    """TFT-inspired lightweight gate using predicted expert error.

    Each expert receives a ridge model that predicts its future normalized
    absolute error from leakage-safe regime features.  Inverse predicted error
    forms the gate.  This is deterministic and interpretable.
    """
    expert_names = tuple(current_predictions)
    if len(history_records) < 10:
        equal = 1.0 / max(len(expert_names), 1)
        weights = {name: equal for name in expert_names}
        return (
            float(np.median(list(current_predictions.values()))),
            weights,
        )

    x = np.stack([np.asarray(row["feature"], dtype=float) for row in history_records])
    predicted_errors: dict[str, float] = {}
    for name in expert_names:
        y = np.asarray(
            [
                math.log1p(
                    abs(float(row["predictions"][name]) - float(row["target"]))
                    / max(abs(float(row["target"])), 1.0)
                )
                for row in history_records
            ],
            dtype=np.float64,
        )
        state = _fit_ridge(x, y, alpha)
        estimate = float(_predict_ridge(state, current_feature[None, :])[0])
        predicted_errors[name] = max(math.expm1(max(estimate, 0.0)), 0.02)

    raw = {
        name: 1.0 / min(max(error, 0.02), 2.0)
        for name, error in predicted_errors.items()
    }
    maximum = max(raw.values())
    minimum_allowed = maximum / 12.0
    clipped = {name: max(value, minimum_allowed) for name, value in raw.items()}
    total = sum(clipped.values())
    weights = {name: value / total for name, value in clipped.items()}
    forecast = sum(weights[name] * current_predictions[name] for name in expert_names)
    return float(max(0.0, forecast)), weights
