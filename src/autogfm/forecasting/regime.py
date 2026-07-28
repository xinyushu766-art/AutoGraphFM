"""Leakage-safe market-regime features for monthly demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RegimeSnapshot:
    """Interpretable state summary computed only from history at an origin."""

    label: str
    change_score: float
    mean_shift: float
    slope_shift: float
    volatility_ratio: float
    yoy_log_change: float
    recent_log_slope: float


def _robust_scale(values: np.ndarray, floor: float = 0.03) -> float:
    if values.size == 0:
        return floor
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return max(1.4826 * mad, floor)


def _slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x -= x.mean()
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12:
        return 0.0
    centered = values - float(values.mean())
    return float(np.dot(x, centered) / denominator)


def _window(log_values: np.ndarray, end: int, length: int) -> np.ndarray:
    start = max(0, end - length + 1)
    return log_values[start : end + 1]


def regime_snapshot(
    values: np.ndarray,
    origin: int,
    *,
    short_window: int = 6,
    reference_window: int = 12,
) -> RegimeSnapshot:
    """Summarize level, slope, and volatility shifts before ``origin``.

    The two-window discrepancy mirrors the logic of window-based change-point
    detection: recent and reference windows are compared without using any
    observations after the forecast origin.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if not 0 <= origin < len(array):
        raise ValueError("origin is outside the series.")
    if short_window < 2 or reference_window < 3:
        raise ValueError("regime windows are too short.")

    log_values = np.log1p(np.clip(array, 0.0, None))
    recent = _window(log_values, origin, short_window)
    reference_end = max(0, origin - len(recent))
    reference = _window(log_values, reference_end, reference_window)
    if reference.size < 3:
        reference = _window(log_values, origin, reference_window + short_window)

    reference_scale = _robust_scale(np.diff(reference))
    mean_shift = float((recent.mean() - reference.mean()) / reference_scale)
    recent_slope = _slope(recent)
    reference_slope = _slope(reference)
    slope_shift = float((recent_slope - reference_slope) / reference_scale)

    recent_diff = np.diff(recent)
    reference_diff = np.diff(reference)
    recent_vol = float(np.std(recent_diff)) if recent_diff.size else 0.0
    reference_vol = max(
        float(np.std(reference_diff)) if reference_diff.size else 0.0,
        0.02,
    )
    volatility_ratio = float(recent_vol / reference_vol)
    volatility_term = math.log(max(volatility_ratio, 1e-6))
    change_score = float(
        min(
            12.0,
            math.sqrt(
                mean_shift * mean_shift
                + slope_shift * slope_shift
                + volatility_term * volatility_term
            ),
        )
    )
    yoy = (
        float(log_values[origin] - log_values[origin - 12])
        if origin >= 12
        else 0.0
    )

    if volatility_ratio >= 2.2:
        label = "shock"
    elif yoy >= 0.12 and recent_slope > 0.0:
        label = "growth"
    elif yoy <= -0.05 or recent_slope < -0.035:
        label = "decline"
    elif change_score >= 3.0:
        label = "transition"
    else:
        label = "stable"
    return RegimeSnapshot(
        label=label,
        change_score=change_score,
        mean_shift=mean_shift,
        slope_shift=slope_shift,
        volatility_ratio=volatility_ratio,
        yoy_log_change=yoy,
        recent_log_slope=recent_slope,
    )


def regime_feature_vector(
    values: np.ndarray,
    months: list[str] | tuple[str, ...],
    origin: int,
    horizon: int,
) -> np.ndarray:
    """Return numerical state and known-calendar covariates for a gate."""
    if horizon < 1 or origin + horizon >= len(values):
        raise ValueError("horizon is not valid for the supplied origin.")
    snapshot = regime_snapshot(values, origin)
    log_values = np.log1p(np.clip(np.asarray(values, dtype=float), 0.0, None))
    mom_1 = float(log_values[origin] - log_values[max(0, origin - 1)])
    mom_3 = float(log_values[origin] - log_values[max(0, origin - 3)])
    recent = _window(log_values, origin, 12)
    level_position = float(
        (log_values[origin] - np.median(recent)) / _robust_scale(recent)
    )
    target_month = int(str(months[origin + horizon])[5:7])
    angle = 2.0 * math.pi * (target_month - 1) / 12.0
    holiday_proxy = float(target_month in {1, 2})
    return np.asarray(
        [
            snapshot.change_score,
            snapshot.mean_shift,
            snapshot.slope_shift,
            math.log(max(snapshot.volatility_ratio, 1e-6)),
            snapshot.yoy_log_change,
            snapshot.recent_log_slope,
            mom_1,
            mom_3,
            level_position,
            math.sin(angle),
            math.cos(angle),
            holiday_proxy,
            min(float(horizon) / 12.0, 1.0),
        ],
        dtype=np.float64,
    )
