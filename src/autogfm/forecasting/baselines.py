"""Deterministic statistical and ridge baselines for monthly panel forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from autogfm.realdata.common import RealDataError


def last_value_forecast(sales: np.ndarray, origin: int, horizons: Iterable[int]) -> np.ndarray:
    values = np.clip(sales[origin], 0.0, None)
    return np.stack([values for _ in horizons], axis=0).astype(np.float32)


def seasonal_naive_forecast(
    sales: np.ndarray,
    origin: int,
    horizons: Iterable[int],
    season_length: int = 12,
) -> np.ndarray:
    forecasts = []
    for horizon in horizons:
        reference = origin + horizon - season_length
        if reference < 0:
            reference = origin
        forecasts.append(np.clip(sales[reference], 0.0, None))
    return np.stack(forecasts, axis=0).astype(np.float32)


def moving_average_forecast(
    sales: np.ndarray,
    origin: int,
    horizons: Iterable[int],
    window: int,
) -> np.ndarray:
    start = max(0, origin - window + 1)
    mean = np.clip(sales[start : origin + 1].mean(axis=0), 0.0, None)
    return np.stack([mean for _ in horizons], axis=0).astype(np.float32)


def drift_forecast(
    sales: np.ndarray,
    origin: int,
    horizons: Iterable[int],
    history: int,
) -> np.ndarray:
    start = max(0, origin - history + 1)
    span = max(origin - start, 1)
    slope = (sales[origin] - sales[start]) / span
    return np.stack(
        [np.clip(sales[origin] + horizon * slope, 0.0, None) for horizon in horizons],
        axis=0,
    ).astype(np.float32)



def damped_trend_forecast(
    sales: np.ndarray,
    origin: int,
    horizons: Iterable[int],
    history: int = 12,
    damping: float = 0.8,
) -> np.ndarray:
    """Robust log-scale damped trend forecast.

    The trend is the median recent log difference, which is less sensitive to
    one-month promotion spikes than endpoint drift.
    """
    if not 0.0 <= damping < 1.0:
        raise RealDataError("damping must be in [0, 1).")
    start = max(0, origin - history + 1)
    log_sales = np.log1p(np.clip(sales, 0.0, None)).astype(np.float64)
    recent = log_sales[start : origin + 1]
    if len(recent) < 2:
        slope = np.zeros(sales.shape[1], dtype=np.float64)
    else:
        slope = np.median(np.diff(recent, axis=0), axis=0)
    slope = np.clip(slope, -0.35, 0.35)
    last = log_sales[origin]
    forecasts = []
    for horizon in horizons:
        multiplier = sum(damping**step for step in range(int(horizon)))
        forecast = np.expm1(np.clip(last + slope * multiplier, 0.0, 20.0))
        forecasts.append(np.clip(forecast, 0.0, None))
    return np.stack(forecasts, axis=0).astype(np.float32)


def seasonal_growth_forecast(
    sales: np.ndarray,
    origin: int,
    horizons: Iterable[int],
    season_length: int = 12,
    history: int = 24,
    damping: float = 0.6,
) -> np.ndarray:
    """Seasonal-naive anchor plus robust damped annual growth."""
    if season_length < 2:
        raise RealDataError("season_length must be at least 2.")
    log_sales = np.log1p(np.clip(sales, 0.0, None)).astype(np.float64)
    start = max(season_length, origin - history + 1)
    annual_changes = [
        log_sales[index] - log_sales[index - season_length]
        for index in range(start, origin + 1)
    ]
    if annual_changes:
        annual_growth = np.median(np.stack(annual_changes), axis=0)
    else:
        annual_growth = np.zeros(sales.shape[1], dtype=np.float64)
    annual_growth = np.clip(annual_growth, -0.7, 0.7)
    forecasts = []
    for horizon in horizons:
        reference = origin + int(horizon) - season_length
        anchor = log_sales[reference] if reference >= 0 else log_sales[origin]
        forecast = np.expm1(
            np.clip(anchor + damping * annual_growth, 0.0, 20.0)
        )
        forecasts.append(np.clip(forecast, 0.0, None))
    return np.stack(forecasts, axis=0).astype(np.float32)


def regime_guarded_forecast(
    sales: np.ndarray,
    origin: int,
    horizons: Iterable[int],
    history: int = 24,
    season_length: int = 12,
) -> np.ndarray:
    """Blend damped trend with seasonal naive under detected regime shift.

    A high shift score increases reliance on the same-month-last-year anchor.
    The score uses only observations available at the forecast origin.
    """
    horizons = tuple(int(value) for value in horizons)
    trend = damped_trend_forecast(
        sales,
        origin,
        horizons,
        history=min(history, origin + 1),
        damping=0.75,
    ).astype(np.float64)
    seasonal = seasonal_naive_forecast(
        sales,
        origin,
        horizons,
        season_length=season_length,
    ).astype(np.float64)
    if origin < season_length + 6:
        return (0.5 * trend + 0.5 * seasonal).astype(np.float32)

    log_sales = np.log1p(np.clip(sales, 0.0, None)).astype(np.float64)
    yoy = log_sales[season_length : origin + 1] - log_sales[: origin + 1 - season_length]
    latest = yoy[-1]
    reference = yoy[max(0, len(yoy) - 13) : -1]
    if len(reference) == 0:
        shift_score = np.zeros_like(latest)
    else:
        center = np.median(reference, axis=0)
        mad = np.median(np.abs(reference - center), axis=0)
        shift_score = np.abs(latest - center) / np.maximum(1.4826 * mad, 0.05)

    recent_diff = np.diff(log_sales[max(0, origin - 5) : origin + 1], axis=0)
    base_diff = np.diff(log_sales[max(0, origin - 29) : max(1, origin - 5)], axis=0)
    recent_vol = np.std(recent_diff, axis=0) if len(recent_diff) else 0.0
    base_vol = np.std(base_diff, axis=0) if len(base_diff) else 0.0
    volatility_ratio = np.asarray(recent_vol) / np.maximum(np.asarray(base_vol), 0.03)
    weight = np.clip(
        np.maximum((shift_score - 0.75) / 2.5, (volatility_ratio - 1.0) / 2.0),
        0.0,
        1.0,
    )
    return (
        trend * (1.0 - weight[None, :]) + seasonal * weight[None, :]
    ).astype(np.float32)

def normalize_rows(adjacency: np.ndarray) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RealDataError("adjacency must be square.")
    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)


@dataclass(frozen=True)
class RidgeState:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeState:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise RealDataError("Invalid ridge training shapes.")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (x - mean) / scale
    design = np.concatenate(
        [np.ones((standardized.shape[0], 1)), standardized], axis=1
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    target = design.T @ y
    try:
        coefficients = np.linalg.solve(gram, target)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ target
    return RidgeState(mean, scale, coefficients)


def _predict_ridge(state: RidgeState, x: np.ndarray) -> np.ndarray:
    standardized = (x - state.mean) / state.scale
    design = np.concatenate([np.ones((x.shape[0], 1)), standardized], axis=1)
    return design @ state.coefficients


class PooledResidualRidge:
    """Pooled autoregressive ridge predicting corrections to Last Value.

    One model is fit for each requested horizon. The graph-aware version adds
    neighbor-history features while retaining exactly the same fitting code.
    """

    def __init__(
        self,
        *,
        alpha: float = 10.0,
        adjacency: np.ndarray | None = None,
        month_numbers: np.ndarray | None = None,
        anchor: str = "last_value",
    ) -> None:
        if alpha < 0:
            raise RealDataError("alpha cannot be negative.")
        self.alpha = float(alpha)
        self.adjacency = None if adjacency is None else normalize_rows(adjacency)
        self.month_numbers = (
            None if month_numbers is None
            else np.asarray(month_numbers, dtype=np.int64)
        )
        if anchor not in {"last_value", "seasonal_naive_12"}:
            raise RealDataError(
                "anchor must be last_value or seasonal_naive_12."
            )
        self.anchor = anchor
        self.states: dict[int, RidgeState] = {}
        self.history = 0

    def _feature_rows(
        self,
        sales: np.ndarray,
        static_features: np.ndarray,
        origins: np.ndarray,
        horizon: int,
        history: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        log_sales = np.log1p(np.clip(sales, 0.0, None))
        rows: list[np.ndarray] = []
        targets: list[float] = []
        for origin in origins:
            own = log_sales[origin - history + 1 : origin + 1].T
            neighbor = None
            if self.adjacency is not None:
                neighbor = (self.adjacency @ own).astype(np.float64)
            target_month_index = origin + horizon
            month_number = (
                int(self.month_numbers[target_month_index])
                if self.month_numbers is not None
                else target_month_index % 12 + 1
            )
            angle = 2.0 * np.pi * (month_number - 1) / 12.0
            calendar = np.array([np.sin(angle), np.cos(angle)], dtype=np.float64)
            for series_index in range(sales.shape[1]):
                own_lags = own[series_index]
                extras = np.array(
                    [
                        own_lags[-1],
                        own_lags[-min(3, history) :].mean(),
                        own_lags.mean(),
                        own_lags[-1] - own_lags[0],
                    ],
                    dtype=np.float64,
                )
                pieces = [own_lags, extras, calendar, static_features[series_index]]
                if neighbor is not None:
                    neighbor_lags = neighbor[series_index]
                    pieces.extend(
                        [
                            neighbor_lags,
                            np.array(
                                [
                                    neighbor_lags[-1] - own_lags[-1],
                                    neighbor_lags.mean() - own_lags.mean(),
                                ],
                                dtype=np.float64,
                            ),
                        ]
                    )
                if self.anchor == "seasonal_naive_12":
                    reference = origin + horizon - 12
                    anchor_value = (
                        log_sales[reference, series_index]
                        if reference >= 0
                        else own_lags[-1]
                    )
                    pieces.append(
                        np.array(
                            [anchor_value, anchor_value - own_lags[-1]],
                            dtype=np.float64,
                        )
                    )
                else:
                    anchor_value = own_lags[-1]
                rows.append(np.concatenate(pieces).astype(np.float64))
                targets.append(
                    float(
                        log_sales[origin + horizon, series_index]
                        - anchor_value
                    )
                )
        return np.stack(rows), np.array(targets, dtype=np.float64)

    def _prediction_rows(
        self,
        sales: np.ndarray,
        static_features: np.ndarray,
        origin: int,
        horizon: int,
        history: int,
    ) -> np.ndarray:
        log_sales = np.log1p(np.clip(sales, 0.0, None))
        own = log_sales[origin - history + 1 : origin + 1].T
        neighbor = None
        if self.adjacency is not None:
            neighbor = (self.adjacency @ own).astype(np.float64)
        target_month_index = origin + horizon
        month_number = (
            int(self.month_numbers[target_month_index])
            if self.month_numbers is not None and target_month_index < len(self.month_numbers)
            else target_month_index % 12 + 1
        )
        angle = 2.0 * np.pi * (month_number - 1) / 12.0
        calendar = np.array([np.sin(angle), np.cos(angle)], dtype=np.float64)
        rows = []
        for series_index in range(sales.shape[1]):
            own_lags = own[series_index]
            extras = np.array(
                [
                    own_lags[-1],
                    own_lags[-min(3, history) :].mean(),
                    own_lags.mean(),
                    own_lags[-1] - own_lags[0],
                ],
                dtype=np.float64,
            )
            pieces = [own_lags, extras, calendar, static_features[series_index]]
            if neighbor is not None:
                neighbor_lags = neighbor[series_index]
                pieces.extend(
                    [
                        neighbor_lags,
                        np.array(
                            [
                                neighbor_lags[-1] - own_lags[-1],
                                neighbor_lags.mean() - own_lags.mean(),
                            ],
                            dtype=np.float64,
                        ),
                    ]
                )
            if self.anchor == "seasonal_naive_12":
                reference = origin + horizon - 12
                anchor_value = (
                    log_sales[reference, series_index]
                    if reference >= 0
                    else own_lags[-1]
                )
                pieces.append(
                    np.array(
                        [anchor_value, anchor_value - own_lags[-1]],
                        dtype=np.float64,
                    )
                )
            rows.append(np.concatenate(pieces).astype(np.float64))
        return np.stack(rows)

    def fit(
        self,
        sales: np.ndarray,
        static_features: np.ndarray,
        cutoff: int,
        *,
        history: int,
        horizons: Iterable[int],
    ) -> "PooledResidualRidge":
        horizons = tuple(sorted(set(int(value) for value in horizons)))
        if not horizons or horizons[0] < 1:
            raise RealDataError("horizons must contain positive integers.")
        if cutoff >= sales.shape[0]:
            raise RealDataError("cutoff is outside sales history.")
        if history < 2:
            raise RealDataError("history must be at least 2 for ridge.")
        self.history = history
        self.states = {}
        for horizon in horizons:
            final_origin = cutoff - horizon
            origins = np.arange(history - 1, final_origin + 1, dtype=int)
            if len(origins) < 2:
                raise RealDataError(
                    f"Not enough training origins for horizon {horizon}."
                )
            x, y = self._feature_rows(
                sales, static_features, origins, horizon, history
            )
            self.states[horizon] = _fit_ridge(x, y, self.alpha)
        return self

    def predict(
        self,
        sales: np.ndarray,
        static_features: np.ndarray,
        origin: int,
        horizons: Iterable[int],
    ) -> np.ndarray:
        if not self.states:
            raise RealDataError("PooledResidualRidge must be fit before predict.")
        log_sales = np.log1p(np.clip(sales, 0.0, None))
        last_log = log_sales[origin]
        predictions = []
        for horizon in horizons:
            horizon = int(horizon)
            state = self.states.get(horizon)
            if state is None:
                raise RealDataError(f"No fitted state for horizon {horizon}.")
            x = self._prediction_rows(
                sales, static_features, origin, horizon, self.history
            )
            delta = _predict_ridge(state, x)
            if self.anchor == "seasonal_naive_12":
                reference = origin + horizon - 12
                anchor = log_sales[reference] if reference >= 0 else last_log
            else:
                anchor = last_log
            prediction = np.expm1(np.clip(anchor + delta, 0.0, 20.0))
            predictions.append(np.clip(prediction, 0.0, None))
        return np.stack(predictions, axis=0).astype(np.float32)
