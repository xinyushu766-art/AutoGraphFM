"""Leakage-safe vehicle-series lifecycle profiling.

Lifecycle labels used as model features are computed strictly at the training
cutoff. Outcome labels use the full evaluation interval and are audit-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from autogfm.realdata.common import RealDataError


@dataclass(frozen=True)
class LifecycleSpec:
    """Thresholds for train-time lifecycle and audit-only outcome labels."""

    recent_launch_months: int = 18
    inactive_gap_months: int = 3
    intermittent_missing_ratio: float = 0.25

    def validate(self) -> None:
        if self.recent_launch_months < 2:
            raise RealDataError("recent_launch_months must be at least 2.")
        if self.inactive_gap_months < 0:
            raise RealDataError("inactive_gap_months cannot be negative.")
        if not 0.0 <= self.intermittent_missing_ratio < 1.0:
            raise RealDataError(
                "intermittent_missing_ratio must be in [0, 1)."
            )


def _month_or_empty(months: Sequence[str], index: int | None) -> str:
    return "" if index is None else str(months[index])


def profile_series_lifecycle(
    raw_values: np.ndarray,
    months: Sequence[str],
    train_end_index: int,
    test_end_index: int,
    spec: LifecycleSpec | None = None,
) -> dict[str, Any]:
    """Return leakage-safe train-time lifecycle and audit-only outcome status."""
    spec = spec or LifecycleSpec()
    spec.validate()
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim != 1:
        raise RealDataError("raw_values must be one-dimensional.")
    if len(values) != len(months):
        raise RealDataError("raw_values and months must have equal length.")
    if not 0 <= train_end_index < len(values):
        raise RealDataError("train_end_index is outside the monthly range.")
    if not train_end_index <= test_end_index < len(values):
        raise RealDataError("test_end_index is outside the monthly range.")

    observed = np.isfinite(values)
    train_observed = np.flatnonzero(observed[: train_end_index + 1])
    full_observed = np.flatnonzero(observed[: test_end_index + 1])

    first_train: int | None = None
    last_train: int | None = None
    train_span = 0
    train_missing_ratio = 1.0
    train_trailing = train_end_index + 1
    if len(train_observed):
        first_train = int(train_observed[0])
        last_train = int(train_observed[-1])
        train_span = last_train - first_train + 1
        train_missing_ratio = 1.0 - len(train_observed) / max(train_span, 1)
        train_trailing = train_end_index - last_train

    if not len(train_observed):
        lifecycle = "not_observed_by_train"
    elif train_trailing > spec.inactive_gap_months:
        lifecycle = "inactive_at_train_end"
    elif train_span < spec.recent_launch_months:
        lifecycle = "new_launch_at_train_end"
    elif train_missing_ratio > spec.intermittent_missing_ratio:
        lifecycle = "intermittent_at_train_end"
    else:
        lifecycle = "established_at_train_end"

    first_full: int | None = None
    last_full: int | None = None
    full_span = 0
    full_missing_ratio = 1.0
    test_trailing = test_end_index + 1
    if len(full_observed):
        first_full = int(full_observed[0])
        last_full = int(full_observed[-1])
        full_span = last_full - first_full + 1
        full_missing_ratio = 1.0 - len(full_observed) / max(full_span, 1)
        test_trailing = test_end_index - last_full

    if not len(full_observed):
        outcome = "never_observed"
    elif first_full is not None and first_full > train_end_index:
        outcome = "launched_after_train"
    elif test_trailing > spec.inactive_gap_months:
        outcome = "inactive_by_test_end"
    else:
        outcome = "active_at_test_end"

    return {
        "lifecycle_at_train_end": lifecycle,
        "outcome_status_at_test_end": outcome,
        "observed_train_months": int(len(train_observed)),
        "first_train_observed_month": _month_or_empty(months, first_train),
        "last_train_observed_month": _month_or_empty(months, last_train),
        "train_span_months": int(train_span),
        "train_missing_ratio": round(float(train_missing_ratio), 6),
        "train_trailing_missing_months": int(train_trailing),
        "observed_total_months": int(len(full_observed)),
        "first_observed_month": _month_or_empty(months, first_full),
        "last_observed_month": _month_or_empty(months, last_full),
        "full_span_months": int(full_span),
        "full_missing_ratio": round(float(full_missing_ratio), 6),
        "test_trailing_missing_months": int(test_trailing),
        "cold_start_evaluable": bool(
            lifecycle in {"new_launch_at_train_end", "not_observed_by_train"}
            and len(full_observed) > len(train_observed)
        ),
    }
