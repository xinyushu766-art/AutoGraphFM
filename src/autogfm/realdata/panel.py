"""Leakage-safe monthly panel construction from canonical automotive CSVs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from autogfm.realdata.common import (
    RealDataError,
    read_csv_rows,
    sha256_file,
    stable_payload_id,
    write_csv_rows,
    write_json,
)
from autogfm.realdata.lifecycle import LifecycleSpec, profile_series_lifecycle


@dataclass(frozen=True)
class PanelSpec:
    """Controls series eligibility, imputation, features, and temporal splits."""

    train_end: date
    validation_end: date
    test_end: date
    start_month: date | None = None
    min_history_months: int = 24
    max_missing_ratio: float = 0.20
    max_trailing_missing_months: int = 3
    imputation: str = "linear"
    include_static_features: bool = True
    category_min_frequency: int = 2
    eligibility_basis: str = "train"
    lifecycle_recent_launch_months: int = 18
    lifecycle_inactive_gap_months: int = 3
    lifecycle_intermittent_missing_ratio: float = 0.25

    def validate(self) -> None:
        for name in ("train_end", "validation_end", "test_end", "start_month"):
            value = getattr(self, name)
            if value is not None and value.day != 1:
                raise RealDataError(f"{name} must be the first day of a month.")
        if not self.train_end < self.validation_end < self.test_end:
            raise RealDataError("Expected train_end < validation_end < test_end.")
        if self.start_month is not None and self.start_month > self.train_end:
            raise RealDataError("start_month cannot follow train_end.")
        if self.min_history_months < 2:
            raise RealDataError("min_history_months must be at least 2.")
        if not 0.0 <= self.max_missing_ratio < 1.0:
            raise RealDataError("max_missing_ratio must be in [0, 1).")
        if self.max_trailing_missing_months < 0:
            raise RealDataError("max_trailing_missing_months cannot be negative.")
        if self.imputation not in {"linear", "ffill", "zero"}:
            raise RealDataError("imputation must be linear, ffill, or zero.")
        if self.category_min_frequency < 1:
            raise RealDataError("category_min_frequency must be positive.")
        if self.eligibility_basis not in {"train", "full"}:
            raise RealDataError("eligibility_basis must be train or full.")
        LifecycleSpec(
            recent_launch_months=self.lifecycle_recent_launch_months,
            inactive_gap_months=self.lifecycle_inactive_gap_months,
            intermittent_missing_ratio=(
                self.lifecycle_intermittent_missing_ratio
            ),
        ).validate()


@dataclass(frozen=True)
class PanelBuildResult:
    output_dir: Path
    panel_id: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PanelData:
    sales: np.ndarray
    observed_mask: np.ndarray
    availability_mask: np.ndarray
    features: np.ndarray
    static_features: np.ndarray
    months: tuple[str, ...]
    series_keys: tuple[str, ...]
    feature_names: tuple[str, ...]
    static_feature_names: tuple[str, ...]
    catalog: tuple[dict[str, str], ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class WindowSplit:
    train_origins: tuple[int, ...]
    validation_origins: tuple[int, ...]
    test_origins: tuple[int, ...]


def _month_ordinal(value: date) -> int:
    return value.year * 12 + value.month - 1


def _ordinal_month(value: int) -> date:
    return date(value // 12, value % 12 + 1, 1)


def _month_range(start: date, end: date) -> list[date]:
    return [_ordinal_month(value) for value in range(_month_ordinal(start), _month_ordinal(end) + 1)]


def _parse_optional_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _months_between(left: date, right: date) -> int:
    return _month_ordinal(right) - _month_ordinal(left)


def _impute_series(values: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    observed = np.isfinite(values)
    if not observed.any():
        return np.zeros_like(values), np.zeros_like(values, dtype=np.float32)
    first = int(np.flatnonzero(observed)[0])
    last = int(np.flatnonzero(observed)[-1])
    availability = np.zeros_like(values, dtype=np.float32)
    availability[first : last + 1] = 1.0

    filled = values.astype(np.float64, copy=True)
    internal = np.arange(first, last + 1)
    if method == "zero":
        filled[internal] = np.nan_to_num(filled[internal], nan=0.0)
    elif method == "ffill":
        last_value = filled[first]
        for index in internal:
            if np.isfinite(filled[index]):
                last_value = filled[index]
            else:
                filled[index] = last_value
    else:
        known = np.flatnonzero(observed)
        filled[internal] = np.interp(internal, known, values[known])

    # Months outside the observed lifespan are structural zeros, not imputed demand.
    filled[:first] = 0.0
    filled[last + 1 :] = 0.0
    return filled.astype(np.float32), availability


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    for index in range(values.shape[0]):
        start = max(0, index - window + 1)
        output[index] = values[start : index + 1].mean(axis=0)
    return output


def _category_features(
    catalog: list[dict[str, str]],
    field: str,
    min_frequency: int,
) -> tuple[np.ndarray, list[str]]:
    values = [str(row.get(field, "") or "unknown").strip() or "unknown" for row in catalog]
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    kept = sorted(value for value, count in counts.items() if count >= min_frequency)
    if not kept:
        return np.zeros((len(values), 0), dtype=np.float32), []
    matrix = np.zeros((len(values), len(kept)), dtype=np.float32)
    lookup = {value: index for index, value in enumerate(kept)}
    for row_index, value in enumerate(values):
        if value in lookup:
            matrix[row_index, lookup[value]] = 1.0
    return matrix, [f"{field}={value}" for value in kept]


def build_monthly_panel(
    canonical_sales_csv: str | Path,
    canonical_catalog_csv: str | Path,
    output_dir: str | Path,
    spec: PanelSpec,
    *,
    overwrite: bool = False,
) -> PanelBuildResult:
    """Build a fixed-node monthly panel using train-only scaling statistics."""
    spec.validate()
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    _, sales_rows, _ = read_csv_rows(canonical_sales_csv)
    _, catalog_rows, _ = read_csv_rows(canonical_catalog_csv)
    if not sales_rows or not catalog_rows:
        raise RealDataError("Canonical sales and catalog must both be non-empty.")

    catalog_by_key = {row["series_key"]: row for row in catalog_rows}
    sales_by_series: dict[str, dict[date, float]] = {}
    for row in sales_rows:
        key = row["series_key"]
        month = date.fromisoformat(row["month"])
        if month > spec.test_end:
            continue
        if spec.start_month is not None and month < spec.start_month:
            continue
        sales_by_series.setdefault(key, {})[month] = float(row["sales"])
    if not sales_by_series:
        raise RealDataError("No sales rows remained inside the requested period.")

    first_month = spec.start_month or min(
        month for values in sales_by_series.values() for month in values
    )
    months = _month_range(first_month, spec.test_end)
    month_index = {month: index for index, month in enumerate(months)}
    train_index = month_index.get(spec.train_end)
    validation_index = month_index.get(spec.validation_end)
    test_index = month_index.get(spec.test_end)
    if train_index is None or validation_index is None or test_index is None:
        raise RealDataError("Split boundaries must lie inside the panel month range.")

    included_keys: list[str] = []
    exclusions: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    lifecycle_by_key: dict[str, dict[str, Any]] = {}
    raw_columns: list[np.ndarray] = []
    observed_columns: list[np.ndarray] = []
    availability_columns: list[np.ndarray] = []
    lifecycle_spec = LifecycleSpec(
        recent_launch_months=spec.lifecycle_recent_launch_months,
        inactive_gap_months=spec.lifecycle_inactive_gap_months,
        intermittent_missing_ratio=(
            spec.lifecycle_intermittent_missing_ratio
        ),
    )
    month_strings = [month.isoformat() for month in months]

    for key in sorted(sales_by_series):
        values = sales_by_series[key]
        raw = np.full(len(months), np.nan, dtype=np.float64)
        for month, sales in values.items():
            if month in month_index:
                raw[month_index[month]] = sales
        lifecycle = profile_series_lifecycle(
            raw,
            month_strings,
            train_index,
            test_index,
            lifecycle_spec,
        )
        catalog = catalog_by_key.get(key, {})
        lifecycle_row = {
            "series_key": key,
            "brand_name": catalog.get("brand_name", ""),
            "series_name": catalog.get("series_name", ""),
            **lifecycle,
        }
        lifecycle_rows.append(lifecycle_row)
        lifecycle_by_key[key] = lifecycle

        full_observed = np.flatnonzero(np.isfinite(raw))
        train_observed = np.flatnonzero(np.isfinite(raw[: train_index + 1]))
        eligibility_observed = (
            train_observed
            if spec.eligibility_basis == "train"
            else full_observed
        )
        eligibility_end = (
            train_index if spec.eligibility_basis == "train" else test_index
        )
        reasons: list[str] = []
        if len(eligibility_observed) < spec.min_history_months:
            reasons.append("insufficient_observed_history")
        if len(eligibility_observed):
            first_observed = int(eligibility_observed[0])
            last_observed = int(eligibility_observed[-1])
            span = last_observed - first_observed + 1
            missing_ratio = 1.0 - len(eligibility_observed) / max(span, 1)
            trailing_missing = eligibility_end - last_observed
            if missing_ratio > spec.max_missing_ratio:
                reasons.append("excessive_internal_missingness")
            if trailing_missing > spec.max_trailing_missing_months:
                reasons.append("stale_series")
        else:
            missing_ratio = 1.0
            trailing_missing = eligibility_end + 1
            reasons.append("no_observations")
        if reasons:
            exclusions.append(
                {
                    "series_key": key,
                    "brand_name": catalog.get("brand_name", ""),
                    "series_name": catalog.get("series_name", ""),
                    "observed_months": len(eligibility_observed),
                    "missing_ratio": round(float(missing_ratio), 6),
                    "trailing_missing_months": int(trailing_missing),
                    "eligibility_basis": spec.eligibility_basis,
                    "lifecycle_at_train_end": lifecycle[
                        "lifecycle_at_train_end"
                    ],
                    "reasons": "|".join(reasons),
                }
            )
            continue
        filled, availability = _impute_series(raw, spec.imputation)
        included_keys.append(key)
        raw_columns.append(filled)
        observed_columns.append(np.isfinite(raw).astype(np.float32))
        availability_columns.append(availability)

    if len(included_keys) < 2:
        raise RealDataError(
            "Fewer than two series passed panel filters; graph forecasting is not viable."
        )

    sales = np.stack(raw_columns, axis=1).astype(np.float32)
    observed_mask = np.stack(observed_columns, axis=1).astype(np.float32)
    availability_mask = np.stack(availability_columns, axis=1).astype(np.float32)
    selected_catalog = [
        {
            **catalog_by_key[key],
            "lifecycle_at_train_end": lifecycle_by_key[key][
                "lifecycle_at_train_end"
            ],
        }
        for key in included_keys
    ]

    log_sales = np.log1p(np.clip(sales, 0.0, None)).astype(np.float32)
    rolling_3 = np.log1p(_rolling_mean(sales, 3)).astype(np.float32)
    rolling_6 = np.log1p(_rolling_mean(sales, 6)).astype(np.float32)
    momentum_1 = np.zeros_like(log_sales)
    momentum_1[1:] = log_sales[1:] - log_sales[:-1]
    month_numbers = np.array([month.month for month in months], dtype=np.float32)
    month_sin = np.sin(2.0 * math.pi * (month_numbers - 1.0) / 12.0)
    month_cos = np.cos(2.0 * math.pi * (month_numbers - 1.0) / 12.0)
    month_sin = np.repeat(month_sin[:, None], len(included_keys), axis=1).astype(np.float32)
    month_cos = np.repeat(month_cos[:, None], len(included_keys), axis=1).astype(np.float32)

    dynamic = [
        log_sales,
        rolling_3,
        rolling_6,
        momentum_1,
        month_sin,
        month_cos,
        observed_mask,
        availability_mask,
    ]
    feature_names = [
        "log_sales",
        "log_rolling_mean_3",
        "log_rolling_mean_6",
        "log_momentum_1",
        "month_sin",
        "month_cos",
        "observed_mask",
        "availability_mask",
    ]

    price_mid = []
    for row in selected_catalog:
        low = _parse_optional_float(row.get("min_price", ""))
        high = _parse_optional_float(row.get("max_price", ""))
        if low is None and high is None:
            price_mid.append(np.nan)
        elif low is None:
            price_mid.append(high)
        elif high is None:
            price_mid.append(low)
        else:
            price_mid.append((low + high) / 2.0)
    price_array = np.array(price_mid, dtype=np.float64)
    finite_price = np.isfinite(price_array)
    price_median = float(np.nanmedian(price_array)) if finite_price.any() else 0.0
    price_array = np.where(finite_price, price_array, price_median)
    price_std = float(price_array.std())
    if price_std < 1e-8:
        price_std = 1.0
    price_z = ((price_array - price_array.mean()) / price_std).astype(np.float32)[:, None]
    static_blocks = [price_z]
    static_feature_names = ["price_mid_z"]
    if spec.include_static_features:
        for field in (
            "brand_name",
            "segment",
            "body_type",
            "energy_type",
            "lifecycle_at_train_end",
        ):
            block, names = _category_features(
                selected_catalog, field, spec.category_min_frequency
            )
            static_blocks.append(block)
            static_feature_names.extend(names)
    static_features = np.concatenate(static_blocks, axis=1).astype(np.float32)
    repeated_static = np.repeat(static_features[None, :, :], len(months), axis=0)
    features = np.concatenate(
        [np.stack(dynamic, axis=-1), repeated_static], axis=-1
    ).astype(np.float32)
    feature_names.extend(static_feature_names)

    # Only non-binary dynamic features are standardized, using train months only.
    standardized_indices = [1, 2, 3]
    scaling: dict[str, dict[str, float]] = {}
    for index in standardized_indices:
        train_values = features[: train_index + 1, :, index]
        mean = float(train_values.mean())
        std = float(train_values.std())
        if std < 1e-8:
            std = 1.0
        features[:, :, index] = (features[:, :, index] - mean) / std
        scaling[feature_names[index]] = {"mean": mean, "std": std}

    npz_path = destination / "panel.npz"
    np.savez_compressed(
        npz_path,
        sales=sales,
        observed_mask=observed_mask,
        availability_mask=availability_mask,
        features=features,
        static_features=static_features,
        months=np.array([month.isoformat() for month in months]),
        series_keys=np.array(included_keys),
        feature_names=np.array(feature_names),
        static_feature_names=np.array(static_feature_names),
    )
    write_csv_rows(
        destination / "catalog.csv",
        list(selected_catalog[0].keys()),
        selected_catalog,
    )
    write_csv_rows(
        destination / "exclusions.csv",
        [
            "series_key", "brand_name", "series_name", "observed_months",
            "missing_ratio", "trailing_missing_months", "eligibility_basis",
            "lifecycle_at_train_end", "reasons",
        ],
        exclusions,
    )
    write_csv_rows(
        destination / "lifecycle.csv",
        list(lifecycle_rows[0].keys()),
        lifecycle_rows,
    )
    write_csv_rows(
        destination / "months.csv",
        ["month_index", "month", "split"],
        [
            {
                "month_index": index,
                "month": month.isoformat(),
                "split": (
                    "train" if index <= train_index else
                    "validation" if index <= validation_index else "test"
                ),
            }
            for index, month in enumerate(months)
        ],
    )

    identity = {
        "schema_version": "autogfm-monthly-panel-v3",
        "canonical_sales_sha256": sha256_file(canonical_sales_csv),
        "canonical_catalog_sha256": sha256_file(canonical_catalog_csv),
        "spec": asdict(spec),
        "panel_sha256": sha256_file(npz_path),
        "series_keys": included_keys,
        "months": [month.isoformat() for month in months],
        "feature_names": feature_names,
    }
    panel_id = stable_payload_id(identity)
    manifest = {
        **identity,
        "panel_id": panel_id,
        "shape": {
            "months": len(months),
            "series": len(included_keys),
            "features": features.shape[-1],
            "static_features": static_features.shape[-1],
        },
        "boundaries": {
            "train_end": spec.train_end.isoformat(),
            "validation_end": spec.validation_end.isoformat(),
            "test_end": spec.test_end.isoformat(),
            "train_end_index": train_index,
            "validation_end_index": validation_index,
            "test_end_index": test_index,
        },
        "quality": {
            "included_series": len(included_keys),
            "excluded_series": len(exclusions),
            "eligibility_basis": spec.eligibility_basis,
            "observed_ratio": float(observed_mask.mean()),
            "availability_ratio": float(availability_mask.mean()),
            "imputed_cells": int(((availability_mask == 1) & (observed_mask == 0)).sum()),
        },
        "lifecycle": {
            "train_time_counts": {
                label: sum(
                    row["lifecycle_at_train_end"] == label
                    for row in lifecycle_rows
                )
                for label in sorted(
                    {row["lifecycle_at_train_end"] for row in lifecycle_rows}
                )
            },
            "outcome_counts": {
                label: sum(
                    row["outcome_status_at_test_end"] == label
                    for row in lifecycle_rows
                )
                for label in sorted(
                    {
                        row["outcome_status_at_test_end"]
                        for row in lifecycle_rows
                    }
                )
            },
            "cold_start_evaluable": sum(
                bool(row["cold_start_evaluable"]) for row in lifecycle_rows
            ),
            "feature_cutoff": spec.train_end.isoformat(),
            "outcome_cutoff": spec.test_end.isoformat(),
        },
        "scaling": scaling,
        "files": {
            "panel": {"path": npz_path.name, "sha256": sha256_file(npz_path)},
            "catalog": {"path": "catalog.csv", "sha256": sha256_file(destination / "catalog.csv")},
            "exclusions": {
                "path": "exclusions.csv",
                "sha256": sha256_file(destination / "exclusions.csv"),
            },
            "lifecycle": {
                "path": "lifecycle.csv",
                "sha256": sha256_file(destination / "lifecycle.csv"),
            },
        },
    }
    write_json(destination / "manifest.json", manifest)
    return PanelBuildResult(destination, panel_id, manifest)


def load_panel(panel_dir: str | Path) -> PanelData:
    directory = Path(panel_dir)
    arrays = np.load(directory / "panel.npz", allow_pickle=False)
    import json

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _, catalog_rows, _ = read_csv_rows(directory / "catalog.csv")
    return PanelData(
        sales=arrays["sales"].astype(np.float32),
        observed_mask=arrays["observed_mask"].astype(np.float32),
        availability_mask=arrays["availability_mask"].astype(np.float32),
        features=arrays["features"].astype(np.float32),
        static_features=arrays["static_features"].astype(np.float32),
        months=tuple(str(value) for value in arrays["months"].tolist()),
        series_keys=tuple(str(value) for value in arrays["series_keys"].tolist()),
        feature_names=tuple(str(value) for value in arrays["feature_names"].tolist()),
        static_feature_names=tuple(
            str(value) for value in arrays["static_feature_names"].tolist()
        ),
        catalog=tuple(catalog_rows),
        manifest=manifest,
    )


def build_window_split(
    panel: PanelData,
    history: int,
    horizon: int,
) -> WindowSplit:
    """Assign forecast origins by target end, preventing split leakage."""
    if history < 1 or horizon < 1:
        raise RealDataError("history and horizon must be positive.")
    boundaries = panel.manifest["boundaries"]
    train_end = int(boundaries["train_end_index"])
    validation_end = int(boundaries["validation_end_index"])
    test_end = int(boundaries["test_end_index"])
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    # origin is the last observed input month; targets are origin+1..origin+horizon.
    for origin in range(history - 1, len(panel.months) - horizon):
        target_end = origin + horizon
        if target_end <= train_end:
            train.append(origin)
        elif target_end <= validation_end:
            validation.append(origin)
        elif target_end <= test_end:
            test.append(origin)
    if not train or not validation or not test:
        raise RealDataError(
            "Panel boundaries do not produce non-empty train/validation/test windows."
        )
    return WindowSplit(tuple(train), tuple(validation), tuple(test))
