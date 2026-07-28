"""Deterministic train/validation/test snapshots for monthly sales."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from autogfm.database.models import Brand, MonthlySales, VehicleSeries

_SNAPSHOT_COLUMNS = (
    "series_id",
    "brand_name",
    "series_name",
    "segment",
    "body_type",
    "energy_type",
    "status",
    "month",
    "sales",
    "sales_rank",
    "source",
    "data_version",
)


class SnapshotError(ValueError):
    """Raised when snapshot constraints cannot be satisfied."""


@dataclass(frozen=True)
class SnapshotSpec:
    """Immutable definition of a leakage-safe monthly-sales snapshot."""

    source: str
    data_version: str
    train_end: date
    validation_end: date
    test_end: date | None = None
    start_month: date | None = None
    min_history_months: int = 1
    require_contiguous_history: bool = False
    output_format: Literal["csv", "parquet"] = "csv"

    def __post_init__(self) -> None:
        for name in (
            "train_end",
            "validation_end",
            "test_end",
            "start_month",
        ):
            value = getattr(self, name)
            if value is not None and value.day != 1:
                raise SnapshotError(f"{name} must be the first day of a month.")
        if self.train_end >= self.validation_end:
            raise SnapshotError("train_end must precede validation_end.")
        if self.test_end is not None and self.validation_end >= self.test_end:
            raise SnapshotError("validation_end must precede test_end.")
        if self.start_month is not None and self.start_month > self.train_end:
            raise SnapshotError("start_month cannot follow train_end.")
        if self.min_history_months < 1:
            raise SnapshotError("min_history_months must be at least 1.")
        if not self.source.strip():
            raise SnapshotError("source cannot be empty.")
        if not self.data_version.strip():
            raise SnapshotError("data_version cannot be empty.")


@dataclass(frozen=True)
class SnapshotResult:
    """Created snapshot location and manifest payload."""

    output_dir: Path
    snapshot_id: str
    manifest: dict[str, Any]


def _month_ordinal(value: date) -> int:
    return value.year * 12 + value.month


def _is_contiguous(months: list[date]) -> bool:
    return all(
        _month_ordinal(current) - _month_ordinal(previous) == 1
        for previous, current in zip(months, months[1:], strict=False)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _serialize_row(
    sales: MonthlySales,
    series: VehicleSeries,
    brand: Brand,
) -> dict[str, Any]:
    return {
        "series_id": str(series.id),
        "brand_name": brand.brand_name,
        "series_name": series.series_name,
        "segment": series.segment or "",
        "body_type": series.body_type or "",
        "energy_type": series.energy_type,
        "status": series.status,
        "month": sales.month.isoformat(),
        "sales": sales.sales,
        "sales_rank": sales.sales_rank if sales.sales_rank is not None else "",
        "source": sales.source,
        "data_version": sales.data_version,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_SNAPSHOT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SnapshotError(
            "Parquet output requires the optional dependency: "
            "pip install 'autogfm[snapshot]'. No CSV fallback was used."
        ) from exc

    schema = pa.schema(
        [
            ("series_id", pa.string()),
            ("brand_name", pa.string()),
            ("series_name", pa.string()),
            ("segment", pa.string()),
            ("body_type", pa.string()),
            ("energy_type", pa.string()),
            ("status", pa.string()),
            ("month", pa.date32()),
            ("sales", pa.int64()),
            ("sales_rank", pa.int64()),
            ("source", pa.string()),
            ("data_version", pa.string()),
        ]
    )
    parquet_rows = []
    for row in rows:
        parquet_rows.append(
            {
                **row,
                "month": date.fromisoformat(str(row["month"])),
                "sales_rank": row["sales_rank"] or None,
            }
        )
    table = pa.Table.from_pylist(parquet_rows, schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def _month_range(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    if not rows:
        return {"min": None, "max": None}
    months = [str(row["month"]) for row in rows]
    return {"min": min(months), "max": max(months)}


def _split_rows(
    rows: list[dict[str, Any]],
    spec: SnapshotSpec,
) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "validation": [], "test": []}
    for row in rows:
        month = date.fromisoformat(str(row["month"]))
        if month <= spec.train_end:
            splits["train"].append(row)
        elif month <= spec.validation_end:
            splits["validation"].append(row)
        elif spec.test_end is None or month <= spec.test_end:
            splits["test"].append(row)
    return splits


def _filter_history(
    rows: list[dict[str, Any]],
    spec: SnapshotSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_series: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_series.setdefault(str(row["series_id"]), []).append(row)

    included: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for series_id, series_rows in sorted(by_series.items()):
        months = sorted(
            date.fromisoformat(str(row["month"])) for row in series_rows
        )
        reasons = []
        if len(months) < spec.min_history_months:
            reasons.append("insufficient_history")
        if spec.require_contiguous_history and not _is_contiguous(months):
            reasons.append("non_contiguous_history")
        if reasons:
            exclusions.append(
                {
                    "series_id": series_id,
                    "series_name": series_rows[0]["series_name"],
                    "history_months": len(months),
                    "reasons": reasons,
                }
            )
        else:
            included.extend(series_rows)
    return included, exclusions


def build_monthly_sales_snapshot(
    session: Session,
    output_dir: str | Path,
    spec: SnapshotSpec,
    *,
    overwrite: bool = False,
) -> SnapshotResult:
    """Build an immutable split snapshot and a content-addressed manifest."""
    statement = (
        select(MonthlySales, VehicleSeries, Brand)
        .join(VehicleSeries, MonthlySales.series_id == VehicleSeries.id)
        .join(Brand, VehicleSeries.brand_id == Brand.id)
        .where(
            MonthlySales.source == spec.source,
            MonthlySales.data_version == spec.data_version,
        )
        .order_by(
            MonthlySales.month,
            Brand.brand_name,
            VehicleSeries.series_name,
            VehicleSeries.id,
        )
    )
    if spec.start_month is not None:
        statement = statement.where(MonthlySales.month >= spec.start_month)
    if spec.test_end is not None:
        statement = statement.where(MonthlySales.month <= spec.test_end)

    raw_rows = [
        _serialize_row(sales, series, brand)
        for sales, series, brand in session.execute(statement)
    ]
    if not raw_rows:
        raise SnapshotError(
            "No monthly sales matched the requested source and data version."
        )

    filtered_rows, exclusions = _filter_history(raw_rows, spec)
    if not filtered_rows:
        raise SnapshotError(
            "All series were excluded by the history filters. "
            "Lower min_history_months or inspect continuity."
        )
    filtered_rows.sort(
        key=lambda row: (
            str(row["month"]),
            str(row["brand_name"]),
            str(row["series_name"]),
            str(row["series_id"]),
        )
    )
    splits = _split_rows(filtered_rows, spec)
    if not splits["train"]:
        raise SnapshotError("The train split is empty.")
    if not splits["validation"]:
        raise SnapshotError("The validation split is empty.")
    if not splits["test"]:
        raise SnapshotError("The test split is empty.")

    destination = Path(output_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Snapshot output already exists: {destination}. "
            "Use overwrite=True only for an intentional rebuild."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )

    try:
        suffix = ".csv" if spec.output_format == "csv" else ".parquet"
        files: dict[str, dict[str, Any]] = {}
        for split_name, split_rows in splits.items():
            filename = f"{split_name}{suffix}"
            path = temporary / filename
            if spec.output_format == "csv":
                _write_csv(path, split_rows)
            else:
                _write_parquet(path, split_rows)
            files[split_name] = {
                "path": filename,
                "sha256": _sha256_file(path),
                "rows": len(split_rows),
                "series": len(
                    {str(row["series_id"]) for row in split_rows}
                ),
                "month_range": _month_range(split_rows),
            }

        identity_payload = {
            "schema_version": "monthly-sales-snapshot-v1",
            "source": spec.source,
            "data_version": spec.data_version,
            "boundaries": {
                "start_month": (
                    spec.start_month.isoformat()
                    if spec.start_month is not None
                    else None
                ),
                "train_end": spec.train_end.isoformat(),
                "validation_end": spec.validation_end.isoformat(),
                "test_end": (
                    spec.test_end.isoformat()
                    if spec.test_end is not None
                    else None
                ),
            },
            "filters": {
                "min_history_months": spec.min_history_months,
                "require_contiguous_history": (
                    spec.require_contiguous_history
                ),
            },
            "output_format": spec.output_format,
            "files": files,
        }
        snapshot_id = hashlib.sha256(
            _canonical_json(identity_payload)
        ).hexdigest()[:16]
        manifest = {
            **identity_payload,
            "snapshot_id": snapshot_id,
            "created_at": datetime.now(UTC).isoformat(),
            "input_rows": len(raw_rows),
            "included_rows": len(filtered_rows),
            "included_series": len(
                {str(row["series_id"]) for row in filtered_rows}
            ),
            "excluded_series": exclusions,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )

        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return SnapshotResult(
        output_dir=destination,
        snapshot_id=snapshot_id,
        manifest=manifest,
    )
