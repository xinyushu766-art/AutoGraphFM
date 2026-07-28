"""Audited CSV ingestion for monthly vehicle-series sales."""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from autogfm.database.repositories import (
    CatalogRepository,
    IngestionRepository,
    MonthlyRepository,
)
from autogfm.schemas import (
    BrandCreate,
    DataQualityResultCreate,
    IngestionRunCreate,
    MonthlySalesCreate,
    VehicleAliasCreate,
    VehicleSeriesCreate,
)

_CANONICAL_COLUMNS = {
    "brand",
    "canonical_series",
    "series",
    "month",
    "sales",
    "sales_rank",
    "country",
    "segment",
    "body_type",
    "energy_type",
    "status",
}

_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "brand": (
        "brand",
        "brand_name",
        "manufacturer",
        "品牌",
        "品牌名称",
        "厂商",
    ),
    "canonical_series": (
        "canonical_series",
        "canonical_series_name",
        "mapped_series",
        "standard_series",
        "标准车系",
        "标准车型",
        "映射车系",
    ),
    "series": (
        "series",
        "series_name",
        "vehicle_series",
        "model",
        "model_name",
        "车系",
        "车型",
        "车型名称",
    ),
    "month": (
        "month",
        "year_month",
        "sale_month",
        "sales_month",
        "date",
        "月份",
        "年月",
        "销售月份",
        "日期",
    ),
    "sales": (
        "sales",
        "monthly_sales",
        "sales_volume",
        "volume",
        "销量",
        "月销量",
        "销量数据",
    ),
    "sales_rank": (
        "sales_rank",
        "rank",
        "ranking",
        "销量排名",
        "排名",
    ),
    "country": ("country", "country_region", "国家", "国家地区"),
    "segment": ("segment", "market_segment", "级别", "细分市场"),
    "body_type": ("body_type", "body", "车身形式", "车身类型"),
    "energy_type": (
        "energy_type",
        "fuel_type",
        "power_type",
        "能源类型",
        "燃料类型",
        "动力类型",
    ),
    "status": ("status", "series_status", "状态", "在售状态"),
}

_STATUS_MAP = {
    "active": "active",
    "on_sale": "active",
    "onsale": "active",
    "在售": "active",
    "discontinued": "discontinued",
    "停售": "discontinued",
    "停产": "discontinued",
    "upcoming": "upcoming",
    "预售": "upcoming",
    "即将上市": "upcoming",
    "unknown": "unknown",
    "未知": "unknown",
}


class CsvImportError(ValueError):
    """Raised when the CSV-level contract cannot be satisfied."""


class DuplicateBusinessKeyError(CsvImportError):
    """Raised when an input file repeats one immutable business key."""


@dataclass(frozen=True)
class CsvImportConfig:
    """Runtime settings for one monthly-sales CSV import."""

    source: str
    data_version: str = "v1"
    encoding: str = "utf-8-sig"
    delimiter: str = ","
    default_brand: str | None = None
    collected_at: datetime | None = None
    code_version: str | None = None
    column_overrides: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source cannot be empty.")
        if not self.data_version.strip():
            raise ValueError("data_version cannot be empty.")
        unknown = set(self.column_overrides) - _CANONICAL_COLUMNS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown canonical column overrides: {names}")


@dataclass(frozen=True)
class RejectedRow:
    """One row rejected before it could become a monthly-sales fact."""

    row_number: int
    check_name: str
    message: str
    raw: Mapping[str, str]


@dataclass(frozen=True)
class ImportReport:
    """Stable, serializable summary of an audited import."""

    run_id: UUID
    status: str
    records_read: int
    records_written: int
    records_skipped: int
    records_rejected: int
    source_file_sha256: str
    resolved_columns: Mapping[str, str]
    rejected_rows: tuple[RejectedRow, ...]


def _normalize_header(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)


def _resolve_columns(
    fieldnames: list[str],
    config: CsvImportConfig,
) -> dict[str, str]:
    if not fieldnames:
        raise CsvImportError("CSV has no header row.")

    normalized_to_original: dict[str, list[str]] = {}
    for original in fieldnames:
        key = _normalize_header(original)
        normalized_to_original.setdefault(key, []).append(original)

    resolved: dict[str, str] = {}
    for canonical, override in config.column_overrides.items():
        if override not in fieldnames:
            raise CsvImportError(
                f"Override for {canonical!r} references missing column "
                f"{override!r}."
            )
        resolved[canonical] = override

    for canonical, aliases in _HEADER_ALIASES.items():
        if canonical in resolved:
            continue
        candidates: list[str] = []
        for alias in aliases:
            candidates.extend(
                normalized_to_original.get(_normalize_header(alias), [])
            )
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) > 1:
            joined = ", ".join(repr(item) for item in candidates)
            raise CsvImportError(
                f"Multiple columns match canonical field {canonical!r}: "
                f"{joined}. Use a column override."
            )
        if candidates:
            resolved[canonical] = candidates[0]

    required = {"series", "month", "sales"}
    if config.default_brand is None:
        required.add("brand")
    missing = sorted(required - set(resolved))
    if missing:
        available = ", ".join(fieldnames)
        raise CsvImportError(
            "Missing required canonical columns: "
            f"{', '.join(missing)}. Available columns: {available}"
        )
    return resolved


def _value(row: Mapping[str, str], columns: Mapping[str, str], name: str) -> str:
    column = columns.get(name)
    if column is None:
        return ""
    return (row.get(column) or "").strip()


def parse_month(value: str) -> date:
    """Parse common monthly formats and canonicalize to the first day."""
    text = unicodedata.normalize("NFKC", value).strip()
    if not text:
        raise ValueError("month is empty.")

    match = re.fullmatch(
        r"(\d{4})\s*年\s*(\d{1,2})\s*月"
        r"(?:\s*\d{1,2}\s*日)?",
        text,
    )
    if match is None:
        match = re.fullmatch(
            r"(\d{4})\s*[-/.]\s*(\d{1,2})"
            r"(?:\s*[-/.]\s*\d{1,2})?",
            text,
        )
    if match is None:
        raise ValueError(
            "month must look like YYYY-MM, YYYY/MM, YYYY年M月, or a date."
        )
    year = int(match.group(1))
    month = int(match.group(2))
    try:
        return date(year, month, 1)
    except ValueError as exc:
        raise ValueError(f"Invalid month value: {value!r}.") from exc


def parse_nonnegative_int(value: str, field_name: str) -> int:
    """Parse integers while rejecting fractional and negative values."""
    text = unicodedata.normalize("NFKC", value).strip().replace(",", "")
    if not text:
        raise ValueError(f"{field_name} is empty.")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{field_name} is not numeric: {value!r}.") from exc
    if number != number.to_integral_value():
        raise ValueError(f"{field_name} must be an integer: {value!r}.")
    integer = int(number)
    if integer < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return integer


def _parse_optional_positive_int(value: str, field_name: str) -> int | None:
    if not value:
        return None
    integer = parse_nonnegative_int(value, field_name)
    if integer <= 0:
        raise ValueError(f"{field_name} must be positive when provided.")
    return integer


def _parse_status(value: str) -> str:
    if not value:
        return "active"
    key = unicodedata.normalize("NFKC", value).strip().casefold()
    key = re.sub(r"[\s\-_/]+", "", key)
    status = _STATUS_MAP.get(key)
    if status is None:
        raise ValueError(
            "status must map to active, discontinued, upcoming, or unknown."
        )
    return status


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, ValidationError):
        messages = []
        for error in exc.errors(include_url=False):
            location = ".".join(str(item) for item in error["loc"])
            messages.append(f"{location}: {error['msg']}")
        return "schema_validation", "; ".join(messages)
    if isinstance(exc, IntegrityError):
        return "database_integrity", str(exc.orig)
    if isinstance(exc, DuplicateBusinessKeyError):
        return "duplicate_business_key", str(exc)
    if isinstance(exc, CsvImportError):
        return "csv_contract", str(exc)
    return "row_contract", str(exc)


def _final_status(written: int, skipped: int, rejected: int) -> str:
    if rejected == 0:
        return "succeeded"
    if written + skipped > 0:
        return "partial"
    return "failed"


def import_monthly_sales_csv(
    session: Session,
    csv_path: str | Path,
    config: CsvImportConfig,
) -> ImportReport:
    """Import a CSV with row-level rejection and persistent audit metadata."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    source_hash = _sha256_file(path)
    now = datetime.now(UTC)
    collected_at = config.collected_at or now
    ingestion = IngestionRepository(session)
    run = ingestion.create_run(
        IngestionRunCreate(
            source=config.source,
            started_at=now,
            code_version=config.code_version,
            run_metadata={
                "input_file": path.name,
                "input_file_sha256": source_hash,
                "data_version": config.data_version,
            },
        )
    )

    read = 0
    written = 0
    skipped = 0
    rejected: list[RejectedRow] = []
    resolved_columns: dict[str, str] = {}
    duplicate_rows = 0
    seen_keys: set[tuple[UUID, date, str, str]] = set()

    try:
        with path.open(
            "r",
            encoding=config.encoding,
            newline="",
        ) as handle:
            reader = csv.DictReader(handle, delimiter=config.delimiter)
            resolved_columns = _resolve_columns(
                list(reader.fieldnames or []),
                config,
            )

            catalog = CatalogRepository(session)
            monthly = MonthlyRepository(session)

            for row_number, raw_row in enumerate(reader, start=2):
                read += 1
                row = {key: value or "" for key, value in raw_row.items()}
                try:
                    with session.begin_nested():
                        brand_name = (
                            _value(row, resolved_columns, "brand")
                            or config.default_brand
                            or ""
                        )
                        if not brand_name:
                            raise ValueError("brand is empty.")

                        source_series = _value(
                            row,
                            resolved_columns,
                            "series",
                        )
                        if not source_series:
                            raise ValueError("series is empty.")
                        canonical_series = (
                            _value(
                                row,
                                resolved_columns,
                                "canonical_series",
                            )
                            or source_series
                        )

                        brand = catalog.get_or_create_brand(
                            BrandCreate(
                                brand_name=brand_name,
                                country=(
                                    _value(row, resolved_columns, "country")
                                    or None
                                ),
                            )
                        )

                        resolved_series = catalog.resolve_alias(
                            config.source,
                            source_series,
                        )
                        if resolved_series is not None:
                            if resolved_series.brand_id != brand.id:
                                raise ValueError(
                                    "Source alias resolves to a different brand."
                                )
                            series = resolved_series
                        else:
                            series = catalog.get_or_create_series(
                                VehicleSeriesCreate(
                                    brand_id=brand.id,
                                    series_name=canonical_series,
                                    segment=(
                                        _value(
                                            row,
                                            resolved_columns,
                                            "segment",
                                        )
                                        or None
                                    ),
                                    body_type=(
                                        _value(
                                            row,
                                            resolved_columns,
                                            "body_type",
                                        )
                                        or None
                                    ),
                                    energy_type=(
                                        _value(
                                            row,
                                            resolved_columns,
                                            "energy_type",
                                        )
                                        or "unknown"
                                    ),
                                    status=_parse_status(
                                        _value(
                                            row,
                                            resolved_columns,
                                            "status",
                                        )
                                    ),
                                )
                            )
                            catalog.add_alias(
                                VehicleAliasCreate(
                                    source=config.source,
                                    source_name=source_series,
                                    series_id=series.id,
                                    match_method="exact",
                                    match_score=1.0,
                                    review_status="approved",
                                )
                            )

                        month = parse_month(
                            _value(row, resolved_columns, "month")
                        )
                        sales = parse_nonnegative_int(
                            _value(row, resolved_columns, "sales"),
                            "sales",
                        )
                        sales_rank = _parse_optional_positive_int(
                            _value(row, resolved_columns, "sales_rank"),
                            "sales_rank",
                        )
                        business_key = (
                            series.id,
                            month,
                            config.source,
                            config.data_version,
                        )
                        if business_key in seen_keys:
                            duplicate_rows += 1
                            raise DuplicateBusinessKeyError(
                                "Duplicate series/month/source/data_version "
                                "inside the input file."
                            )

                        existing = monthly.get_sales_record(
                            series_id=series.id,
                            month=month,
                            source=config.source,
                            data_version=config.data_version,
                        )
                        if existing is None:
                            monthly.add_sales(
                                MonthlySalesCreate(
                                    series_id=series.id,
                                    run_id=run.id,
                                    month=month,
                                    sales=sales,
                                    sales_rank=sales_rank,
                                    source=config.source,
                                    collected_at=collected_at,
                                    data_version=config.data_version,
                                )
                            )
                            written += 1
                        else:
                            skipped += 1
                        seen_keys.add(business_key)
                except (ValueError, IntegrityError) as exc:
                    check_name, message = _serialize_error(exc)
                    rejected.append(
                        RejectedRow(
                            row_number=row_number,
                            check_name=check_name,
                            message=message,
                            raw=dict(row),
                        )
                    )
    except (CsvImportError, UnicodeError, csv.Error) as exc:
        check_name, message = _serialize_error(exc)
        rejected.append(
            RejectedRow(
                row_number=1,
                check_name=check_name,
                message=message,
                raw={},
            )
        )

    status = _final_status(written, skipped, len(rejected))
    finished_at = datetime.now(UTC)
    run.finished_at = finished_at
    run.status = status
    run.records_read = read
    run.records_written = written
    run.records_rejected = len(rejected)
    run.error_message = rejected[0].message if status == "failed" else None
    run.run_metadata = {
        **run.run_metadata,
        "resolved_columns": resolved_columns,
        "records_skipped": skipped,
    }

    ingestion.add_quality_result(
        DataQualityResultCreate(
            run_id=run.id,
            check_name="csv_contract",
            table_name="monthly_sales",
            severity="error",
            passed=not any(item.row_number == 1 for item in rejected),
            failed_rows=sum(item.row_number == 1 for item in rejected),
            details={"resolved_columns": resolved_columns},
        )
    )
    ingestion.add_quality_result(
        DataQualityResultCreate(
            run_id=run.id,
            check_name="row_contracts",
            table_name="monthly_sales",
            severity="error",
            passed=not rejected,
            failed_rows=len(rejected),
            details={
                "examples": [
                    {
                        "row_number": item.row_number,
                        "check_name": item.check_name,
                        "message": item.message,
                    }
                    for item in rejected[:20]
                ]
            },
        )
    )
    ingestion.add_quality_result(
        DataQualityResultCreate(
            run_id=run.id,
            check_name="duplicate_business_keys_in_file",
            table_name="monthly_sales",
            severity="error",
            passed=duplicate_rows == 0,
            failed_rows=duplicate_rows,
            details={},
        )
    )
    ingestion.add_quality_result(
        DataQualityResultCreate(
            run_id=run.id,
            check_name="idempotent_existing_rows",
            table_name="monthly_sales",
            severity="info",
            passed=True,
            failed_rows=0,
            details={"records_skipped": skipped},
        )
    )
    session.flush()

    return ImportReport(
        run_id=run.id,
        status=status,
        records_read=read,
        records_written=written,
        records_skipped=skipped,
        records_rejected=len(rejected),
        source_file_sha256=source_hash,
        resolved_columns=dict(resolved_columns),
        rejected_rows=tuple(rejected),
    )
