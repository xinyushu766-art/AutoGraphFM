"""Adaptive three-file canonicalization for automotive sales projects."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from autogfm.realdata.common import (
    RealDataError,
    normalize_token,
    parse_float,
    parse_month,
    parse_nonnegative_int,
    read_csv_rows,
    sha256_file,
    stable_payload_id,
    stable_series_key,
    write_csv_rows,
    write_json,
)


_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "sales_series": (
        "series_name", "series", "model_name", "vehicle_series", "车型", "车系",
        "车系名称", "车型名称", "name", "汽车名称", "车名",
    ),
    "month": (
        "month", "date", "year_month", "stat_month", "月份", "年月", "时间",
        "销售月份", "统计月份", "月份日期",
    ),
    "sales": (
        "sales", "sales_volume", "monthly_sales", "销量", "月销量", "销量数据",
        "零售销量", "批发销量", "sales_count",
    ),
    "rank": ("rank", "sales_rank", "排名", "销量排名", "月排名"),
    "sales_brand": ("brand", "brand_name", "品牌", "品牌名称", "厂商品牌"),
    "mapping_source": (
        "source_series", "raw_series", "original_series", "source_name", "原始车系",
        "原车系", "源车系", "待映射车系", "series_name", "车系名称",
    ),
    "mapping_canonical": (
        "canonical_series", "mapped_series", "target_series", "standard_series",
        "标准车系", "统一车系", "映射车系", "canonical_name", "车型标准名",
    ),
    "mapping_brand": ("brand", "brand_name", "品牌", "品牌名称"),
    "vehicle_series": (
        "series_name", "series", "vehicle_series", "车型", "车系", "车系名称",
        "车型名称", "name",
    ),
    "vehicle_brand": ("brand", "brand_name", "品牌", "品牌名称", "厂商品牌"),
    "segment": ("segment", "level", "class", "级别", "车型级别", "细分市场"),
    "body_type": ("body_type", "body", "车身类型", "车型类别", "类别"),
    "energy_type": (
        "energy_type", "power_type", "fuel_type", "能源类型", "动力类型", "燃料类型",
        "能源",
    ),
    "min_price": (
        "min_price", "price_min", "最低价", "指导价最低", "厂商指导价最低",
        "最低指导价",
    ),
    "max_price": (
        "max_price", "price_max", "最高价", "指导价最高", "厂商指导价最高",
        "最高指导价",
    ),
    "status": ("status", "sale_status", "状态", "在售状态", "销售状态"),
    "launch_date": ("launch_date", "上市时间", "上市日期", "首次上市"),
}


@dataclass(frozen=True)
class ColumnResolution:
    field: str
    column: str | None
    method: str


@dataclass(frozen=True)
class CanonicalizationConfig:
    source_name: str = "automotive_csv"
    data_version: str = "v1"
    strict_unmatched: bool = False
    duplicate_policy: str = "sum"

    def validate(self) -> None:
        if self.duplicate_policy not in {"sum", "max", "error"}:
            raise RealDataError("duplicate_policy must be sum, max, or error.")
        if not self.source_name.strip():
            raise RealDataError("source_name cannot be empty.")
        if not self.data_version.strip():
            raise RealDataError("data_version cannot be empty.")


@dataclass(frozen=True)
class CanonicalizationResult:
    output_dir: Path
    dataset_id: str
    manifest: dict[str, Any]


def _normalized_aliases(field: str) -> set[str]:
    return {normalize_token(value) for value in _FIELD_ALIASES[field]}


def resolve_column(
    headers: Sequence[str],
    field: str,
    explicit: Mapping[str, str] | None = None,
    *,
    required: bool = True,
) -> ColumnResolution:
    """Resolve one logical field from an explicit map or alias dictionary."""
    explicit = explicit or {}
    if field in explicit:
        column = explicit[field]
        if column not in headers:
            raise RealDataError(
                f"Explicit column {column!r} for {field!r} is not in the CSV header."
            )
        return ColumnResolution(field, column, "explicit")

    normalized_headers = {normalize_token(header): header for header in headers}
    for alias in _FIELD_ALIASES[field]:
        match = normalized_headers.get(normalize_token(alias))
        if match is not None:
            return ColumnResolution(field, match, "alias")

    # Conservative scored partial matching is useful for headers such as 月销量(辆).
    aliases = _normalized_aliases(field)
    scored: list[tuple[float, str]] = []
    for header in headers:
        normalized = normalize_token(header)
        matches = [
            alias for alias in aliases
            if alias and (alias in normalized or normalized in alias)
        ]
        if matches:
            longest = max(matches, key=len)
            score = len(longest) / max(len(normalized), 1)
            # Sales-rank columns should never win the sales-value field.
            if field == "sales" and any(token in normalized for token in ("排名", "rank")):
                score -= 1.0
            scored.append((score, header))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if scored and (len(scored) == 1 or scored[0][0] > scored[1][0] + 1e-9):
        return ColumnResolution(field, scored[0][1], "partial_alias")
    if scored:
        candidates = [header for score, header in scored if abs(score - scored[0][0]) < 1e-9]
        raise RealDataError(
            f"Ambiguous columns for {field!r}: {candidates}. Use an explicit map."
        )
    if required:
        raise RealDataError(
            f"Could not resolve required field {field!r}. Headers: {list(headers)}"
        )
    return ColumnResolution(field, None, "missing_optional")


def _value(row: Mapping[str, str], resolution: ColumnResolution) -> str:
    if resolution.column is None:
        return ""
    return str(row.get(resolution.column, "") or "").strip()


def _canonicalize_status(raw: str) -> str:
    token = normalize_token(raw)
    if not token:
        return "unknown"
    if any(value in token for value in ("在售", "active", "onsale", "销售中")):
        return "active"
    if any(value in token for value in ("停售", "discontinued", "停产")):
        return "discontinued"
    if any(value in token for value in ("预售", "upcoming", "即将上市")):
        return "upcoming"
    return raw.strip()[:80]


def canonicalize_three_files(
    sales_csv: str | Path,
    mapping_csv: str | Path,
    vehicles_csv: str | Path,
    output_dir: str | Path,
    *,
    config: CanonicalizationConfig | None = None,
    sales_columns: Mapping[str, str] | None = None,
    mapping_columns: Mapping[str, str] | None = None,
    vehicle_columns: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> CanonicalizationResult:
    """Resolve, audit, and canonicalize the project's three real CSV files."""
    config = config or CanonicalizationConfig()
    config.validate()
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    sales_headers, sales_rows, sales_encoding = read_csv_rows(sales_csv)
    mapping_headers, mapping_rows, mapping_encoding = read_csv_rows(mapping_csv)
    vehicle_headers, vehicle_rows, vehicle_encoding = read_csv_rows(vehicles_csv)

    resolutions = {
        "sales": {
            name: resolve_column(
                sales_headers,
                field,
                sales_columns,
                required=name in {"series", "month", "sales"},
            )
            for name, field in {
                "series": "sales_series",
                "month": "month",
                "sales": "sales",
                "rank": "rank",
                "brand": "sales_brand",
            }.items()
        },
        "mapping": {
            name: resolve_column(
                mapping_headers,
                field,
                mapping_columns,
                required=name == "source",
            )
            for name, field in {
                "source": "mapping_source",
                "canonical": "mapping_canonical",
                "brand": "mapping_brand",
            }.items()
        },
        "vehicles": {
            name: resolve_column(
                vehicle_headers,
                field,
                vehicle_columns,
                required=name == "series",
            )
            for name, field in {
                "series": "vehicle_series",
                "brand": "vehicle_brand",
                "segment": "segment",
                "body_type": "body_type",
                "energy_type": "energy_type",
                "min_price": "min_price",
                "max_price": "max_price",
                "status": "status",
                "launch_date": "launch_date",
            }.items()
        },
    }

    mapping_lookup: dict[str, tuple[str, str]] = {}
    mapping_conflicts: list[dict[str, str]] = []
    for row in mapping_rows:
        source_name = _value(row, resolutions["mapping"]["source"])
        if not source_name:
            continue
        canonical_name = _value(row, resolutions["mapping"]["canonical"]) or source_name
        brand_name = _value(row, resolutions["mapping"]["brand"])
        key = normalize_token(source_name)
        value = (canonical_name.strip(), brand_name.strip())
        previous = mapping_lookup.get(key)
        if previous is not None and previous != value:
            mapping_conflicts.append(
                {
                    "source_series_name": source_name,
                    "first_mapping": " | ".join(previous),
                    "conflicting_mapping": " | ".join(value),
                }
            )
        else:
            mapping_lookup[key] = value

    catalog_by_key: dict[str, dict[str, Any]] = {}
    catalog_name_index: dict[str, list[str]] = defaultdict(list)
    rejected_vehicle_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(vehicle_rows, start=2):
        try:
            series_name = _value(row, resolutions["vehicles"]["series"])
            if not series_name:
                raise RealDataError("empty vehicle series")
            brand_name = _value(row, resolutions["vehicles"]["brand"])
            mapped = mapping_lookup.get(normalize_token(series_name))
            if mapped is not None:
                series_name = mapped[0]
                brand_name = brand_name or mapped[1]
            brand_name = brand_name or "unknown"
            key = stable_series_key(brand_name, series_name)
            min_price = parse_float(_value(row, resolutions["vehicles"]["min_price"]))
            max_price = parse_float(_value(row, resolutions["vehicles"]["max_price"]))
            if min_price is not None and max_price is not None and min_price > max_price:
                min_price, max_price = max_price, min_price
            launch_raw = _value(row, resolutions["vehicles"]["launch_date"])
            launch_date = parse_month(launch_raw).isoformat() if launch_raw else ""
            record = {
                "series_key": key,
                "brand_name": brand_name.strip(),
                "series_name": series_name.strip(),
                "segment": _value(row, resolutions["vehicles"]["segment"]),
                "body_type": _value(row, resolutions["vehicles"]["body_type"]),
                "energy_type": _value(row, resolutions["vehicles"]["energy_type"]) or "unknown",
                "min_price": "" if min_price is None else min_price,
                "max_price": "" if max_price is None else max_price,
                "status": _canonicalize_status(
                    _value(row, resolutions["vehicles"]["status"])
                ),
                "launch_month": launch_date,
            }
            if key in catalog_by_key:
                # Prefer non-empty values while retaining deterministic first-row authority.
                existing = catalog_by_key[key]
                for field, value in record.items():
                    if existing.get(field, "") in {"", "unknown"} and value not in {"", "unknown"}:
                        existing[field] = value
            else:
                catalog_by_key[key] = record
                catalog_name_index[normalize_token(series_name)].append(key)
        except Exception as exc:  # row-level quarantine is intentional
            rejected_vehicle_rows.append(
                {"row_number": row_number, "reason": str(exc), "raw_row": dict(row)}
            )

    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    unmatched: list[dict[str, Any]] = []
    rejected_sales_rows: list[dict[str, Any]] = []
    duplicate_count = 0
    for row_number, row in enumerate(sales_rows, start=2):
        try:
            raw_series = _value(row, resolutions["sales"]["series"])
            if not raw_series:
                raise RealDataError("empty sales series")
            raw_brand = _value(row, resolutions["sales"]["brand"])
            mapped = mapping_lookup.get(normalize_token(raw_series))
            canonical_name = mapped[0] if mapped else raw_series
            brand_name = raw_brand or (mapped[1] if mapped else "")

            candidate_keys = catalog_name_index.get(normalize_token(canonical_name), [])
            if brand_name:
                direct_key = stable_series_key(brand_name, canonical_name)
                if direct_key in catalog_by_key:
                    series_key = direct_key
                else:
                    series_key = direct_key
            elif len(candidate_keys) == 1:
                series_key = candidate_keys[0]
                brand_name = str(catalog_by_key[series_key]["brand_name"])
            else:
                brand_name = brand_name or "unknown"
                series_key = stable_series_key(brand_name, canonical_name)

            if series_key not in catalog_by_key:
                unmatched.append(
                    {
                        "row_number": row_number,
                        "raw_series_name": raw_series,
                        "canonical_series_name": canonical_name,
                        "brand_name": brand_name,
                        "series_key": series_key,
                        "reason": "not_found_in_vehicle_catalog",
                    }
                )
                if config.strict_unmatched:
                    continue
                catalog_by_key[series_key] = {
                    "series_key": series_key,
                    "brand_name": brand_name,
                    "series_name": canonical_name,
                    "segment": "",
                    "body_type": "",
                    "energy_type": "unknown",
                    "min_price": "",
                    "max_price": "",
                    "status": "unknown",
                    "launch_month": "",
                }
                catalog_name_index[normalize_token(canonical_name)].append(series_key)

            month = parse_month(_value(row, resolutions["sales"]["month"])).isoformat()
            sales = parse_nonnegative_int(_value(row, resolutions["sales"]["sales"]))
            rank_raw = _value(row, resolutions["sales"]["rank"])
            rank = parse_nonnegative_int(rank_raw) if rank_raw else ""
            business_key = (series_key, month)
            if business_key in aggregated:
                duplicate_count += 1
                if config.duplicate_policy == "error":
                    raise RealDataError(f"duplicate monthly key: {business_key}")
                if config.duplicate_policy == "sum":
                    aggregated[business_key]["sales"] += sales
                elif config.duplicate_policy == "max":
                    aggregated[business_key]["sales"] = max(
                        int(aggregated[business_key]["sales"]), sales
                    )
                if rank != "":
                    previous_rank = aggregated[business_key]["sales_rank"]
                    aggregated[business_key]["sales_rank"] = (
                        rank if previous_rank == "" else min(int(previous_rank), int(rank))
                    )
            else:
                catalog = catalog_by_key[series_key]
                aggregated[business_key] = {
                    "series_key": series_key,
                    "brand_name": catalog["brand_name"],
                    "series_name": catalog["series_name"],
                    "month": month,
                    "sales": sales,
                    "sales_rank": rank,
                    "source": config.source_name,
                    "data_version": config.data_version,
                }
        except Exception as exc:  # row-level quarantine is intentional
            rejected_sales_rows.append(
                {"row_number": row_number, "reason": str(exc), "raw_row": dict(row)}
            )

    if not aggregated:
        raise RealDataError("No valid monthly sales rows were produced.")
    if config.strict_unmatched and unmatched:
        raise RealDataError(
            f"Strict unmatched mode rejected {len(unmatched)} sales rows."
        )

    canonical_catalog = sorted(catalog_by_key.values(), key=lambda row: row["series_key"])
    canonical_sales = sorted(
        aggregated.values(), key=lambda row: (row["month"], row["series_key"])
    )

    catalog_path = destination / "canonical_catalog.csv"
    sales_path = destination / "canonical_monthly_sales.csv"
    unmatched_path = destination / "unmatched_series.csv"
    rejected_sales_path = destination / "rejected_sales_rows.json"
    rejected_vehicle_path = destination / "rejected_vehicle_rows.json"
    conflicts_path = destination / "mapping_conflicts.csv"

    write_csv_rows(
        catalog_path,
        [
            "series_key", "brand_name", "series_name", "segment", "body_type",
            "energy_type", "min_price", "max_price", "status", "launch_month",
        ],
        canonical_catalog,
    )
    write_csv_rows(
        sales_path,
        [
            "series_key", "brand_name", "series_name", "month", "sales",
            "sales_rank", "source", "data_version",
        ],
        canonical_sales,
    )
    write_csv_rows(
        unmatched_path,
        [
            "row_number", "raw_series_name", "canonical_series_name", "brand_name",
            "series_key", "reason",
        ],
        unmatched,
    )
    write_csv_rows(
        conflicts_path,
        ["source_series_name", "first_mapping", "conflicting_mapping"],
        mapping_conflicts,
    )
    write_json(rejected_sales_path, rejected_sales_rows)
    write_json(rejected_vehicle_path, rejected_vehicle_rows)

    source_files = {
        "sales": {
            "path": str(Path(sales_csv)),
            "sha256": sha256_file(sales_csv),
            "encoding": sales_encoding,
            "rows": len(sales_rows),
        },
        "mapping": {
            "path": str(Path(mapping_csv)),
            "sha256": sha256_file(mapping_csv),
            "encoding": mapping_encoding,
            "rows": len(mapping_rows),
        },
        "vehicles": {
            "path": str(Path(vehicles_csv)),
            "sha256": sha256_file(vehicles_csv),
            "encoding": vehicle_encoding,
            "rows": len(vehicle_rows),
        },
    }
    identity = {
        "schema_version": "autogfm-canonical-three-file-v1",
        "source_files": {name: value["sha256"] for name, value in source_files.items()},
        "config": asdict(config),
        "resolutions": {
            group: {name: asdict(value) for name, value in fields.items()}
            for group, fields in resolutions.items()
        },
        "canonical_sales_sha256": sha256_file(sales_path),
        "canonical_catalog_sha256": sha256_file(catalog_path),
    }
    dataset_id = stable_payload_id(identity)
    month_counts = Counter(row["month"] for row in canonical_sales)
    series_counts = Counter(row["series_key"] for row in canonical_sales)
    manifest = {
        **identity,
        "dataset_id": dataset_id,
        "source_files": source_files,
        "outputs": {
            "canonical_catalog": {"path": catalog_path.name, "sha256": sha256_file(catalog_path)},
            "canonical_monthly_sales": {"path": sales_path.name, "sha256": sha256_file(sales_path)},
            "unmatched_series": {"path": unmatched_path.name, "sha256": sha256_file(unmatched_path)},
            "mapping_conflicts": {"path": conflicts_path.name, "sha256": sha256_file(conflicts_path)},
        },
        "quality": {
            "canonical_series": len(canonical_catalog),
            "canonical_monthly_rows": len(canonical_sales),
            "unmatched_rows": len(unmatched),
            "rejected_sales_rows": len(rejected_sales_rows),
            "rejected_vehicle_rows": len(rejected_vehicle_rows),
            "mapping_conflicts": len(mapping_conflicts),
            "duplicate_monthly_keys_aggregated": duplicate_count,
            "month_min": min(month_counts),
            "month_max": max(month_counts),
            "min_months_per_series": min(series_counts.values()),
            "max_months_per_series": max(series_counts.values()),
        },
    }
    write_json(destination / "manifest.json", manifest)
    return CanonicalizationResult(destination, dataset_id, manifest)
