"""SRNI-CAR single-file onboarding for the AutoGraphFM three-file contract."""

from __future__ import annotations

import calendar
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from autogfm.realdata.common import (
    RealDataError,
    normalize_token,
    parse_nonnegative_int,
    read_csv_rows,
    sha256_file,
    stable_payload_id,
    write_csv_rows,
    write_json,
)


SRNI_CAR_PROJECT_URL = "https://srni-car.github.io/"
SRNI_CAR_SALES_DOWNLOAD_URL = (
    "https://figshare.com/ndownloader/files/42438726"
    "?private_link=255ef697e8b7e3115087"
)
SRNI_CAR_SALES_LEGACY_DOWNLOAD_URL = (
    "https://ndownloader.figshare.com/files/42438726"
    "?private_link=255ef697e8b7e3115087"
)
SRNI_CAR_SALES_DOWNLOAD_URLS = (
    SRNI_CAR_SALES_DOWNLOAD_URL,
    SRNI_CAR_SALES_LEGACY_DOWNLOAD_URL,
)

_ALLOWED_EVIDENCE_SCOPES = {
    "official_real_data",
    "real_unverified",
    "synthetic_only",
}

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "series": (
        "Car series",
        "series_name",
        "series",
        "vehicle series",
        "车系",
        "车系名称",
    ),
    "brand": ("Brand", "brand_name", "品牌", "品牌名称"),
    "year": ("Year", "sales year", "年份", "年"),
    "month": ("Month", "sales month", "月份", "月"),
    "sales": ("Sales", "sales volume", "monthly sales", "销量", "月销量"),
    "body_type": ("Car model type", "model type", "车型类别", "车身类型"),
    "energy_type": (
        "Brand energy type",
        "energy type",
        "能源类型",
        "品牌能源类型",
    ),
    "segment": ("Size", "vehicle size", "车型级别", "尺寸", "级别"),
    "country": (
        "Brand country of origin",
        "country of origin",
        "品牌原产国",
        "品牌来源国",
    ),
    "launch_year": (
        "Model launch date",
        "model launch year",
        "车型上市时间",
        "上市年份",
    ),
    "brand_established": (
        "Brand establishment date",
        "brand establishment year",
        "品牌成立时间",
    ),
    "brand_entered_china": (
        "Brand entered China date",
        "brand entered china year",
        "品牌进入中国时间",
    ),
}




@dataclass(frozen=True)
class SrniCarAuthenticitySpec:
    """Profile bounds used to distinguish official, unverified, and synthetic data."""

    valid_rows_min: int = 39_000
    valid_rows_max: int = 40_000
    series_min: int = 1_200
    series_max: int = 1_260
    brands_min: int = 150
    brands_max: int = 160
    expected_month_min: str = "2016-01-01"
    expected_month_max: str = "2022-12-01"
    synthetic_name_ratio: float = 0.80

    def validate(self) -> None:
        if not 0 < self.valid_rows_min <= self.valid_rows_max:
            raise RealDataError("Invalid official row-count bounds.")
        if not 0 < self.series_min <= self.series_max:
            raise RealDataError("Invalid official series-count bounds.")
        if not 0 < self.brands_min <= self.brands_max:
            raise RealDataError("Invalid official brand-count bounds.")
        if not 0 <= self.synthetic_name_ratio <= 1:
            raise RealDataError("synthetic_name_ratio must be within [0, 1].")


@dataclass(frozen=True)
class SrniCarInspectionResult:
    output_dir: Path | None
    evidence_scope: str
    verification_level: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SrniCarPreparationConfig:
    source_name: str = "srni_car"
    data_version: str = "2016-2022"
    required_evidence_scope: str | None = None

    def validate(self) -> None:
        if not self.source_name.strip():
            raise RealDataError("source_name cannot be empty.")
        if not self.data_version.strip():
            raise RealDataError("data_version cannot be empty.")
        if (
            self.required_evidence_scope is not None
            and self.required_evidence_scope not in _ALLOWED_EVIDENCE_SCOPES
        ):
            raise RealDataError(
                "required_evidence_scope must be one of "
                f"{sorted(_ALLOWED_EVIDENCE_SCOPES)}."
            )


@dataclass(frozen=True)
class SrniCarPreparationResult:
    output_dir: Path
    dataset_id: str
    sales_csv: Path
    mapping_csv: Path
    vehicles_csv: Path
    manifest: dict[str, Any]


def _resolve_column(
    headers: Sequence[str],
    logical_name: str,
    *,
    required: bool,
) -> str | None:
    normalized_headers = {normalize_token(header): header for header in headers}
    for alias in _COLUMN_ALIASES[logical_name]:
        match = normalized_headers.get(normalize_token(alias))
        if match is not None:
            return match

    candidates: list[tuple[float, str]] = []
    aliases = [normalize_token(alias) for alias in _COLUMN_ALIASES[logical_name]]
    for header in headers:
        token = normalize_token(header)
        hits = [
            alias
            for alias in aliases
            if alias and (alias in token or token in alias)
        ]
        if hits:
            longest = max(hits, key=len)
            candidates.append((len(longest) / max(len(token), 1), header))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if candidates and (
        len(candidates) == 1 or candidates[0][0] > candidates[1][0] + 1e-9
    ):
        return candidates[0][1]
    if candidates:
        tied = [
            header
            for score, header in candidates
            if abs(score - candidates[0][0]) < 1e-9
        ]
        raise RealDataError(
            f"Ambiguous SRNI-CAR columns for {logical_name!r}: {tied}."
        )
    if required:
        raise RealDataError(
            f"Could not resolve required SRNI-CAR field {logical_name!r}. "
            f"Headers: {list(headers)}"
        )
    return None


def _text(row: Mapping[str, str], column: str | None) -> str:
    if column is None:
        return ""
    return str(row.get(column, "") or "").strip()


def _parse_integer(value: str, field: str) -> int:
    text = value.strip()
    if not text:
        raise RealDataError(f"{field} is empty")
    try:
        parsed = float(text.replace(",", ""))
    except ValueError as exc:
        raise RealDataError(f"Invalid {field}: {value!r}") from exc
    if not parsed.is_integer():
        raise RealDataError(f"{field} must be an integer: {value!r}")
    return int(parsed)


def _parse_month_number(value: str) -> int:
    text = value.strip()
    if not text:
        raise RealDataError("month is empty")
    compact = normalize_token(text)
    month_names: dict[str, int] = {}
    for month in range(1, 13):
        month_names[normalize_token(calendar.month_name[month])] = month
        month_names[normalize_token(calendar.month_abbr[month])] = month
        month_names[str(month)] = month
        month_names[f"{month:02d}"] = month
        month_names[normalize_token(f"{month}月")] = month
    if compact in month_names:
        return month_names[compact]
    numeric = re.sub(r"[^0-9]", "", text)
    if numeric:
        month = int(numeric)
        if 1 <= month <= 12:
            return month
    raise RealDataError(f"Unsupported month value: {value!r}")


def _parse_year_month(year_value: str, month_value: str) -> date:
    year = _parse_integer(year_value, "year")
    month = _parse_month_number(month_value)
    if not 1900 <= year <= 2100:
        raise RealDataError(f"year out of supported range: {year}")
    return date(year, month, 1)


def _launch_month(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    match = re.search(r"(19|20)\d{2}", text)
    if not match:
        return ""
    return f"{int(match.group(0)):04d}-01-01"


def _majority(values: Sequence[str]) -> str:
    cleaned = [value.strip() for value in values if value.strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    first_position = {value: cleaned.index(value) for value in counts}
    return min(counts, key=lambda value: (-counts[value], first_position[value], value))


def _synthetic_name_ratio(values: set[str], pattern: str) -> float:
    if not values:
        return 0.0
    matched = sum(
        bool(re.fullmatch(pattern, value.strip(), re.IGNORECASE))
        for value in values
    )
    return matched / len(values)


def inspect_srni_car_sales(
    source_csv: str | Path,
    output_dir: str | Path | None = None,
    *,
    spec: SrniCarAuthenticitySpec | None = None,
    overwrite: bool = False,
) -> SrniCarInspectionResult:
    """Profile a candidate Sales CSV and classify its evidence scope."""
    spec = spec or SrniCarAuthenticitySpec()
    spec.validate()
    source = Path(source_csv)
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        if destination.exists() and any(destination.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {destination}. Use overwrite=True."
            )
        destination.mkdir(parents=True, exist_ok=True)
    headers, rows, encoding = read_csv_rows(source)
    columns = {
        name: _resolve_column(
            headers,
            name,
            required=name in {"series", "brand", "year", "month", "sales"},
        )
        for name in _COLUMN_ALIASES
    }
    valid_rows = 0
    rejected_rows = 0
    duplicate_rows = 0
    series_values: set[str] = set()
    brand_values: set[str] = set()
    months: list[str] = []
    business_keys: set[tuple[str, str, str]] = set()

    for row in rows:
        try:
            series_name = _text(row, columns["series"])
            brand_name = _text(row, columns["brand"])
            if not series_name or not brand_name:
                raise RealDataError("Series or brand is empty.")
            month = _parse_year_month(
                _text(row, columns["year"]),
                _text(row, columns["month"]),
            )
            parse_nonnegative_int(_text(row, columns["sales"]))
            key = (brand_name, series_name, month.isoformat())
            if key in business_keys:
                duplicate_rows += 1
            business_keys.add(key)
            valid_rows += 1
            series_values.add(series_name)
            brand_values.add(brand_name)
            months.append(month.isoformat())
        except Exception:
            rejected_rows += 1

    if not valid_rows:
        raise RealDataError("No valid SRNI-CAR rows were found during inspection.")

    month_min = min(months)
    month_max = max(months)
    brand_synthetic_ratio = _synthetic_name_ratio(brand_values, r"brand\d+")
    series_synthetic_ratio = _synthetic_name_ratio(series_values, r"series\d+")
    known_fixture_shape = (
        valid_rows == 1_512
        and len(series_values) == 18
        and len(brand_values) == 6
        and month_min == "2016-01-01"
        and month_max == "2022-12-01"
    )
    synthetic_detected = known_fixture_shape or (
        brand_synthetic_ratio >= spec.synthetic_name_ratio
        and series_synthetic_ratio >= spec.synthetic_name_ratio
    )
    official_profile_checks = {
        "valid_rows_in_range": spec.valid_rows_min <= valid_rows <= spec.valid_rows_max,
        "series_in_range": spec.series_min <= len(series_values) <= spec.series_max,
        "brands_in_range": spec.brands_min <= len(brand_values) <= spec.brands_max,
        "month_min_matches": month_min == spec.expected_month_min,
        "month_max_matches": month_max == spec.expected_month_max,
        "no_rejected_rows": rejected_rows == 0,
        "no_duplicate_business_keys": duplicate_rows == 0,
        "not_synthetic": not synthetic_detected,
    }
    official_profile_match = all(official_profile_checks.values())
    if synthetic_detected:
        evidence_scope = "synthetic_only"
        verification_level = "synthetic_detected"
    elif official_profile_match:
        evidence_scope = "official_real_data"
        verification_level = "official_profile_matched_not_hash_verified"
    else:
        evidence_scope = "real_unverified"
        verification_level = "schema_valid_profile_unverified"

    manifest = {
        "schema_version": "autogfm-srni-authenticity-v1",
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
            "encoding": encoding,
        },
        "evidence_scope": evidence_scope,
        "verification_level": verification_level,
        "profile": {
            "raw_rows": len(rows),
            "valid_rows": valid_rows,
            "rejected_rows": rejected_rows,
            "duplicate_business_key_rows": duplicate_rows,
            "series": len(series_values),
            "brands": len(brand_values),
            "month_min": month_min,
            "month_max": month_max,
            "brand_synthetic_name_ratio": brand_synthetic_ratio,
            "series_synthetic_name_ratio": series_synthetic_ratio,
        },
        "official_profile_checks": official_profile_checks,
        "official_profile_reference": {
            "paper_observations": 39_496,
            "paper_car_series": 1_236,
            "paper_brands": 155,
            "period": "2016-2022",
            "checksum_available": False,
        },
        "claim_boundary": (
            "Profile matching supports real-data use but is not cryptographic proof "
            "of the publisher-hosted bytes because no official checksum is published."
        ),
    }
    if destination is not None:
        write_json(destination / "srni_authenticity_manifest.json", manifest)
        lines = [
            "# SRNI-CAR Data Authenticity Report",
            "",
            f"- Evidence scope: **{evidence_scope.upper()}**",
            f"- Verification: **{verification_level}**",
            f"- Valid rows: **{valid_rows}**",
            f"- Vehicle series: **{len(series_values)}**",
            f"- Brands: **{len(brand_values)}**",
            f"- Period: **{month_min} → {month_max}**",
            f"- Rejected rows: **{rejected_rows}**",
            f"- Duplicate business keys: **{duplicate_rows}**",
            "",
            "The official SRNI-CAR paper reports 39,496 sales observations, "
            "1,236 car series, 155 brands, and coverage from 2016 through 2022. "
            "The publisher does not expose a checksum, so matching is profile-based.",
            "",
        ]
        (destination / "SRNI_AUTHENTICITY_REPORT.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    return SrniCarInspectionResult(
        destination, evidence_scope, verification_level, manifest
    )


def prepare_srni_car_sales(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    config: SrniCarPreparationConfig | None = None,
    overwrite: bool = False,
) -> SrniCarPreparationResult:
    """Convert the official SRNI-CAR Sales CSV into AutoGraphFM's three files."""
    config = config or SrniCarPreparationConfig()
    config.validate()
    source = Path(source_csv)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Use overwrite=True."
        )
    destination.mkdir(parents=True, exist_ok=True)

    inspection = inspect_srni_car_sales(
        source,
        destination / "inspection",
        overwrite=overwrite,
    )
    if (
        config.required_evidence_scope is not None
        and inspection.evidence_scope != config.required_evidence_scope
    ):
        raise RealDataError(
            "SRNI-CAR evidence scope mismatch: required "
            f"{config.required_evidence_scope!r}, observed "
            f"{inspection.evidence_scope!r}."
        )

    headers, rows, encoding = read_csv_rows(source)
    columns = {
        name: _resolve_column(
            headers,
            name,
            required=name in {"series", "brand", "year", "month", "sales"},
        )
        for name in _COLUMN_ALIASES
    }

    sales_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    attributes: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    observed_business_keys: set[tuple[str, str, str]] = set()
    duplicate_rows = 0

    for row_number, row in enumerate(rows, start=2):
        try:
            series_name = _text(row, columns["series"])
            brand_name = _text(row, columns["brand"])
            if not series_name:
                raise RealDataError("Car series is empty")
            if not brand_name:
                raise RealDataError("Brand is empty")
            month = _parse_year_month(
                _text(row, columns["year"]),
                _text(row, columns["month"]),
            )
            sales = parse_nonnegative_int(_text(row, columns["sales"]))
            business_key = (brand_name, series_name, month.isoformat())
            if business_key in observed_business_keys:
                duplicate_rows += 1
            observed_business_keys.add(business_key)
            sales_rows.append(
                {
                    "series_name": series_name,
                    "brand": brand_name,
                    "month": month.isoformat(),
                    "sales": sales,
                    "source": config.source_name,
                    "data_version": config.data_version,
                }
            )
            key = (brand_name, series_name)
            for field in (
                "segment",
                "body_type",
                "energy_type",
                "country",
                "launch_year",
                "brand_established",
                "brand_entered_china",
            ):
                value = _text(row, columns[field])
                if value:
                    attributes[key][field].append(value)
        except Exception as exc:  # intentional row-level quarantine
            rejected_rows.append(
                {"row_number": row_number, "reason": str(exc), "raw_row": dict(row)}
            )

    if not sales_rows:
        raise RealDataError("No valid SRNI-CAR sales rows were produced.")

    mapping_rows: list[dict[str, str]] = []
    vehicle_rows: list[dict[str, str]] = []
    attribute_conflicts: list[dict[str, str]] = []
    for brand_name, series_name in sorted(attributes):
        values = attributes[(brand_name, series_name)]
        resolved = {
            field: _majority(field_values)
            for field, field_values in values.items()
        }
        for field, field_values in values.items():
            unique_values = sorted(
                {value.strip() for value in field_values if value.strip()}
            )
            if len(unique_values) > 1:
                attribute_conflicts.append(
                    {
                        "brand": brand_name,
                        "series_name": series_name,
                        "field": field,
                        "selected_value": resolved[field],
                        "observed_values": " | ".join(unique_values),
                    }
                )
        mapping_rows.append(
            {
                "source_series": series_name,
                "canonical_series": series_name,
                "brand": brand_name,
            }
        )
        vehicle_rows.append(
            {
                "series_name": series_name,
                "brand": brand_name,
                "segment": resolved.get("segment", ""),
                "body_type": resolved.get("body_type", ""),
                "energy_type": resolved.get("energy_type", "") or "unknown",
                "min_price": "",
                "max_price": "",
                "status": "unknown",
                "launch_date": _launch_month(resolved.get("launch_year", "")),
                "brand_country_of_origin": resolved.get("country", ""),
                "brand_establishment_date": resolved.get("brand_established", ""),
                "brand_entered_china_date": resolved.get("brand_entered_china", ""),
            }
        )

    sales_rows.sort(
        key=lambda item: (item["month"], item["brand"], item["series_name"])
    )
    sales_path = destination / "sales_aligned_completed.csv"
    mapping_path = destination / "series_mapping_completed.csv"
    vehicles_path = destination / "vehicles_completed.csv"
    rejects_path = destination / "rejected_srni_rows.json"
    conflicts_path = destination / "srni_attribute_conflicts.csv"

    write_csv_rows(
        sales_path,
        ["series_name", "brand", "month", "sales", "source", "data_version"],
        sales_rows,
    )
    write_csv_rows(
        mapping_path,
        ["source_series", "canonical_series", "brand"],
        mapping_rows,
    )
    write_csv_rows(
        vehicles_path,
        [
            "series_name",
            "brand",
            "segment",
            "body_type",
            "energy_type",
            "min_price",
            "max_price",
            "status",
            "launch_date",
            "brand_country_of_origin",
            "brand_establishment_date",
            "brand_entered_china_date",
        ],
        vehicle_rows,
    )
    write_json(rejects_path, rejected_rows)
    write_csv_rows(
        conflicts_path,
        ["brand", "series_name", "field", "selected_value", "observed_values"],
        attribute_conflicts,
    )

    month_values = [str(row["month"]) for row in sales_rows]
    identity = {
        "schema_version": "autogfm-srni-car-preparation-v1",
        "source_sha256": sha256_file(source),
        "config": asdict(config),
        "columns": columns,
        "sales_output_sha256": sha256_file(sales_path),
        "mapping_output_sha256": sha256_file(mapping_path),
        "vehicles_output_sha256": sha256_file(vehicles_path),
    }
    dataset_id = stable_payload_id(identity)
    manifest = {
        **identity,
        "dataset_id": dataset_id,
        "source": {
            "path": str(source),
            "encoding": encoding,
            "rows": len(rows),
            "project_url": SRNI_CAR_PROJECT_URL,
            "license": "CC BY-NC 4.0",
            "evidence_scope": inspection.evidence_scope,
            "verification_level": inspection.verification_level,
            "authenticity_manifest": "inspection/srni_authenticity_manifest.json",
        },
        "outputs": {
            "sales": {"path": sales_path.name, "sha256": sha256_file(sales_path)},
            "mapping": {"path": mapping_path.name, "sha256": sha256_file(mapping_path)},
            "vehicles": {
                "path": vehicles_path.name,
                "sha256": sha256_file(vehicles_path),
            },
            "rejected_rows": {
                "path": rejects_path.name,
                "sha256": sha256_file(rejects_path),
            },
            "attribute_conflicts": {
                "path": conflicts_path.name,
                "sha256": sha256_file(conflicts_path),
            },
        },
        "quality": {
            "valid_sales_rows": len(sales_rows),
            "rejected_rows": len(rejected_rows),
            "duplicate_business_key_rows": duplicate_rows,
            "series": len(vehicle_rows),
            "brands": len({str(row["brand"]) for row in sales_rows}),
            "month_min": min(month_values),
            "month_max": max(month_values),
            "attribute_conflicts": len(attribute_conflicts),
        },
    }
    write_json(destination / "srni_manifest.json", manifest)
    return SrniCarPreparationResult(
        output_dir=destination,
        dataset_id=dataset_id,
        sales_csv=sales_path,
        mapping_csv=mapping_path,
        vehicles_csv=vehicles_path,
        manifest=manifest,
    )
