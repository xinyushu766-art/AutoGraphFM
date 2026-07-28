"""Small deterministic I/O helpers for real automotive CSV files."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class RealDataError(ValueError):
    """Raised when a real-data contract cannot be satisfied safely."""


def normalize_token(value: object) -> str:
    """Normalize headers and entity names for robust matching."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def stable_series_key(brand_name: str, series_name: str) -> str:
    """Return a human-readable, deterministic canonical key."""
    brand = normalize_token(brand_name) or "unknownbrand"
    series = normalize_token(series_name)
    if not series:
        raise RealDataError("series_name cannot be empty after normalization.")
    return f"{brand}::{series}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def stable_payload_id(payload: Mapping[str, Any], length: int = 16) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:length]


def read_csv_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]], str]:
    """Read a CSV with encoding and delimiter detection.

    UTF-8 is preferred. GB18030 is accepted because many Chinese automotive
    exports use it. The detected encoding is returned for the audit manifest.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    raw = source.read_bytes()
    decoded: str | None = None
    used_encoding = ""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            decoded = raw.decode(encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise RealDataError(f"Could not decode CSV: {source}")

    sample = decoded[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(decoded.splitlines(), dialect=dialect)
    if not reader.fieldnames:
        raise RealDataError(f"CSV has no header: {source}")
    fieldnames = [str(name).strip() for name in reader.fieldnames]
    rows: list[dict[str, str]] = []
    for raw_row in reader:
        normalized = {
            fieldnames[index]: str(raw_row.get(name, "") or "").strip()
            for index, name in enumerate(reader.fieldnames)
        }
        if any(value != "" for value in normalized.values()):
            rows.append(normalized)
    return fieldnames, rows, used_encoding


def write_csv_rows(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def parse_month(value: object) -> date:
    """Parse common monthly formats and return the first day of the month."""
    text = str(value or "").strip()
    if not text:
        raise RealDataError("month is empty")

    compact = re.sub(r"[^0-9]", "", text)
    candidates: list[tuple[str, str]] = [
        (text, "%Y-%m-%d"),
        (text, "%Y/%m/%d"),
        (text, "%Y-%m"),
        (text, "%Y/%m"),
        (text, "%Y年%m月"),
        (compact, "%Y%m%d"),
        (compact, "%Y%m"),
    ]
    for candidate, pattern in candidates:
        try:
            parsed = datetime.strptime(candidate, pattern).date()
            return date(parsed.year, parsed.month, 1)
        except ValueError:
            continue
    raise RealDataError(f"Unsupported month format: {text!r}")


def parse_float(value: object, *, allow_empty: bool = True) -> float | None:
    text = str(value or "").strip()
    if not text:
        if allow_empty:
            return None
        raise RealDataError("numeric value is empty")
    cleaned = text.replace(",", "").replace("￥", "").replace("¥", "")
    cleaned = cleaned.replace("万元", "").replace("万", "")
    cleaned = cleaned.replace("%", "")
    try:
        return float(cleaned)
    except ValueError as exc:
        raise RealDataError(f"Invalid numeric value: {text!r}") from exc


def parse_nonnegative_int(value: object) -> int:
    number = parse_float(value, allow_empty=False)
    assert number is not None
    if number < 0:
        raise RealDataError("sales must be non-negative")
    return int(round(number))
