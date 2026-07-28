"""Shared schema helpers."""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Strict data contract with ORM serialization support."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        from_attributes=True,
    )


def normalize_entity_name(value: str) -> str:
    """Normalize English and Chinese entity names for deterministic matching."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[\s\-_/·•.]+", "", normalized)
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
    if not normalized:
        raise ValueError("Entity name becomes empty after normalization.")
    return normalized


def validate_month_start(value: date) -> date:
    """Require canonical monthly dates to use the first calendar day."""
    if value.day != 1:
        raise ValueError("month must be the first day of a calendar month.")
    return value
