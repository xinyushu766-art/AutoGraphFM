"""Catalog and entity-resolution contracts."""

from __future__ import annotations

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from autogfm.schemas.common import ContractModel, normalize_entity_name


class BrandCreate(ContractModel):
    """Create or resolve a canonical brand."""

    brand_name: str = Field(min_length=1, max_length=120)
    normalized_name: str | None = Field(default=None, max_length=120)
    country: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def fill_normalized_name(self) -> "BrandCreate":
        if self.normalized_name is None:
            self.normalized_name = normalize_entity_name(self.brand_name)
        return self


class VehicleSeriesCreate(ContractModel):
    """Create one vehicle-series forecast node."""

    brand_id: UUID
    series_name: str = Field(min_length=1, max_length=160)
    normalized_name: str | None = Field(default=None, max_length=160)
    segment: str | None = Field(default=None, max_length=80)
    body_type: str | None = Field(default=None, max_length=80)
    energy_type: str = Field(default="unknown", max_length=40)
    launch_date: date | None = None
    discontinue_date: date | None = None
    status: Literal["active", "discontinued", "upcoming", "unknown"] = "active"

    @model_validator(mode="after")
    def validate_contract(self) -> "VehicleSeriesCreate":
        if self.normalized_name is None:
            self.normalized_name = normalize_entity_name(self.series_name)
        if (
            self.launch_date is not None
            and self.discontinue_date is not None
            and self.discontinue_date < self.launch_date
        ):
            raise ValueError("discontinue_date cannot precede launch_date.")
        return self


class VehicleAliasCreate(ContractModel):
    """Map a source-specific spelling to a vehicle series."""

    source: str = Field(min_length=1, max_length=80)
    source_name: str = Field(min_length=1, max_length=200)
    normalized_name: str | None = Field(default=None, max_length=200)
    series_id: UUID
    match_method: Literal["exact", "rule", "fuzzy", "manual"] = "manual"
    match_score: float = Field(default=1.0, ge=0.0, le=1.0)
    review_status: Literal["pending", "approved", "rejected"] = "approved"

    @model_validator(mode="after")
    def fill_normalized_name(self) -> "VehicleAliasCreate":
        if self.normalized_name is None:
            self.normalized_name = normalize_entity_name(self.source_name)
        return self
