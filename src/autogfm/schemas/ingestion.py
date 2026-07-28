"""Ingestion and data-quality contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from autogfm.schemas.common import ContractModel


class IngestionRunCreate(ContractModel):
    """Start one ingestion audit record."""

    source: str = Field(min_length=1, max_length=80)
    started_at: datetime
    finished_at: datetime | None = None
    status: Literal["running", "succeeded", "failed", "partial"] = "running"
    records_read: int = Field(default=0, ge=0)
    records_written: int = Field(default=0, ge=0)
    records_rejected: int = Field(default=0, ge=0)
    error_message: str | None = None
    code_version: str | None = Field(default=None, max_length=80)
    run_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "IngestionRunCreate":
        if (
            self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at cannot precede started_at.")
        return self


class DataQualityResultCreate(ContractModel):
    """Persist one data-quality check result."""

    run_id: UUID
    check_name: str = Field(min_length=1, max_length=160)
    table_name: str = Field(min_length=1, max_length=120)
    severity: Literal["info", "warning", "error"] = "error"
    passed: bool
    failed_rows: int = Field(default=0, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
