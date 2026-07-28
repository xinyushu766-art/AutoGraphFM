"""Persisted data-quality checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autogfm.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from autogfm.database.models.ingestion_run import IngestionRun


class DataQualityResult(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Result of one quality check for one ingestion run."""

    __tablename__ = "data_quality_results"
    __table_args__ = (
        CheckConstraint(
            "severity IN ('info', 'warning', 'error')",
            name="severity_allowed",
        ),
        CheckConstraint(
            "failed_rows >= 0",
            name="failed_rows_nonnegative",
        ),
    )

    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    check_name: Mapped[str] = mapped_column(String(160), nullable=False)
    table_name: Mapped[str] = mapped_column(String(120), nullable=False)
    severity: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="error",
    )
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    failed_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    details: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    ingestion_run: Mapped["IngestionRun"] = relationship(
        back_populates="quality_results"
    )
