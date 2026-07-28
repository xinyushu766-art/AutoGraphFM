"""Data-ingestion audit records."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autogfm.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from autogfm.database.models.data_quality_result import DataQualityResult
    from autogfm.database.models.monthly_sales import MonthlySales
    from autogfm.database.models.monthly_series_feature import MonthlySeriesFeature


class IngestionRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One auditable import or crawling run."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'partial')",
            name="status_allowed",
        ),
        CheckConstraint(
            "records_read >= 0 AND records_written >= 0 AND records_rejected >= 0",
            name="record_counts_nonnegative",
        ),
        CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at",
            name="finished_after_started",
        ),
    )

    source: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    status: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="running",
    )
    records_read: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    records_written: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    records_rejected: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    code_version: Mapped[str | None] = mapped_column(String(80))
    run_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    sales_records: Mapped[list["MonthlySales"]] = relationship(
        back_populates="ingestion_run",
    )
    feature_records: Mapped[list["MonthlySeriesFeature"]] = relationship(
        back_populates="ingestion_run",
    )
    quality_results: Mapped[list["DataQualityResult"]] = relationship(
        back_populates="ingestion_run",
        cascade="all, delete-orphan",
    )
