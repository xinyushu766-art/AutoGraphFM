"""Monthly vehicle-series sales fact table."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    column,
    extract,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autogfm.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from autogfm.database.models.ingestion_run import IngestionRun
    from autogfm.database.models.vehicle_series import VehicleSeries


class MonthlySales(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Monthly sales for one canonical vehicle series."""

    __tablename__ = "monthly_sales"
    __table_args__ = (
        UniqueConstraint(
            "series_id",
            "month",
            "source",
            "data_version",
            name="uq_monthly_sales_series_month_source_version",
        ),
        CheckConstraint("sales >= 0", name="sales_nonnegative"),
        CheckConstraint(
            "sales_rank IS NULL OR sales_rank > 0",
            name="sales_rank_positive",
        ),
        CheckConstraint(
            extract("day", column("month")) == 1,
            name="month_first_day",
        ),
        Index("ix_monthly_sales_month_series", "month", "series_id"),
    )

    series_id: Mapped[UUID] = mapped_column(
        ForeignKey("vehicle_series.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL"),
        index=True,
    )
    month: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    sales: Mapped[int] = mapped_column(Integer, nullable=False)
    sales_rank: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(80), nullable=False)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    data_version: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
        default="v1",
    )

    series: Mapped["VehicleSeries"] = relationship(back_populates="sales_records")
    ingestion_run: Mapped["IngestionRun | None"] = relationship(
        back_populates="sales_records"
    )
