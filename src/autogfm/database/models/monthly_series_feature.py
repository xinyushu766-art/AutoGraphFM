"""Monthly dynamic features for each vehicle series."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
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


class MonthlySeriesFeature(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Versioned monthly price, rating, and review features."""

    __tablename__ = "monthly_series_features"
    __table_args__ = (
        UniqueConstraint(
            "series_id",
            "month",
            "source",
            "feature_version",
            name="uq_monthly_features_series_month_source_version",
        ),
        CheckConstraint(
            "guide_price IS NULL OR guide_price >= 0",
            name="guide_price_nonnegative",
        ),
        CheckConstraint(
            "dealer_price IS NULL OR dealer_price >= 0",
            name="dealer_price_nonnegative",
        ),
        CheckConstraint(
            "rating IS NULL OR (rating >= 0 AND rating <= 5)",
            name="rating_range",
        ),
        CheckConstraint(
            "review_count IS NULL OR review_count >= 0",
            name="review_count_nonnegative",
        ),
        CheckConstraint(
            (
                "sentiment_mean IS NULL "
                "OR (sentiment_mean >= -1 AND sentiment_mean <= 1)"
            ),
            name="sentiment_range",
        ),
        CheckConstraint(
            (
                "positive_ratio IS NULL "
                "OR (positive_ratio >= 0 AND positive_ratio <= 1)"
            ),
            name="positive_ratio_range",
        ),
        CheckConstraint(
            "popularity_index IS NULL OR popularity_index >= 0",
            name="popularity_nonnegative",
        ),
        CheckConstraint(
            "months_since_launch IS NULL OR months_since_launch >= 0",
            name="months_since_launch_nonnegative",
        ),
        CheckConstraint(
            extract("day", column("month")) == 1,
            name="month_first_day",
        ),
        Index("ix_monthly_features_month_series", "month", "series_id"),
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
    guide_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    dealer_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    rating: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    review_count: Mapped[int | None] = mapped_column(Integer)
    sentiment_mean: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))
    positive_ratio: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))
    popularity_index: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    months_since_launch: Mapped[int | None] = mapped_column(Integer)
    is_on_sale: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    source: Mapped[str] = mapped_column(String(80), nullable=False)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    feature_version: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
        default="v1",
    )

    series: Mapped["VehicleSeries"] = relationship(
        back_populates="monthly_features"
    )
    ingestion_run: Mapped["IngestionRun | None"] = relationship(
        back_populates="feature_records"
    )
