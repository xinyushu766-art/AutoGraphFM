"""Vehicle-series master data."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, Date, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autogfm.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from autogfm.database.models.brand import Brand
    from autogfm.database.models.monthly_sales import MonthlySales
    from autogfm.database.models.monthly_series_feature import MonthlySeriesFeature
    from autogfm.database.models.vehicle_alias import VehicleAlias


class VehicleSeries(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Forecasting node representing a vehicle series."""

    __tablename__ = "vehicle_series"
    __table_args__ = (
        UniqueConstraint(
            "brand_id",
            "normalized_name",
            name="uq_vehicle_series_brand_normalized_name",
        ),
        CheckConstraint(
            "status IN ('active', 'discontinued', 'upcoming', 'unknown')",
            name="status_allowed",
        ),
        CheckConstraint(
            (
                "discontinue_date IS NULL OR launch_date IS NULL "
                "OR discontinue_date >= launch_date"
            ),
            name="valid_lifecycle_dates",
        ),
    )

    brand_id: Mapped[UUID] = mapped_column(
        ForeignKey("brands.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    series_name: Mapped[str] = mapped_column(String(160), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(160), nullable=False)
    segment: Mapped[str | None] = mapped_column(String(80))
    body_type: Mapped[str | None] = mapped_column(String(80))
    energy_type: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="unknown",
    )
    launch_date: Mapped[date | None] = mapped_column(Date)
    discontinue_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="active",
    )

    brand: Mapped["Brand"] = relationship(back_populates="vehicle_series")
    aliases: Mapped[list["VehicleAlias"]] = relationship(
        back_populates="series",
        cascade="all, delete-orphan",
    )
    sales_records: Mapped[list["MonthlySales"]] = relationship(
        back_populates="series",
        cascade="all, delete-orphan",
    )
    monthly_features: Mapped[list["MonthlySeriesFeature"]] = relationship(
        back_populates="series",
        cascade="all, delete-orphan",
    )
