"""Brand master data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autogfm.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from autogfm.database.models.vehicle_series import VehicleSeries


class Brand(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Automotive brand such as BYD, Toyota, or Tesla."""

    __tablename__ = "brands"

    brand_name: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
    )
    normalized_name: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        unique=True,
        index=True,
    )
    country: Mapped[str | None] = mapped_column(String(80))

    vehicle_series: Mapped[list["VehicleSeries"]] = relationship(
        back_populates="brand",
        cascade="all, delete-orphan",
    )
