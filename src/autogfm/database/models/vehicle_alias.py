"""Source-specific vehicle aliases used for entity resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from autogfm.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from autogfm.database.models.vehicle_series import VehicleSeries


class VehicleAlias(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Map one source spelling to a canonical vehicle series."""

    __tablename__ = "vehicle_aliases"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "normalized_name",
            name="uq_vehicle_aliases_source_normalized_name",
        ),
        CheckConstraint(
            "match_score >= 0 AND match_score <= 1",
            name="match_score_range",
        ),
        CheckConstraint(
            "match_method IN ('exact', 'rule', 'fuzzy', 'manual')",
            name="match_method_allowed",
        ),
        CheckConstraint(
            "review_status IN ('pending', 'approved', 'rejected')",
            name="review_status_allowed",
        ),
    )

    source: Mapped[str] = mapped_column(String(80), nullable=False)
    source_name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
    )
    series_id: Mapped[UUID] = mapped_column(
        ForeignKey("vehicle_series.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    match_method: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="manual",
    )
    match_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
    )
    review_status: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="approved",
    )

    series: Mapped["VehicleSeries"] = relationship(back_populates="aliases")
