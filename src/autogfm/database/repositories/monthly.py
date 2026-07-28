"""Monthly fact and feature repository."""

from __future__ import annotations

from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from autogfm.database.models import MonthlySales, MonthlySeriesFeature
from autogfm.schemas import MonthlySalesCreate, MonthlySeriesFeatureCreate


class MonthlyRepository:
    """Store and query versioned monthly records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_sales(
        self,
        payload: MonthlySalesCreate,
    ) -> MonthlySales:
        record = MonthlySales(**payload.model_dump())
        self.session.add(record)
        self.session.flush()
        return record

    def add_features(
        self,
        payload: MonthlySeriesFeatureCreate,
    ) -> MonthlySeriesFeature:
        record = MonthlySeriesFeature(**payload.model_dump())
        self.session.add(record)
        self.session.flush()
        return record

    def get_sales_record(
        self,
        series_id: UUID,
        month: date,
        source: str,
        data_version: str,
    ) -> MonthlySales | None:
        """Return one fact by its full immutable business key."""
        return self.session.scalar(
            select(MonthlySales).where(
                MonthlySales.series_id == series_id,
                MonthlySales.month == month,
                MonthlySales.source == source,
                MonthlySales.data_version == data_version,
            )
        )

    def list_sales(
        self,
        series_id: UUID,
        start_month: date | None = None,
        end_month: date | None = None,
    ) -> list[MonthlySales]:
        statement = (
            select(MonthlySales)
            .where(MonthlySales.series_id == series_id)
            .order_by(MonthlySales.month)
        )
        if start_month is not None:
            statement = statement.where(
                MonthlySales.month >= start_month
            )
        if end_month is not None:
            statement = statement.where(
                MonthlySales.month <= end_month
            )
        return list(self.session.scalars(statement))
