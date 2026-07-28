"""Monthly sales and feature contracts."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import Field, field_validator

from autogfm.schemas.common import ContractModel, validate_month_start


class MonthlySalesCreate(ContractModel):
    """Validated monthly sales fact."""

    series_id: UUID
    run_id: UUID | None = None
    month: date
    sales: int = Field(ge=0)
    sales_rank: int | None = Field(default=None, gt=0)
    source: str = Field(min_length=1, max_length=80)
    collected_at: datetime
    data_version: str = Field(default="v1", min_length=1, max_length=80)

    @field_validator("month")
    @classmethod
    def month_must_start_on_first_day(cls, value: date) -> date:
        return validate_month_start(value)


class MonthlySeriesFeatureCreate(ContractModel):
    """Validated monthly dynamic feature record."""

    series_id: UUID
    run_id: UUID | None = None
    month: date
    guide_price: Decimal | None = Field(default=None, ge=0)
    dealer_price: Decimal | None = Field(default=None, ge=0)
    rating: Decimal | None = Field(default=None, ge=0, le=5)
    review_count: int | None = Field(default=None, ge=0)
    sentiment_mean: Decimal | None = Field(default=None, ge=-1, le=1)
    positive_ratio: Decimal | None = Field(default=None, ge=0, le=1)
    popularity_index: Decimal | None = Field(default=None, ge=0)
    months_since_launch: int | None = Field(default=None, ge=0)
    is_on_sale: bool = True
    source: str = Field(min_length=1, max_length=80)
    collected_at: datetime
    feature_version: str = Field(default="v1", min_length=1, max_length=80)

    @field_validator("month")
    @classmethod
    def month_must_start_on_first_day(cls, value: date) -> date:
        return validate_month_start(value)
