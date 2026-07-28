"""Catalog and entity-resolution repository."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from autogfm.database.models import Brand, VehicleAlias, VehicleSeries
from autogfm.schemas import BrandCreate, VehicleAliasCreate, VehicleSeriesCreate
from autogfm.schemas.common import normalize_entity_name


class CatalogRepository:
    """Encapsulate canonical entity writes and lookups."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_brand_by_normalized_name(
        self,
        normalized_name: str,
    ) -> Brand | None:
        return self.session.scalar(
            select(Brand).where(
                Brand.normalized_name == normalized_name
            )
        )

    def get_or_create_brand(
        self,
        payload: BrandCreate,
    ) -> Brand:
        existing = self.get_brand_by_normalized_name(
            payload.normalized_name or normalize_entity_name(payload.brand_name)
        )
        if existing is not None:
            return existing

        brand = Brand(**payload.model_dump())
        self.session.add(brand)
        self.session.flush()
        return brand

    def get_series(
        self,
        brand_id: UUID,
        normalized_name: str,
    ) -> VehicleSeries | None:
        return self.session.scalar(
            select(VehicleSeries).where(
                VehicleSeries.brand_id == brand_id,
                VehicleSeries.normalized_name == normalized_name,
            )
        )

    def get_or_create_series(
        self,
        payload: VehicleSeriesCreate,
    ) -> VehicleSeries:
        normalized = payload.normalized_name or normalize_entity_name(
            payload.series_name
        )
        existing = self.get_series(payload.brand_id, normalized)
        if existing is not None:
            return existing

        series = VehicleSeries(**payload.model_dump())
        self.session.add(series)
        self.session.flush()
        return series

    def add_alias(
        self,
        payload: VehicleAliasCreate,
    ) -> VehicleAlias:
        alias = VehicleAlias(**payload.model_dump())
        self.session.add(alias)
        self.session.flush()
        return alias

    def resolve_alias(
        self,
        source: str,
        source_name: str,
    ) -> VehicleSeries | None:
        normalized = normalize_entity_name(source_name)
        statement = (
            select(VehicleSeries)
            .join(VehicleAlias)
            .where(
                VehicleAlias.source == source,
                VehicleAlias.normalized_name == normalized,
                VehicleAlias.review_status == "approved",
            )
        )
        return self.session.scalar(statement)
