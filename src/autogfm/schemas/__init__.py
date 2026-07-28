"""Validated input contracts for database writes."""

from autogfm.schemas.catalog import BrandCreate, VehicleAliasCreate, VehicleSeriesCreate
from autogfm.schemas.ingestion import DataQualityResultCreate, IngestionRunCreate
from autogfm.schemas.monthly import MonthlySalesCreate, MonthlySeriesFeatureCreate

__all__ = [
    "BrandCreate",
    "DataQualityResultCreate",
    "IngestionRunCreate",
    "MonthlySalesCreate",
    "MonthlySeriesFeatureCreate",
    "VehicleAliasCreate",
    "VehicleSeriesCreate",
]
