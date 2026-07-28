"""Import every ORM model so metadata and Alembic see all tables."""

from autogfm.database.models.brand import Brand
from autogfm.database.models.data_quality_result import DataQualityResult
from autogfm.database.models.ingestion_run import IngestionRun
from autogfm.database.models.monthly_sales import MonthlySales
from autogfm.database.models.monthly_series_feature import MonthlySeriesFeature
from autogfm.database.models.vehicle_alias import VehicleAlias
from autogfm.database.models.vehicle_series import VehicleSeries

__all__ = [
    "Brand",
    "DataQualityResult",
    "IngestionRun",
    "MonthlySales",
    "MonthlySeriesFeature",
    "VehicleAlias",
    "VehicleSeries",
]
