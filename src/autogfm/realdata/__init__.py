"""Real-data ingestion, canonicalization, and panel construction."""

from autogfm.realdata.adapters import (
    CanonicalizationConfig,
    CanonicalizationResult,
    canonicalize_three_files,
    resolve_column,
)
from autogfm.realdata.common import RealDataError
from autogfm.realdata.srni_car import (
    SRNI_CAR_PROJECT_URL,
    SRNI_CAR_SALES_DOWNLOAD_URL,
    SRNI_CAR_SALES_DOWNLOAD_URLS,
    SRNI_CAR_SALES_LEGACY_DOWNLOAD_URL,
    SrniCarAuthenticitySpec,
    SrniCarInspectionResult,
    SrniCarPreparationConfig,
    SrniCarPreparationResult,
    inspect_srni_car_sales,
    prepare_srni_car_sales,
)
from autogfm.realdata.kaggle_china_auto import (
    ChinaAutoPreparationResult,
    KAGGLE_DATASET_SLUG,
    KAGGLE_DATASET_URL,
    prepare_china_auto_sales,
)
from autogfm.realdata.lifecycle import LifecycleSpec, profile_series_lifecycle
from autogfm.realdata.panel import (
    PanelBuildResult,
    PanelData,
    PanelSpec,
    WindowSplit,
    build_monthly_panel,
    build_window_split,
    load_panel,
)

__all__ = [
    "CanonicalizationConfig",
    "CanonicalizationResult",
    "ChinaAutoPreparationResult",
    "KAGGLE_DATASET_SLUG",
    "KAGGLE_DATASET_URL",
    "LifecycleSpec",
    "PanelBuildResult",
    "PanelData",
    "PanelSpec",
    "RealDataError",
    "SRNI_CAR_PROJECT_URL",
    "SRNI_CAR_SALES_DOWNLOAD_URL",
    "SRNI_CAR_SALES_DOWNLOAD_URLS",
    "SRNI_CAR_SALES_LEGACY_DOWNLOAD_URL",
    "SrniCarAuthenticitySpec",
    "SrniCarInspectionResult",
    "SrniCarPreparationConfig",
    "SrniCarPreparationResult",
    "WindowSplit",
    "build_monthly_panel",
    "build_window_split",
    "canonicalize_three_files",
    "inspect_srni_car_sales",
    "prepare_srni_car_sales",
    "prepare_china_auto_sales",
    "profile_series_lifecycle",
    "load_panel",
    "resolve_column",
]
