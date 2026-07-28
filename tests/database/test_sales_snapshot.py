from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from autogfm.database.repositories import CatalogRepository, MonthlyRepository
from autogfm.schemas import BrandCreate, MonthlySalesCreate, VehicleSeriesCreate
from autogfm.snapshots import (
    SnapshotError,
    SnapshotSpec,
    build_monthly_sales_snapshot,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_series(
    session: Session,
    name: str,
    months: list[date],
) -> None:
    catalog = CatalogRepository(session)
    monthly = MonthlyRepository(session)
    brand = catalog.get_or_create_brand(BrandCreate(brand_name="BYD"))
    series = catalog.get_or_create_series(
        VehicleSeriesCreate(brand_id=brand.id, series_name=name)
    )
    for index, month in enumerate(months, start=1):
        monthly.add_sales(
            MonthlySalesCreate(
                series_id=series.id,
                month=month,
                sales=1000 + index,
                source="demo",
                collected_at=datetime(2025, 1, 1, tzinfo=UTC),
                data_version="v1",
            )
        )


def test_snapshot_is_time_split_and_content_addressed(
    session: Session,
    tmp_path: Path,
) -> None:
    months = [date(2024, month, 1) for month in range(1, 9)]
    _add_series(session, "Series A", months)
    _add_series(session, "Series B", months)
    spec = SnapshotSpec(
        source="demo",
        data_version="v1",
        train_end=date(2024, 4, 1),
        validation_end=date(2024, 6, 1),
        test_end=date(2024, 8, 1),
        min_history_months=8,
        require_contiguous_history=True,
    )

    first = build_monthly_sales_snapshot(
        session,
        tmp_path / "snapshot_a",
        spec,
    )
    second = build_monthly_sales_snapshot(
        session,
        tmp_path / "snapshot_b",
        spec,
    )

    assert first.snapshot_id == second.snapshot_id
    assert first.manifest["files"]["train"]["rows"] == 8
    assert first.manifest["files"]["validation"]["rows"] == 4
    assert first.manifest["files"]["test"]["rows"] == 4
    assert first.manifest["files"]["train"]["month_range"] == {
        "min": "2024-01-01",
        "max": "2024-04-01",
    }
    for split in ("train", "validation", "test"):
        path_a = first.output_dir / f"{split}.csv"
        path_b = second.output_dir / f"{split}.csv"
        assert _sha256(path_a) == _sha256(path_b)
        assert _sha256(path_a) == first.manifest["files"][split]["sha256"]


def test_history_filters_exclude_gapped_series(
    session: Session,
    tmp_path: Path,
) -> None:
    full = [date(2024, month, 1) for month in range(1, 7)]
    gapped = [
        date(2024, 1, 1),
        date(2024, 2, 1),
        date(2024, 4, 1),
        date(2024, 5, 1),
        date(2024, 6, 1),
    ]
    _add_series(session, "Complete", full)
    _add_series(session, "Gapped", gapped)

    result = build_monthly_sales_snapshot(
        session,
        tmp_path / "snapshot",
        SnapshotSpec(
            source="demo",
            data_version="v1",
            train_end=date(2024, 2, 1),
            validation_end=date(2024, 4, 1),
            test_end=date(2024, 6, 1),
            min_history_months=5,
            require_contiguous_history=True,
        ),
    )

    assert result.manifest["included_series"] == 1
    assert len(result.manifest["excluded_series"]) == 1
    assert result.manifest["excluded_series"][0]["series_name"] == "Gapped"
    assert result.manifest["excluded_series"][0]["reasons"] == [
        "non_contiguous_history"
    ]


def test_snapshot_refuses_empty_or_invalid_splits(
    session: Session,
    tmp_path: Path,
) -> None:
    _add_series(
        session,
        "Series",
        [date(2024, month, 1) for month in range(1, 4)],
    )
    with pytest.raises(SnapshotError, match="validation split is empty"):
        build_monthly_sales_snapshot(
            session,
            tmp_path / "snapshot",
            SnapshotSpec(
                source="demo",
                data_version="v1",
                train_end=date(2024, 3, 1),
                validation_end=date(2024, 4, 1),
                test_end=date(2024, 5, 1),
            ),
        )
