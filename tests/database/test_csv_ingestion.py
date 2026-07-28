from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from autogfm.database.models import (
    Brand,
    DataQualityResult,
    IngestionRun,
    MonthlySales,
    VehicleAlias,
    VehicleSeries,
)
from autogfm.ingestion import (
    CsvImportConfig,
    import_monthly_sales_csv,
    parse_month,
    parse_nonnegative_int,
)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8-sig")


def test_common_month_and_integer_formats() -> None:
    assert parse_month("2025-7").isoformat() == "2025-07-01"
    assert parse_month("2025年07月").isoformat() == "2025-07-01"
    assert parse_month("2025年07月31日").isoformat() == "2025-07-01"
    assert parse_month("2025/07/31").isoformat() == "2025-07-01"
    assert parse_nonnegative_int("12,345", "sales") == 12345
    assert parse_nonnegative_int("120.0", "sales") == 120


def test_bilingual_csv_import_is_idempotent(
    session: Session,
    tmp_path: Path,
) -> None:
    source = tmp_path / "sales.csv"
    _write(
        source,
        "品牌,车系,年月,销量,能源类型\n"
        "BYD,秦PLUS新能源,2025-01,24000,phev\n"
        "BYD,秦PLUS新能源,2025-02,21000,phev\n"
        "BYD,海豚,2025-01,18000,bev\n",
    )
    config = CsvImportConfig(
        source="pacificauto",
        data_version="2025-01",
        collected_at=datetime(2025, 3, 1, tzinfo=UTC),
        code_version="test",
    )

    first = import_monthly_sales_csv(session, source, config)
    second = import_monthly_sales_csv(session, source, config)

    assert first.status == "succeeded"
    assert first.records_written == 3
    assert first.records_skipped == 0
    assert second.status == "succeeded"
    assert second.records_written == 0
    assert second.records_skipped == 3
    assert session.scalar(select(func.count()).select_from(Brand)) == 1
    assert session.scalar(select(func.count()).select_from(VehicleSeries)) == 2
    assert session.scalar(select(func.count()).select_from(VehicleAlias)) == 2
    assert session.scalar(select(func.count()).select_from(MonthlySales)) == 3
    assert session.scalar(select(func.count()).select_from(IngestionRun)) == 2


def test_invalid_and_duplicate_rows_produce_partial_audit(
    session: Session,
    tmp_path: Path,
) -> None:
    source = tmp_path / "sales_bad.csv"
    _write(
        source,
        "brand,series,month,sales\n"
        "BYD,Qin PLUS,2025-01,100\n"
        "BYD,Qin PLUS,2025-02,-1\n"
        "BYD,Qin PLUS,2025-01,100\n",
    )

    report = import_monthly_sales_csv(
        session,
        source,
        CsvImportConfig(source="demo", data_version="v1"),
    )

    assert report.status == "partial"
    assert report.records_read == 3
    assert report.records_written == 1
    assert report.records_rejected == 2
    checks = {row.check_name for row in report.rejected_rows}
    assert "row_contract" in checks
    assert "duplicate_business_key" in checks

    run = session.get(IngestionRun, report.run_id)
    assert run is not None
    assert run.status == "partial"
    assert run.records_read == 3
    assert run.records_written == 1
    assert run.records_rejected == 2

    quality = list(
        session.scalars(
            select(DataQualityResult).where(
                DataQualityResult.run_id == report.run_id
            )
        )
    )
    assert len(quality) == 4
    assert any(
        result.check_name == "duplicate_business_keys_in_file"
        and not result.passed
        for result in quality
    )


def test_missing_required_columns_is_failed_not_silent(
    session: Session,
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing.csv"
    _write(source, "brand,series,sales\nBYD,Qin PLUS,100\n")

    report = import_monthly_sales_csv(
        session,
        source,
        CsvImportConfig(source="demo"),
    )

    assert report.status == "failed"
    assert report.records_read == 0
    assert report.records_written == 0
    assert report.records_rejected == 1
    assert report.rejected_rows[0].row_number == 1
    assert "month" in report.rejected_rows[0].message
