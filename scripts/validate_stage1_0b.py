"""Run a self-contained Stage 1.0B SQLite acceptance workflow."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from autogfm.database import (
    create_database_engine,
    create_session_factory,
    session_scope,
)
from autogfm.database.base import Base
from autogfm.database.settings import DatabaseSettings
from autogfm.ingestion import CsvImportConfig, import_monthly_sales_csv
from autogfm.snapshots import SnapshotSpec, build_monthly_sales_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage1_0b_validation"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    database_path = output_dir / "acceptance.db"
    settings = DatabaseSettings(
        database_url=f"sqlite+pysqlite:///{database_path.as_posix()}"
    )
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)

    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "examples/stage1_0b/monthly_sales_sample.csv"
    config = CsvImportConfig(
        source="stage1_0b_validation",
        data_version="sample-v1",
        code_version="v0.4.0",
    )

    with session_scope(factory) as session:
        first = import_monthly_sales_csv(session, source, config)
    with session_scope(factory) as session:
        second = import_monthly_sales_csv(session, source, config)
    with session_scope(factory) as session:
        snapshot = build_monthly_sales_snapshot(
            session,
            output_dir / "snapshot",
            SnapshotSpec(
                source=config.source,
                data_version=config.data_version,
                train_end=date(2024, 4, 1),
                validation_end=date(2024, 6, 1),
                test_end=date(2024, 8, 1),
                min_history_months=8,
                require_contiguous_history=True,
            ),
        )

    checks = {
        "first_import_succeeded": first.status == "succeeded",
        "first_import_wrote_16": first.records_written == 16,
        "second_import_succeeded": second.status == "succeeded",
        "second_import_skipped_16": second.records_skipped == 16,
        "train_rows_8": snapshot.manifest["files"]["train"]["rows"] == 8,
        "validation_rows_4": (
            snapshot.manifest["files"]["validation"]["rows"] == 4
        ),
        "test_rows_4": snapshot.manifest["files"]["test"]["rows"] == 4,
    }
    passed = all(checks.values())
    report = {
        "status": "passed" if passed else "failed",
        "checks": checks,
        "first_import": {
            "run_id": str(first.run_id),
            "records_written": first.records_written,
            "records_rejected": first.records_rejected,
        },
        "second_import": {
            "run_id": str(second.run_id),
            "records_skipped": second.records_skipped,
            "records_rejected": second.records_rejected,
        },
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_manifest": str(
            (snapshot.output_dir / "manifest.json").relative_to(output_dir)
        ),
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
