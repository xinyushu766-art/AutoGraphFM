"""Import one monthly-sales CSV into the audited relational store."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from autogfm.database import (
    create_database_engine,
    create_session_factory,
    session_scope,
)
from autogfm.ingestion import CsvImportConfig, import_monthly_sales_csv


def _column_override(value: str) -> tuple[str, str]:
    canonical, separator, source_column = value.partition("=")
    if not separator or not canonical or not source_column:
        raise argparse.ArgumentTypeError(
            "Column overrides must use CANONICAL=SOURCE_COLUMN."
        )
    return canonical, source_column


def _write_rejections(path: Path, report) -> None:
    fieldnames = ["row_number", "check_name", "message", "raw_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for rejected in report.rejected_rows:
            writer.writerow(
                {
                    "row_number": rejected.row_number,
                    "check_name": rejected.check_name,
                    "message": rejected.message,
                    "raw_json": json.dumps(
                        dict(rejected.raw),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--data-version", default="v1")
    parser.add_argument("--default-brand")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--code-version")
    parser.add_argument(
        "--column",
        action="append",
        default=[],
        type=_column_override,
        metavar="CANONICAL=SOURCE_COLUMN",
    )
    parser.add_argument(
        "--rejects",
        type=Path,
        help="Optional CSV path for rejected rows.",
    )
    args = parser.parse_args()

    engine = create_database_engine()
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        report = import_monthly_sales_csv(
            session,
            args.csv_path,
            CsvImportConfig(
                source=args.source,
                data_version=args.data_version,
                encoding=args.encoding,
                delimiter=args.delimiter,
                default_brand=args.default_brand,
                code_version=args.code_version,
                column_overrides=dict(args.column),
            ),
        )

    if args.rejects is not None and report.rejected_rows:
        _write_rejections(args.rejects, report)

    print(
        json.dumps(
            {
                "run_id": str(report.run_id),
                "status": report.status,
                "records_read": report.records_read,
                "records_written": report.records_written,
                "records_skipped": report.records_skipped,
                "records_rejected": report.records_rejected,
                "source_file_sha256": report.source_file_sha256,
                "resolved_columns": dict(report.resolved_columns),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if report.status == "succeeded":
        return 0
    if report.status == "partial":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
