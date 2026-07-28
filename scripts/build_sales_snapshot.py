"""Build leakage-safe monthly-sales train/validation/test files."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from autogfm.database import (
    create_database_engine,
    create_session_factory,
    session_scope,
)
from autogfm.snapshots import SnapshotSpec, build_monthly_sales_snapshot


def _month(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Dates must use YYYY-MM-01."
        ) from exc
    if parsed.day != 1:
        raise argparse.ArgumentTypeError("Dates must use the first day of a month.")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--train-end", required=True, type=_month)
    parser.add_argument("--validation-end", required=True, type=_month)
    parser.add_argument("--test-end", type=_month)
    parser.add_argument("--start-month", type=_month)
    parser.add_argument("--min-history-months", type=int, default=1)
    parser.add_argument("--require-contiguous-history", action="store_true")
    parser.add_argument("--format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    engine = create_database_engine()
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        result = build_monthly_sales_snapshot(
            session,
            args.output_dir,
            SnapshotSpec(
                source=args.source,
                data_version=args.data_version,
                train_end=args.train_end,
                validation_end=args.validation_end,
                test_end=args.test_end,
                start_month=args.start_month,
                min_history_months=args.min_history_months,
                require_contiguous_history=(
                    args.require_contiguous_history
                ),
                output_format=args.format,
            ),
            overwrite=args.overwrite,
        )

    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
