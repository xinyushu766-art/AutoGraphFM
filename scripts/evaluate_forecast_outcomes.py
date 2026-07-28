"""Evaluate closed forecast months against canonical realized sales."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.monitoring import OutcomeSpec, evaluate_forecast_outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-csv", type=Path, required=True)
    parser.add_argument("--actual-canonical-sales", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wape-warn", type=float, default=20.0)
    parser.add_argument("--wape-block", type=float, default=35.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = evaluate_forecast_outcomes(
        args.forecast_csv,
        args.actual_canonical_sales,
        args.output_dir,
        OutcomeSpec(
            wape_warn_percent=args.wape_warn,
            wape_block_percent=args.wape_block,
        ),
        overwrite=args.overwrite,
    )
    print(f"outcome_id={result.outcome_id}")
    print(f"status={result.status}")
    print(f"action={result.action}")


if __name__ == "__main__":
    main()
