"""Run AutoGraphFM Stage 10A on the bundled real aggregate market data."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.pipeline.stage10a import run_stage10a_real_market


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/china_auto_market/ev_sales_monthly.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_stage10a_real_market(
        args.input,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"run_id={result.run_id}")
    print(f"report={result.output_dir / 'STAGE10A_REPORT.html'}")


if __name__ == "__main__":
    main()
