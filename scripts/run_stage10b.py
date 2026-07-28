"""Run literature-informed Stage 10B on the bundled real market series."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogfm.pipeline.stage10b import run_stage10b_literature_market


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
    result = run_stage10b_literature_market(
        args.input,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"run_id={result.run_id}")
    print(f"promotion={result.manifest['promotion_decision']}")
    print(f"report={result.output_dir / 'STAGE10B_REPORT.html'}")


if __name__ == "__main__":
    main()
